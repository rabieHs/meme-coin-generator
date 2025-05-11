import os
import json
import torch
import argparse
import warnings
from PIL import Image
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from tqdm import tqdm

# Check if flash-attn is available
try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    warnings.warn("flash-attn is not installed. Training will proceed without it, but may be slower.")

def load_dataset(data_dir, processor):
    """Load and process the dataset for fine-tuning."""

    def process_conversation(conversation, images_dir):
        """Process a single conversation."""
        # Extract user message
        user_message = conversation[0]
        assistant_message = conversation[1]

        # Get image path if present
        image_path = None
        text_prompt = ""

        for content in user_message["content"]:
            if content["type"] == "image":
                image_path = os.path.join(images_dir, content["image_path"].split("/")[-1])
            elif content["type"] == "text":
                text_prompt += content["text"]

        # Get assistant response (expected to be JSON)
        response = assistant_message["content"]

        return {
            "image_path": image_path,
            "text_prompt": text_prompt,
            "response": response
        }

    # Load train and validation data
    train_file = os.path.join(data_dir, "train", "conversations.json")
    val_file = os.path.join(data_dir, "val", "conversations.json")
    images_dir = os.path.join(data_dir, "images")

    with open(train_file, "r") as f:
        train_data = json.load(f)

    with open(val_file, "r") as f:
        val_data = json.load(f)

    # Process the data
    train_processed = [process_conversation(conv, images_dir) for conv in tqdm(train_data, desc="Processing train data")]
    val_processed = [process_conversation(conv, images_dir) for conv in tqdm(val_data, desc="Processing val data")]

    # Create datasets
    train_dataset = Dataset.from_list(train_processed)
    val_dataset = Dataset.from_list(val_processed)

    # Tokenize the datasets
    def tokenize_function(examples):
        images = [Image.open(path).convert("RGB") for path in examples["image_path"]]

        # Create conversation format
        conversations = []
        for i in range(len(examples["text_prompt"])):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": examples["text_prompt"][i]},
                        {"type": "image"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": examples["response"][i]
                }
            ]
            conversations.append(conversation)

        # Apply chat template
        prompts = [processor.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]

        # Tokenize
        tokenized = processor(images=images, text=prompts, padding="max_length", truncation=True, return_tensors="pt")

        return tokenized

    # Apply tokenization
    train_tokenized = train_dataset.map(tokenize_function, batched=True, batch_size=8)
    val_tokenized = val_dataset.map(tokenize_function, batched=True, batch_size=8)

    return train_tokenized, val_tokenized

def main(args):
    # Load model and processor
    processor = LlavaNextProcessor.from_pretrained(args.model_name)

    # Load model with quantization if specified
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto"
    }

    # Add flash attention if available
    if HAS_FLASH_ATTN:
        model_kwargs["use_flash_attention_2"] = True

    if args.quantize:
        model_kwargs["load_in_4bit"] = True
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_name,
            **model_kwargs
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_name,
            **model_kwargs
        )

    # Configure LoRA
    if args.use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load and process the dataset
    train_dataset, val_dataset = load_dataset(args.data_dir, processor)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="none"
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=processor.tokenizer,
        mlm=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"Fine-tuning completed. Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA-Next model")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="Model name or path")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")

    args = parser.parse_args()
    main(args)
