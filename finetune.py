import os
import json
import torch
import argparse
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
import warnings

# Check if flash-attn is available
try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    warnings.warn("flash-attn is not installed. Training will proceed without it, but may be slower.")

def load_dataset(data_dir):
    """Load the dataset for fine-tuning."""
    # Load train and validation data
    train_file = os.path.join(data_dir, "train", "data.json")
    val_file = os.path.join(data_dir, "val", "data.json")
    
    with open(train_file, "r") as f:
        train_data = json.load(f)
    
    with open(val_file, "r") as f:
        val_data = json.load(f)
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset

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
    
    # Load the dataset
    train_dataset, val_dataset = load_dataset(args.data_dir)
    
    # Define preprocessing function
    def preprocess_function(examples):
        # Process each example individually to avoid padding issues
        processed_examples = {
            "input_ids": [],
            "attention_mask": [],
            "pixel_values": []
        }
        
        for i in range(len(examples["text"])):
            # Load image
            try:
                image_path = examples["image_path"][i]
                full_path = os.path.join(args.data_dir, image_path)
                if os.path.exists(full_path):
                    img = Image.open(full_path).convert("RGB")
                else:
                    # Use a blank image if path is invalid
                    print(f"Warning: Image not found at {full_path}")
                    img = Image.new('RGB', (224, 224), color='white')
            except Exception as e:
                print(f"Error loading image {examples['image_path'][i]}: {e}")
                # Use a blank image if there's an error
                img = Image.new('RGB', (224, 224), color='white')
            
            # Process single example
            inputs = processor(
                text=examples["text"][i],
                images=img,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )
            
            # Add to processed examples
            processed_examples["input_ids"].append(inputs["input_ids"][0])
            processed_examples["attention_mask"].append(inputs["attention_mask"][0])
            processed_examples["pixel_values"].append(inputs["pixel_values"][0])
        
        # Stack tensors
        processed_examples["input_ids"] = torch.stack(processed_examples["input_ids"])
        processed_examples["attention_mask"] = torch.stack(processed_examples["attention_mask"])
        processed_examples["pixel_values"] = torch.stack(processed_examples["pixel_values"])
        
        # Add labels for training
        processed_examples["labels"] = processed_examples["input_ids"].clone()
        
        return processed_examples
    
    # Preprocess the datasets
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1,  # Process one example at a time
        remove_columns=["text", "image_path", "response"]
    )
    
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1,  # Process one example at a time
        remove_columns=["text", "image_path", "response"]
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
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
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
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