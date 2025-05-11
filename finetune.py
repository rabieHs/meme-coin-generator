import os
import json
import torch
import argparse
from PIL import Image
from transformers import (
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    TrainingArguments,
    Trainer
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
    
    # Create a custom dataset class
    class LlavaDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, processor, data_dir):
            self.dataset = dataset
            self.processor = processor
            self.data_dir = data_dir
            
            # Prepare the prompts
            self.prompts = []
            self.image_paths = []
            
            for item in self.dataset:
                # Create prompt
                prompt = f"<image>\n{item['text']}\n"
                self.prompts.append(prompt)
                
                # Get image path
                image_path = os.path.join(self.data_dir, item["image_path"])
                self.image_paths.append(image_path)
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            # Get prompt and image path
            prompt = self.prompts[idx]
            image_path = self.image_paths[idx]
            
            # Load image
            try:
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB")
                else:
                    # Use a blank image if path is invalid
                    print(f"Warning: Image not found at {image_path}")
                    image = Image.new('RGB', (224, 224), color='white')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Use a blank image if there's an error
                image = Image.new('RGB', (224, 224), color='white')
            
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Remove batch dimension
            for k, v in inputs.items():
                inputs[k] = v.squeeze(0)
            
            # Add labels for training
            inputs["labels"] = inputs["input_ids"].clone()
            
            return inputs
    
    # Create datasets
    train_dataset_processed = LlavaDataset(train_dataset, processor, args.data_dir)
    val_dataset_processed = LlavaDataset(val_dataset, processor, args.data_dir)
    
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
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=True,
        report_to="none",
        remove_unused_columns=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_processed,
        eval_dataset=val_dataset_processed
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