import torch
import argparse
import requests
import warnings
import json
from PIL import Image
from io import BytesIO
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel

# Check if flash-attn is available
try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    warnings.warn("flash-attn is not installed. Inference will proceed without it, but may be slower.")

def load_model(model_path, use_lora=False, base_model="llava-hf/llava-v1.6-mistral-7b-hf", quantize=False):
    """Load the fine-tuned model."""
    # Load processor
    processor = LlavaNextProcessor.from_pretrained(model_path if not use_lora else base_model)
    
    # Load model with quantization if specified
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto"
    }
    
    # Add flash attention if available
    if HAS_FLASH_ATTN:
        model_kwargs["use_flash_attention_2"] = True
    
    if quantize:
        model_kwargs["load_in_4bit"] = True
        model = LlavaNextForConditionalGeneration.from_pretrained(
            base_model if use_lora else model_path,
            **model_kwargs
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            base_model if use_lora else model_path,
            **model_kwargs
        )
    
    # Load LoRA weights if using LoRA
    if use_lora:
        model = PeftModel.from_pretrained(model, model_path)
    
    return model, processor

def generate_meme_coin(model, processor, image_url, tweet_text, max_new_tokens=100):
    """Generate meme coin name and ticker from image and tweet."""
    try:
        # Download image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Process inputs
        inputs = processor(
            text=f"Tweet text: {tweet_text}" if tweet_text else "",
            images=image,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        # Decode response
        generated_text = processor.decode(output[0], skip_special_tokens=True)
        
        # Try to extract JSON from the response
        try:
            # Look for JSON-like structure
            start_idx = generated_text.find("{")
            end_idx = generated_text.rfind("}") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = generated_text[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                return {"error": "No JSON found in response", "raw_response": generated_text}
        
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in response", "raw_response": generated_text}
    
    except Exception as e:
        return {"error": str(e)}

def main(args):
    # Load model and processor
    model, processor = load_model(
        args.model_path, 
        use_lora=args.use_lora, 
        base_model=args.base_model,
        quantize=args.quantize
    )
    
    # Generate meme coin
    result = generate_meme_coin(
        model, 
        processor, 
        args.image_url, 
        args.tweet_text, 
        max_new_tokens=args.max_new_tokens
    )
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate meme coin name and ticker")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--image_url", type=str, required=True, help="URL of the image")
    parser.add_argument("--tweet_text", type=str, default="", help="Tweet text")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--use_lora", action="store_true", help="Whether the model is fine-tuned with LoRA")
    parser.add_argument("--base_model", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="Base model path (for LoRA)")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    
    args = parser.parse_args()
    main(args)