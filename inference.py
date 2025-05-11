# Create an updated inference script
import torch
import argparse
import requests
from PIL import Image
from io import BytesIO
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
import json
import re

def load_model(model_path, use_lora=False, base_model="llava-hf/llava-v1.6-mistral-7b-hf", quantize=False):
    """Load the fine-tuned model."""
    # Load processor
    processor = LlavaNextProcessor.from_pretrained(model_path if not use_lora else base_model)
    
    # Load model with quantization if specified
    if quantize:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            base_model if use_lora else model_path,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto"
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            base_model if use_lora else model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Load LoRA weights if using LoRA
    if use_lora:
        model = PeftModel.from_pretrained(model, model_path)
    
    return model, processor

def extract_token_info_from_text(text):
    """Extract token name and ticker from generated text using regex."""
    # Try to find token name pattern
    token_name_match = re.search(r'token\s*name[:\s]+([A-Za-z0-9\s]+)', text, re.IGNORECASE)
    ticker_match = re.search(r'ticker[:\s]+([A-Z0-9]+)', text, re.IGNORECASE)
    
    token_name = token_name_match.group(1).strip() if token_name_match else None
    ticker = ticker_match.group(1).strip() if ticker_match else None
    
    # If not found, try to extract any capitalized words as token name and ticker
    if not token_name:
        words = re.findall(r'\b[A-Z][A-Z0-9]+\b', text)
        if words:
            if not ticker and len(words) > 0:
                ticker = words[0]
            if not token_name and len(words) > 1:
                token_name = ' '.join(words[:-1])
    
    # If still not found, use default values
    if not token_name:
        token_name = "UNKNOWN TOKEN"
    if not ticker:
        ticker = "UNKN"
    
    return {"tokenName": token_name, "ticker": ticker}

def generate_meme_coin(model, processor, image_url, tweet_text, max_new_tokens=100):
    """Generate meme coin name and ticker from image and tweet."""
    try:
        # Download image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Create a better prompt
        prompt = f"""<image>
Based on the image and tweet, generate a meme coin name and ticker.
Tweet: {tweet_text}

Generate a JSON response with the following format:
{{
  "tokenName": "MEME COIN NAME",
  "ticker": "TICKER"
}}

JSON response:
"""
        
        # Process inputs
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
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
                try:
                    result = json.loads(json_str)
                    # Validate the result has the expected fields
                    if "tokenName" not in result or "ticker" not in result:
                        # Extract from text if JSON is invalid
                        result = extract_token_info_from_text(generated_text)
                    return result
                except json.JSONDecodeError:
                    # Extract from text if JSON is invalid
                    result = extract_token_info_from_text(generated_text)
                    return result
            else:
                # Extract from text if no JSON found
                result = extract_token_info_from_text(generated_text)
                return result
        
        except Exception as e:
            print(f"Error extracting JSON: {e}")
            # Extract from text as fallback
            result = extract_token_info_from_text(generated_text)
            return result
    
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
