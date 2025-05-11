import argparse
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel

def test_model(model_path, image_url, tweet_text, use_lora=False, base_model="llava-hf/llava-v1.6-mistral-7b-hf", quantize=False):
    """Test the model with a sample image and tweet."""
    print("Loading model...")
    
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
    
    print("Model loaded successfully!")
    
    # Download image
    print(f"Downloading image from {image_url}...")
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    print("Image downloaded successfully!")
    
    # Create conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Tweet text: {tweet_text}" if tweet_text else ""},
                {"type": "image"}
            ]
        }
    ]
    
    # Apply chat template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process inputs
    print("Processing inputs...")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    print("Generating response...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )
    
    # Decode response
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract the assistant's response
    assistant_response = generated_text.split("[/INST]")[-1].strip()
    
    print("\nGenerated Response:")
    print(assistant_response)
    
    return assistant_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--image_url", type=str, required=True, help="URL of the image")
    parser.add_argument("--tweet_text", type=str, default="", help="Tweet text")
    parser.add_argument("--use_lora", action="store_true", help="Whether the model is fine-tuned with LoRA")
    parser.add_argument("--base_model", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="Base model path (for LoRA)")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    
    args = parser.parse_args()
    test_model(
        args.model_path,
        args.image_url,
        args.tweet_text,
        args.use_lora,
        args.base_model,
        args.quantize
    )
