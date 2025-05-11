
import os
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import re
from PIL import Image
from io import BytesIO
import requests
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel

# Load environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/fine_tuned_model")
USE_LORA = os.environ.get("USE_LORA", "true").lower() == "true"
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/llava-v1.6-mistral-7b-hf")
QUANTIZE = os.environ.get("QUANTIZE", "true").lower() == "true"
PORT = int(os.environ.get("PORT", 8000))

# Initialize FastAPI app
app = FastAPI(title="Meme Coin Generator API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
class ImageUrlRequest(BaseModel):
    image_url: str
    tweet_text: Optional[str] = ""

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

# Load model and processor
@app.on_event("startup")
async def startup_event():
    global model, processor
    
    # Load processor
    processor = LlavaNextProcessor.from_pretrained(MODEL_PATH if not USE_LORA else BASE_MODEL)
    
    # Load model with quantization if specified
    if QUANTIZE:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            BASE_MODEL if USE_LORA else MODEL_PATH,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto"
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            BASE_MODEL if USE_LORA else MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Load LoRA weights if using LoRA
    if USE_LORA:
        model = PeftModel.from_pretrained(model, MODEL_PATH)
    
    print("Model loaded successfully!")

# Define API endpoints
@app.post("/generate")
async def generate_from_url(request: ImageUrlRequest):
    """Generate meme coin name and ticker from image URL and tweet text."""
    try:
        # Download image
        response = requests.get(request.image_url, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Create a better prompt
        prompt = f"""<image>
Based on the image and tweet, generate a meme coin name and ticker.
Tweet: {request.tweet_text}

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
                max_new_tokens=100,
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/upload")
async def generate_from_upload(
    file: UploadFile = File(...),
    tweet_text: str = Form("")
):
    """Generate meme coin name and ticker from uploaded image and tweet text."""
    try:
        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
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
                max_new_tokens=100,
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "ok"}

# Run the API server
if __name__ == "__main__":
    uvicorn.run("api_server_improved:app", host="0.0.0.0", port=PORT, log_level="info")
