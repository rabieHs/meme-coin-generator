import os
import json
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from inference import load_model, generate_meme_coin

# Define request model
class MemeRequest(BaseModel):
    image_url: str
    tweet_text: Optional[str] = ""

# Define response model
class MemeResponse(BaseModel):
    tokenName: str
    ticker: str

# Initialize FastAPI app
app = FastAPI(title="Meme Coin Generator API")

# Global variables for model and processor
model = None
processor = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, processor
    
    model_path = os.environ.get("MODEL_PATH", "fine_tuned_model")
    use_lora = os.environ.get("USE_LORA", "false").lower() == "true"
    base_model = os.environ.get("BASE_MODEL", "llava-hf/llava-v1.6-mistral-7b-hf")
    quantize = os.environ.get("QUANTIZE", "true").lower() == "true"
    
    try:
        model, processor = load_model(
            model_path=model_path,
            use_lora=use_lora,
            base_model=base_model,
            quantize=quantize
        )
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Meme Coin Generator API is running"}

@app.post("/generate", response_model=MemeResponse)
async def generate(request: MemeRequest):
    """Generate meme coin name and ticker."""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate meme coin
        result = generate_meme_coin(
            model=model,
            processor=processor,
            image_url=request.image_url,
            tweet_text=request.tweet_text,
            max_new_tokens=100
        )
        
        # Parse the result
        try:
            # Try to parse as JSON
            if isinstance(result, str):
                result_json = json.loads(result)
            else:
                result_json = result
                
            # Validate the response
            if "tokenName" not in result_json or "ticker" not in result_json:
                # Try to extract from the response if it's not in the expected format
                if "error" in result_json:
                    raise HTTPException(status_code=500, detail=result_json["error"])
                else:
                    raise ValueError("Response does not contain required fields")
            
            return MemeResponse(
                tokenName=result_json["tokenName"],
                ticker=result_json["ticker"]
            )
        except json.JSONDecodeError:
            # If not valid JSON, try to extract the information
            raise HTTPException(status_code=500, detail="Model did not return valid JSON")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=port)
