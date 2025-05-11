import os
import json
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import argparse
from tqdm import tqdm
import random

def download_image(url, save_path):
    """Download an image from URL and save it to the specified path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def prepare_dataset(csv_path, output_dir, split_ratio=0.9):
    """
    Prepare the dataset for fine-tuning LLaVA-Next.
    
    Args:
        csv_path: Path to the CSV file containing the dataset
        output_dir: Directory to save the processed dataset
        split_ratio: Train/validation split ratio (default: 0.9)
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Process each row
    train_data = []
    val_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):
        # Extract data
        tweet_text = row.get("Tweet Text", "")
        image_url = row.get("Image URL", "")
        token_name = row.get("Token Name", "")
        ticker = row.get("Ticker", "")
        
        if not image_url or pd.isna(image_url):
            continue
            
        # Download and save the image
        image_filename = f"image_{idx}.jpg"
        image_path = os.path.join(output_dir, "images", image_filename)
        
        if download_image(image_url, image_path):
            # Create simple data format
            data_item = {
                "text": f"Tweet text: {tweet_text}" if tweet_text and not pd.isna(tweet_text) else "",
                "image_path": os.path.join("images", image_filename),
                "response": json.dumps({"tokenName": token_name, "ticker": ticker})
            }
            
            # Add to train or validation set
            if random.random() < split_ratio:
                train_data.append(data_item)
            else:
                val_data.append(data_item)
    
    # Save the processed data
    with open(os.path.join(output_dir, "train", "data.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_dir, "val", "data.json"), "w") as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Dataset prepared: {len(train_data)} training samples, {len(val_data)} validation samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for LLaVA-Next fine-tuning")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/validation split ratio")
    
    args = parser.parse_args()
    prepare_dataset(args.csv_path, args.output_dir, args.split_ratio)