import pandas as pd
import json
import argparse

def convert_csv_to_json(csv_path, json_path):
    """
    Convert CSV file to JSON format.
    
    Args:
        csv_path: Path to the CSV file
        json_path: Path to save the JSON file
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Convert to list of dictionaries
    data = []
    for _, row in df.iterrows():
        item = {
            "tweet_text": row.get("Tweet Text", "") if pd.notna(row.get("Tweet Text", "")) else "",
            "image_url": row.get("Image URL", "") if pd.notna(row.get("Image URL", "")) else "",
            "token_name": row.get("Token Name", "") if pd.notna(row.get("Token Name", "")) else "",
            "ticker": row.get("Ticker", "") if pd.notna(row.get("Ticker", "")) else ""
        }
        data.append(item)
    
    # Save as JSON
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Converted {len(data)} rows from CSV to JSON")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to JSON")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--json_path", type=str, required=True, help="Path to save the JSON file")
    
    args = parser.parse_args()
    convert_csv_to_json(args.csv_path, args.json_path)
