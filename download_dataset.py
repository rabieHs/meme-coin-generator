import pandas as pd
import argparse
import os

def download_dataset(sheet_url, output_csv):
    """
    Download dataset from Google Sheets.
    
    Args:
        sheet_url: URL of the Google Sheet
        output_csv: Path to save the CSV file
    """
    try:
        # Extract the sheet ID from the URL
        if "docs.google.com/spreadsheets/d/" in sheet_url:
            sheet_id = sheet_url.split("/d/")[1].split("/")[0]
        else:
            sheet_id = sheet_url
        
        # Create the export URL
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        # Read the CSV data
        df = pd.read_csv(export_url)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        
        print(f"Dataset downloaded successfully with {len(df)} rows")
        print(f"Saved to {output_csv}")
        
        return True
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from Google Sheets")
    parser.add_argument("--sheet_url", type=str, required=True, help="URL of the Google Sheet")
    parser.add_argument("--output_csv", type=str, default="dataset.csv", help="Path to save the CSV file")
    
    args = parser.parse_args()
    download_dataset(args.sheet_url, args.output_csv)
