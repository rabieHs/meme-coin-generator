
import argparse
import requests
import json

def generate_from_url(api_url, image_url, tweet_text):
    """Generate meme coin name and ticker from image URL and tweet text."""
    # Prepare request
    url = f"{api_url}/generate"
    payload = {
        "image_url": image_url,
        "tweet_text": tweet_text
    }
    
    print(f"Sending request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    # Send request
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        # Check response
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API request failed with status code {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def main(args):
    # Generate meme coin
    result = generate_from_url(args.api_url, args.image_url, args.tweet_text)
    
    # Print result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Meme Coin Generator API")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8000", help="API URL")
    parser.add_argument("--image_url", type=str, required=True, help="URL of the image")
    parser.add_argument("--tweet_text", type=str, default="", help="Tweet text")
    
    args = parser.parse_args()
    main(args)
