import requests
import json
import argparse

def generate_meme_coin(api_url, image_url, tweet_text=""):
    """
    Generate a meme coin name and ticker using the API.
    
    Args:
        api_url: URL of the API endpoint
        image_url: URL of the image
        tweet_text: Tweet text (optional)
    
    Returns:
        JSON response with tokenName and ticker
    """
    # Prepare request data
    data = {
        "image_url": image_url,
        "tweet_text": tweet_text
    }
    
    # Send request to API
    try:
        response = requests.post(f"{api_url}/generate", json=data)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"API Error: {error_detail}")
            except:
                print(f"Status code: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
        return None

def main(args):
    # Generate meme coin
    result = generate_meme_coin(args.api_url, args.image_url, args.tweet_text)
    
    # Print result
    if result:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate meme coin using the API")
    parser.add_argument("--api_url", type=str, required=True, help="URL of the API endpoint")
    parser.add_argument("--image_url", type=str, required=True, help="URL of the image")
    parser.add_argument("--tweet_text", type=str, default="", help="Tweet text")
    
    args = parser.parse_args()
    main(args)
