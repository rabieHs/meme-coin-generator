# Comprehensive Guide: Meme Coin Generator with LLaVA-Next

This guide provides step-by-step instructions for setting up, fine-tuning, and deploying the LLaVA-Next (7B Mistral) model for generating meme coin names and tickers from images and tweets.

## Table of Contents

1. [Setting Up RunPod](#1-setting-up-runpod)
2. [Downloading and Preparing the Dataset](#2-downloading-and-preparing-the-dataset)
3. [Fine-tuning the Model](#3-fine-tuning-the-model)
4. [Testing the Fine-tuned Model](#4-testing-the-fine-tuned-model)
5. [Deploying the API Server](#5-deploying-the-api-server)
6. [Using the API](#6-using-the-api)
7. [Troubleshooting](#7-troubleshooting)

## 1. Setting Up RunPod

### 1.1. Create a RunPod Account

1. Go to [RunPod.io](https://www.runpod.io/) and create an account
2. Add a payment method to your account

### 1.2. Launch a GPU Instance

1. Go to the RunPod dashboard
2. Click on "Deploy" to create a new pod
3. Select a GPU with at least 24GB VRAM (e.g., A10, RTX 3090, RTX 4090, A100)
4. Choose the "PyTorch" template
5. Set the disk size to at least 50GB
6. Deploy the pod

### 1.3. Set Up the Environment

1. Connect to the pod using SSH or the web terminal
2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/meme-coin-generator.git
   cd meme-coin-generator
   ```

3. Make the setup script executable and run it:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

## 2. Downloading and Preparing the Dataset

### 2.1. Download the Dataset from Google Sheets

1. Use the provided script to download the dataset:
   ```bash
   python download_dataset.py --sheet_url "https://docs.google.com/spreadsheets/d/1I6e4T9CdLsquIHBZsNlZsoNgyZPSsobF58D-jkiGXLk/edit?usp=sharing" --output_csv dataset.csv
   ```

### 2.2. Prepare the Dataset for Fine-tuning

1. Run the dataset preparation script:
   ```bash
   python prepare_dataset.py --csv_path dataset.csv --output_dir data
   ```

   This will:
   - Download images from the URLs in the dataset
   - Format the data for fine-tuning
   - Split the data into training and validation sets

2. Verify the dataset preparation:
   ```bash
   ls -la data/images
   cat data/train/conversations.json | head -n 20
   ```

## 3. Fine-tuning the Model

### 3.1. Download the Base Model

The fine-tuning script will automatically download the base model, but you can pre-download it to avoid issues:

```bash
python -c "from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration; processor = LlavaNextProcessor.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf'); model = LlavaNextForConditionalGeneration.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf', torch_dtype='auto')"
```

### 3.2. Fine-tune the Model

Run the fine-tuning script with LoRA and quantization to reduce memory requirements:

```bash
python finetune.py \
  --model_name llava-hf/llava-v1.6-mistral-7b-hf \
  --data_dir data \
  --output_dir fine_tuned_model \
  --batch_size 1 \
  --epochs 3 \
  --learning_rate 2e-5 \
  --gradient_accumulation_steps 4 \
  --eval_steps 50 \
  --save_steps 100 \
  --use_lora \
  --quantize
```

This will:
- Load the base model
- Apply LoRA for parameter-efficient fine-tuning
- Use 4-bit quantization to reduce memory usage
- Fine-tune the model on your dataset
- Save the fine-tuned model to the `fine_tuned_model` directory

The fine-tuning process may take several hours depending on the size of your dataset and the GPU you're using.

## 4. Testing the Fine-tuned Model

### 4.1. Test with a Sample Image and Tweet

Use the test script to verify that the fine-tuned model works correctly:

```bash
python test_model.py \
  --model_path fine_tuned_model \
  --image_url "https://pump.mypinata.cloud/ipfs/QmNzT34PBLpgB9GUCkcebkQa1yt3vnXg8bKCiiVmbrGuyo?img-width=800&img-dpr=2&img-onerror=redirect" \
  --tweet_text "Time is the true currency. — Elon Musk" \
  --use_lora \
  --quantize
```

This should generate a meme coin name and ticker in JSON format.

## 5. Deploying the API Server

### 5.1. Start the API Server

Start the API server with the fine-tuned model:

```bash
export MODEL_PATH="fine_tuned_model"
export USE_LORA="true"
export BASE_MODEL="llava-hf/llava-v1.6-mistral-7b-hf"
export QUANTIZE="true"
export PORT=8000

python api_server.py
```

The API server will be available at `http://your-runpod-ip:8000`.

### 5.2. Make the API Server Persistent

To keep the API server running even after you disconnect from the RunPod instance, you can use `tmux`:

```bash
# Install tmux
apt-get update && apt-get install -y tmux

# Create a new tmux session
tmux new -s api_server

# Inside the tmux session, start the API server
export MODEL_PATH="fine_tuned_model"
export USE_LORA="true"
export BASE_MODEL="llava-hf/llava-v1.6-mistral-7b-hf"
export QUANTIZE="true"
export PORT=8000

python api_server.py
```

To detach from the tmux session, press `Ctrl+B` followed by `D`. You can reattach to the session later with:

```bash
tmux attach -t api_server
```

## 6. Using the API

### 6.1. Test the API with the Sample Script

Use the sample script to test the API:

```bash
python sample_usage.py \
  --api_url "http://your-runpod-ip:8000" \
  --image_url "https://pump.mypinata.cloud/ipfs/QmNzT34PBLpgB9GUCkcebkQa1yt3vnXg8bKCiiVmbrGuyo?img-width=800&img-dpr=2&img-onerror=redirect" \
  --tweet_text "Time is the true currency. — Elon Musk"
```

This should return a JSON response with the generated meme coin name and ticker:

```json
{
  "tokenName": "TRUE CURRENCY",
  "ticker": "TIME"
}
```

### 6.2. Integrate with Your Application

To integrate the API with your application, send a POST request to the `/generate` endpoint with the following JSON payload:

```json
{
  "image_url": "https://example.com/image.jpg",
  "tweet_text": "Sample tweet text"
}
```

The API will return a JSON response with the generated meme coin name and ticker:

```json
{
  "tokenName": "Generated Token Name",
  "ticker": "TICKER"
}
```

## 7. Troubleshooting

### 7.1. Out of Memory Errors

If you encounter out of memory errors during fine-tuning or inference:

1. Use a GPU with more VRAM
2. Enable quantization with the `--quantize` flag
3. Use LoRA with the `--use_lora` flag
4. Reduce the batch size with the `--batch_size` flag
5. Increase gradient accumulation steps with the `--gradient_accumulation_steps` flag

### 7.2. Model Loading Issues

If you encounter issues loading the model:

1. Check that you have enough disk space
2. Verify that the model path is correct
3. Make sure you're using the correct flags (`--use_lora`, `--base_model`, etc.)

### 7.3. API Server Issues

If the API server doesn't start or crashes:

1. Check the logs for error messages
2. Verify that the model is loaded correctly
3. Make sure the environment variables are set correctly
4. Check that the port is not already in use

### 7.4. Output Format Issues

If the model doesn't generate the expected JSON format:

1. Check the fine-tuning dataset to make sure the format is correct
2. Verify that the model is trained to generate JSON output
3. Try adjusting the prompt template

## Conclusion

You now have a fully functional meme coin generator using the LLaVA-Next (7B Mistral) model. The model takes an image and tweet text as input and generates a meme coin name and ticker in JSON format.

For any issues or questions, please refer to the documentation or open an issue on the GitHub repository.
