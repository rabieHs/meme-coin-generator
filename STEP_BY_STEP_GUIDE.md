# Step-by-Step Guide: Fine-tuning and Deploying LLaVA-Next on RunPod

This guide provides detailed instructions for fine-tuning the LLaVA-Next (7B Mistral) model on RunPod and deploying it for meme coin generation.

> **Note**: The setup has been modified to handle cases where flash-attn installation fails. The model will still work without flash-attn, but may be slightly slower.

## 1. Setting Up RunPod

1. Create a RunPod account at [RunPod.io](https://www.runpod.io/)
2. Launch a GPU instance with at least 24GB VRAM (A10, RTX 3090, or A100)
3. Select PyTorch template with at least 50GB disk space

## 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/memcoin_generator.git
cd memcoin_generator

# Make scripts executable
chmod +x make_executable.sh
./make_executable.sh

# Set up the environment
./setup.sh
```

## 3. Dataset Preparation

```bash
# Download dataset from Google Sheets
python download_dataset.py --sheet_url "https://docs.google.com/spreadsheets/d/1I6e4T9CdLsquIHBZsNlZsoNgyZPSsobF58D-jkiGXLk/edit?usp=sharing" --output_csv dataset.csv

# Prepare dataset for fine-tuning
python prepare_dataset.py --csv_path dataset.csv --output_dir data
```

## 4. Fine-tuning the Model

```bash
# Fine-tune the model with LoRA and quantization
python finetune.py \
  --model_name llava-hf/llava-v1.6-mistral-7b-hf \
  --data_dir data \
  --output_dir fine_tuned_model \
  --batch_size 1 \
  --epochs 3 \
  --learning_rate 2e-5 \
  --gradient_accumulation_steps 4 \
  --use_lora \
  --quantize
```

## 5. Testing the Fine-tuned Model

```bash
# Test with a sample image and tweet
python test_model.py \
  --model_path fine_tuned_model \
  --image_url "https://pump.mypinata.cloud/ipfs/QmNzT34PBLpgB9GUCkcebkQa1yt3vnXg8bKCiiVmbrGuyo?img-width=800&img-dpr=2&img-onerror=redirect" \
  --tweet_text "Time is the true currency. — Elon Musk" \
  --use_lora \
  --quantize
```

## 6. Deploying the API Server

```bash
# Set environment variables
export MODEL_PATH="fine_tuned_model"
export USE_LORA="true"
export BASE_MODEL="llava-hf/llava-v1.6-mistral-7b-hf"
export QUANTIZE="true"
export PORT=8000

# Start the API server
python api_server.py
```

For persistent deployment:

```bash
# Install tmux
apt-get update && apt-get install -y tmux

# Create a new tmux session
tmux new -s api_server

# Inside tmux, start the API server
export MODEL_PATH="fine_tuned_model"
export USE_LORA="true"
export BASE_MODEL="llava-hf/llava-v1.6-mistral-7b-hf"
export QUANTIZE="true"
export PORT=8000

python api_server.py

# Detach from tmux: press Ctrl+B, then D
```

## 7. Using the API

```bash
# Test the API with sample script
python sample_usage.py \
  --api_url "http://your-runpod-ip:8000" \
  --image_url "https://pump.mypinata.cloud/ipfs/QmNzT34PBLpgB9GUCkcebkQa1yt3vnXg8bKCiiVmbrGuyo?img-width=800&img-dpr=2&img-onerror=redirect" \
  --tweet_text "Time is the true currency. — Elon Musk"
```

Expected output:

```json
{
  "tokenName": "TRUE CURRENCY",
  "ticker": "TIME"
}
```

## 8. Alternative Docker Deployment

```bash
# Build Docker image
docker build -t memcoin-generator .

# Run Docker container
docker run -d -p 8000:8000 --gpus all memcoin-generator
```

## 9. Saving for Redeployment

To save all the necessary files and steps for redeployment if the RunPod instance gets deleted:

```bash
# Make the script executable
chmod +x save_for_redeployment.sh

# Run the script
./save_for_redeployment.sh
```

This will create a `redeployment_backup` directory with all the necessary files and instructions for redeploying on a new RunPod instance.

## 10. Troubleshooting Common Issues

### Flash Attention Installation Failures

If you encounter errors when installing flash-attn:

```
Building wheel for flash-attn (setup.py) ... error
```

Don't worry! The setup has been modified to handle this case. The model will still work without flash-attn, but may be slightly slower.

### Out of Memory Errors

If you encounter out of memory errors during fine-tuning or inference:

1. Use a GPU with more VRAM (at least 24GB recommended)
2. Make sure to use the `--quantize` flag when fine-tuning or running inference
3. Use the `--use_lora` flag for more efficient fine-tuning
4. Reduce the batch size with `--batch_size 1`
5. Increase gradient accumulation steps with `--gradient_accumulation_steps 8`

### Model Loading Issues

If you encounter issues loading the model:

```
OSError: Unable to load weights from pytorch checkpoint file
```

Make sure you have enough disk space and that the model path is correct. You can also try downloading the model manually:

```bash
python -c "from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration; processor = LlavaNextProcessor.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf'); model = LlavaNextForConditionalGeneration.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf', torch_dtype='auto')"
```
