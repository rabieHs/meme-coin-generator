#!/bin/bash

# This script sets up the environment on RunPod and starts the API server

# Update system packages
apt-get update && apt-get install -y git wget

# Create and activate conda environment
conda create -n llava_env python=3.10 -y
source activate llava_env

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Set environment variables
export MODEL_PATH=${MODEL_PATH:-"fine_tuned_model"}
export USE_LORA=${USE_LORA:-"false"}
export BASE_MODEL=${BASE_MODEL:-"llava-hf/llava-v1.6-mistral-7b-hf"}
export QUANTIZE=${QUANTIZE:-"true"}
export PORT=${PORT:-8000}

# Start the API server
python api_server.py
