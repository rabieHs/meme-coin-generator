#!/bin/bash

# Update system packages
apt-get update && apt-get install -y git wget ninja-build

# Create and activate conda environment
conda create -n llava_env python=3.10 -y
source activate llava_env

# Install PyTorch with CUDA support
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Try to install flash-attn from pre-built wheel (optional)
echo "Attempting to install flash-attn (optional)..."
pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed, but this is optional and won't affect basic functionality"

# Create directories for model and data
mkdir -p models
mkdir -p data
mkdir -p fine_tuned_model

echo "Setup completed successfully!"
