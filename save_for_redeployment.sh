#!/bin/bash

# This script saves all the necessary files and steps for redeployment if the RunPod instance gets deleted

# Create a directory for saving files
mkdir -p redeployment_backup

# Copy all code files
cp -r *.py *.sh *.md requirements.txt Dockerfile redeployment_backup/

# Save the fine-tuned model if it exists
if [ -d "fine_tuned_model" ]; then
    echo "Backing up fine-tuned model..."
    
    # Create a compressed archive of the fine-tuned model
    tar -czf redeployment_backup/fine_tuned_model.tar.gz fine_tuned_model
    
    echo "Fine-tuned model backed up successfully!"
else
    echo "No fine-tuned model found. Skipping model backup."
fi

# Save the dataset if it exists
if [ -d "data" ]; then
    echo "Backing up dataset..."
    
    # Create a compressed archive of the dataset
    tar -czf redeployment_backup/data.tar.gz data
    
    echo "Dataset backed up successfully!"
else
    echo "No dataset found. Skipping dataset backup."
fi

# Create a README file with instructions for redeployment
cat > redeployment_backup/REDEPLOYMENT_INSTRUCTIONS.md << 'EOF'
# Redeployment Instructions

Follow these steps to redeploy your meme coin generator on a new RunPod instance:

## 1. Create a New RunPod Instance

1. Create a new RunPod instance with at least 24GB VRAM
2. Select PyTorch template with at least 50GB disk space

## 2. Upload the Backup Files

Upload the contents of this backup directory to the new RunPod instance:

```bash
# On your local machine
scp -r redeployment_backup/* user@your-runpod-ip:/workspace/
```

Or use the RunPod web interface to upload the files.

## 3. Extract the Backup Files

```bash
# Extract the fine-tuned model (if it exists)
if [ -f "fine_tuned_model.tar.gz" ]; then
    tar -xzf fine_tuned_model.tar.gz
fi

# Extract the dataset (if it exists)
if [ -f "data.tar.gz" ]; then
    tar -xzf data.tar.gz
fi

# Make scripts executable
chmod +x *.sh
```

## 4. Set Up the Environment

```bash
# Run the setup script
./setup.sh
```

## 5. Start the API Server

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

## 6. Test the API

```bash
# Test the API with sample script
python sample_usage.py \
  --api_url "http://your-runpod-ip:8000" \
  --image_url "https://pump.mypinata.cloud/ipfs/QmNzT34PBLpgB9GUCkcebkQa1yt3vnXg8bKCiiVmbrGuyo?img-width=800&img-dpr=2&img-onerror=redirect" \
  --tweet_text "Time is the true currency. — Elon Musk"
```
EOF

echo "Redeployment backup created successfully in the 'redeployment_backup' directory."
echo "To save these files, download the 'redeployment_backup' directory to your local machine."
echo "For instructions on redeploying, see the 'REDEPLOYMENT_INSTRUCTIONS.md' file in the backup directory."
