# Create the startup script

#!/bin/bash

# Activate the virtual environment
source /workspace/llava-env/bin/activate

# Check if the repository exists
if [ ! -d "/workspace/meme-coin-generator" ]; then
  # Clone the repository
  git clone https://github.com/rabieHs/meme-coin-generator.git /workspace/meme-coin-generator
fi

# Create symbolic links
cd /workspace/meme-coin-generator
ln -sf /workspace/data data
ln -sf /workspace/models models
ln -sf /workspace/fine_tuned_model fine_tuned_model
ln -sf /workspace/logs logs

# Kill the existing API server if it's running
if [ -f "/workspace/api_server.pid" ]; then
  kill $(cat /workspace/api_server.pid) 2>/dev/null || true
  rm /workspace/api_server.pid
fi

# Set environment variables
export MODEL_PATH=/workspace/fine_tuned_model
export USE_LORA=true
export BASE_MODEL=/workspace/models/llava-v1.6-mistral-7b-hf
export QUANTIZE=true
export PORT=8000

# Start the API server in the background
mkdir -p /workspace/logs
nohup python /workspace/meme-coin-generator/api_server.py > /workspace/logs/api_server.log 2>&1 &

# Save the process ID
echo $! > /workspace/api_server.pid

echo "API server started with PID $(cat /workspace/api_server.pid)"
echo "To check the logs, run: cat /workspace/logs/api_server.log"
