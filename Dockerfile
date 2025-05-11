FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/miniconda3/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /root/miniconda3 \
    && rm /tmp/miniconda.sh \
    && conda clean -a -y

# Create conda environment
RUN conda create -n llava_env python=3.10 -y \
    && echo "source activate llava_env" > ~/.bashrc

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN . /root/miniconda3/bin/activate llava_env \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip install -r requirements.txt

# Copy application files
COPY . .

# Make scripts executable
RUN chmod +x runpod_setup.sh

# Expose port
EXPOSE 8000

# Set environment variables
ENV MODEL_PATH=fine_tuned_model
ENV USE_LORA=false
ENV BASE_MODEL=llava-hf/llava-v1.6-mistral-7b-hf
ENV QUANTIZE=true
ENV PORT=8000

# Start the API server
CMD ["/bin/bash", "-c", "source activate llava_env && python api_server.py"]
