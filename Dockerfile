# Use the official NVIDIA CUDA 12.1 runtime image with cuDNN 8 on Ubuntu 20.04
FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (including git)
RUN apt-get update && apt-get install -y \
    python python3-pip \
    ffmpeg \
    libavcodec-extra \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY multicore_detection_debug.py .

# Command to run your script
CMD ["python", "multicore_detection_debug.py"]
