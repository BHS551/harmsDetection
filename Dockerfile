# Use an official Python runtime as the base image.
FROM python:3.8-slim

# Install system dependencies (if needed for OpenCV and other libraries)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file and install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Run the detection script.
CMD ["python", "rtsp_detection.py"]
