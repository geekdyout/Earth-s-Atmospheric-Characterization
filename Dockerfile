FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    libnetcdf-dev \
    libhdf5-dev \
    libproj-dev \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/temporal data/metrics

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["bash"]

# CubeSat simulation configuration
# This simulates the resource constraints of a CubeSat
# Memory: 200MB, CPU: 1 core
# Usage: docker build -t cubesat-sim --memory=200mb --cpus=1.0 -f Dockerfile.jetson .