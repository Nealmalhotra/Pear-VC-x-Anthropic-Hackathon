#!/bin/bash

# Change to the parent directory
cd "$(dirname "$0")/.."

# Build the Docker image
echo "Building Docker image..."
docker build -t math-tokenizer-service:latest -f docker/Dockerfile .

# Run the container
echo "Running container..."
docker run -p 8000:8000 -v "$(pwd)"/data:/app/data math-tokenizer-service:latest

# Note: To stop the container, press Ctrl+C 