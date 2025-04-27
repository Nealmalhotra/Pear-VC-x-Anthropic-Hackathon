FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data

# Copy source code
COPY src/ /app/

# Set environment variables
ENV PYTHONPATH=/app
ENV TOKENIZER_PATH=/app/data/math_tokenizer.json

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"] 