# Python slim base image
FROM python:3.11-slim

# Install ffmpeg (required for audio decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .
COPY config.py .
COPY platform_services.py .
COPY base_requests.py .
COPY detectors/ ./detectors/
COPY services/ ./services/
COPY models/ ./models/

# Copy model weights (bundled in image)
COPY model_weights/ ./model_weights/

# Set environment variables
ENV PORT=8080
ENV AASIST_DEVICE=cpu
ENV AASIST_WEIGHTS_PATH=./model_weights/aasist_original.pth

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "api.py"]
