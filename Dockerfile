# Dockerfile for AI-Generated Voice Detection API
# Optimized for Google Cloud Run (CPU-only)

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Install system dependencies
# - ffmpeg: audio processing
# - libsndfile1: required by soundfile Python package
# - ca-certificates: SSL/TLS support
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies (production only)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port (Cloud Run uses PORT env var)
EXPOSE 8080

# Run the application
# Cloud Run injects PORT; container must listen on 0.0.0.0
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
