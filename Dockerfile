# Dockerfile for AI-Generated Voice Detection API
# Optimized for Google Cloud Run

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Install system dependencies (ffmpeg for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port (Cloud Run uses PORT env var)
EXPOSE 8080

# Health check (optional, Cloud Run handles this)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8080/api/voice-detection || exit 1

# Run the application
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
