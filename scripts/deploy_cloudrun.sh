#!/bin/bash
# Deploy to Google Cloud Run
# 
# Prerequisites:
# - gcloud CLI installed and authenticated
# - Docker installed (for local builds)
# - Project ID and region configured

set -e

# =============================================================================
# CONFIGURATION - Update these values
# =============================================================================

PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
SERVICE_NAME="${CLOUD_RUN_SERVICE:-voice-detection-api}"
REGION="${GCP_REGION:-asia-south1}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# API configuration
VOICE_API_KEYS="${VOICE_API_KEYS:-your-production-api-key}"

# AASIST model configuration
AASIST_MODEL_PATH="${AASIST_MODEL_PATH:-/app/models/aasist_finetuned.pt}"
AASIST_DEVICE="${AASIST_DEVICE:-cpu}"

# =============================================================================
# BUILD AND DEPLOY
# =============================================================================

echo "========================================"
echo "Deploying ${SERVICE_NAME} to Cloud Run"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "========================================"

# Ensure we're in the project directory
cd "$(dirname "$0")/.."

# Option 1: Build using Cloud Build (recommended)
echo "Building image with Cloud Build..."
gcloud builds submit --tag "${IMAGE_NAME}" --project "${PROJECT_ID}"

# Option 2: Build locally and push (uncomment if preferred)
# echo "Building image locally..."
# docker build -t "${IMAGE_NAME}" .
# docker push "${IMAGE_NAME}"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --platform managed \
    --allow-unauthenticated \
    --set-env-vars="VOICE_API_KEYS=${VOICE_API_KEYS}" \
    --set-env-vars="AASIST_MODEL_PATH=${AASIST_MODEL_PATH}" \
    --set-env-vars="AASIST_DEVICE=${AASIST_DEVICE}" \
    --set-env-vars="ENABLE_DOCS=0" \
    --set-env-vars="MAX_MP3_BYTES=15000000" \
    --set-env-vars="MAX_DURATION_SECONDS=300" \
    --memory=1Gi \
    --cpu=1 \
    --timeout=120s \
    --concurrency=1 \
    --min-instances=0 \
    --max-instances=10

# Note on settings:
# - concurrency=1: Each instance handles one request at a time (safer for ML models)
# - timeout=120s: Allow up to 2 minutes per request for audio processing
# - memory=1Gi: Increased for PyTorch model loading
# - min-instances=0: Scale to zero when idle (cost savings)

echo ""
echo "========================================"
echo "Deployment complete!"
echo ""
echo "Get the service URL with:"
echo "  gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)'"
echo "========================================"
