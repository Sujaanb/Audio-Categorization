#!/bin/bash
# Deploy to Google Cloud Run
# 
# Prerequisites:
# - gcloud CLI installed and authenticated
# - Docker installed (for local builds)
# - Project ID and region configured
# - Model weights uploaded to GCS bucket

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

# AASIST ensemble model configuration
AASIST_DEVICE="cpu"
AASIST_ORIG_WEIGHTS_GCS_URI="${AASIST_ORIG_WEIGHTS_GCS_URI:-gs://your-bucket/aasist_original.pth}"
AASIST_FT_WEIGHTS_GCS_URI="${AASIST_FT_WEIGHTS_GCS_URI:-gs://your-bucket/aasist_finetuned_best.pth}"
AASIST_ORIG_CACHE_PATH="/tmp/aasist_original.pth"
AASIST_FT_CACHE_PATH="/tmp/aasist_finetuned_best.pth"
AASIST_MAX_WINDOWS="3"
AASIST_THRESHOLD="0.5"

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
    --set-env-vars="AASIST_DEVICE=${AASIST_DEVICE}" \
    --set-env-vars="AASIST_ORIG_WEIGHTS_GCS_URI=${AASIST_ORIG_WEIGHTS_GCS_URI}" \
    --set-env-vars="AASIST_FT_WEIGHTS_GCS_URI=${AASIST_FT_WEIGHTS_GCS_URI}" \
    --set-env-vars="AASIST_ORIG_CACHE_PATH=${AASIST_ORIG_CACHE_PATH}" \
    --set-env-vars="AASIST_FT_CACHE_PATH=${AASIST_FT_CACHE_PATH}" \
    --set-env-vars="AASIST_MAX_WINDOWS=${AASIST_MAX_WINDOWS}" \
    --set-env-vars="AASIST_THRESHOLD=${AASIST_THRESHOLD}" \
    --set-env-vars="ENABLE_DOCS=0" \
    --set-env-vars="MAX_MP3_BYTES=15000000" \
    --set-env-vars="MAX_DURATION_SECONDS=300" \
    --memory=4Gi \
    --cpu=2 \
    --timeout=300s \
    --concurrency=1 \
    --min-instances=0 \
    --max-instances=10

# Note on settings:
# - concurrency=1: Each instance handles one request at a time (safer for ML models)
# - timeout=300s: Allow up to 5 minutes per request for audio processing
# - memory=4Gi: Increased for PyTorch ensemble model loading (2 models)
# - cpu=2: Two CPUs for ensemble inference

echo ""
echo "========================================"
echo "Deployment complete!"
echo ""
echo "Get the service URL with:"
echo "  gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)'"
echo ""
echo "Test the API with:"
echo "  curl -X POST <SERVICE_URL>/api/voice-detection \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -H 'x-api-key: ${VOICE_API_KEYS}' \\"
echo "    -d '{\"language\": \"English\", \"audioFormat\": \"mp3\", \"audioBase64\": \"<BASE64_MP3>\"}'"
echo "========================================"
