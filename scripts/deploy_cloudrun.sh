#!/bin/bash
# =============================================================================
# Canonical Cloud Run Deployment Script
# Builds via Cloud Build, pushes to Artifact Registry, deploys to Cloud Run
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION (do not change unless project structure changes)
# =============================================================================
PROJECT_ID="voice-detect-cloudrun"
REGION="asia-south1"
SERVICE_NAME="voice-detect-api"
REPO_NAME="voice-detect-docker"
IMAGE_NAME="voice-detect-api"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
SERVICE_ACCOUNT="voice-detect-runtime@${PROJECT_ID}.iam.gserviceaccount.com"

# GCS paths for model weights
AASIST_ORIG_WEIGHTS_GCS_URI="gs://voice-detect-168345068797-models/models/aasist_original.pth"
AASIST_FT_WEIGHTS_GCS_URI="gs://voice-detect-168345068797-models/models/aasist_finetuned_best.pth"

# =============================================================================
# LOAD ENV VARS FROM .env IF PRESENT
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment from .env file..."
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# =============================================================================
# VALIDATE PREREQUISITES
# =============================================================================
echo "========================================"
echo "Validating prerequisites..."
echo "========================================"

# Check gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI not installed. Install from https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "ERROR: Not logged in to gcloud. Run: gcloud auth login"
    exit 1
fi

# Check project is set correctly
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
    echo "WARNING: Current project is '$CURRENT_PROJECT', expected '$PROJECT_ID'"
    echo "Setting project to $PROJECT_ID..."
    gcloud config set project "$PROJECT_ID"
fi

# Check VOICE_API_KEYS is set
if [ -z "$VOICE_API_KEYS" ]; then
    echo "ERROR: VOICE_API_KEYS is not set."
    echo "Set it in .env file or export VOICE_API_KEYS=your-key before running."
    exit 1
fi

echo "✓ gcloud installed and authenticated"
echo "✓ Project: $PROJECT_ID"
echo "✓ VOICE_API_KEYS is set"

# =============================================================================
# CONFIGURE DOCKER AUTH FOR ARTIFACT REGISTRY
# =============================================================================
echo ""
echo "========================================"
echo "Configuring Artifact Registry auth..."
echo "========================================"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "✓ Docker auth configured for ${REGION}-docker.pkg.dev"

# =============================================================================
# BUILD AND PUSH VIA CLOUD BUILD
# =============================================================================
echo ""
echo "========================================"
echo "Building and pushing image via Cloud Build..."
echo "========================================"
echo "Image URI: $IMAGE_URI"

cd "$PROJECT_ROOT"

gcloud builds submit --tag "$IMAGE_URI" .

echo "✓ Image built and pushed: $IMAGE_URI"

# =============================================================================
# DEPLOY TO CLOUD RUN
# =============================================================================
echo ""
echo "========================================"
echo "Deploying to Cloud Run..."
echo "========================================"

gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_URI" \
    --platform managed \
    --region "$REGION" \
    --service-account "$SERVICE_ACCOUNT" \
    --memory 4Gi \
    --cpu 2 \
    --concurrency 1 \
    --timeout 300 \
    --max-instances 2 \
    --min-instances 0 \
    --allow-unauthenticated \
    --set-env-vars="VOICE_API_KEYS=${VOICE_API_KEYS}" \
    --set-env-vars="AASIST_DEVICE=cpu" \
    --set-env-vars="AASIST_ORIG_WEIGHTS_GCS_URI=${AASIST_ORIG_WEIGHTS_GCS_URI}" \
    --set-env-vars="AASIST_FT_WEIGHTS_GCS_URI=${AASIST_FT_WEIGHTS_GCS_URI}" \
    --set-env-vars="AASIST_ORIG_CACHE_PATH=/tmp/aasist_original.pth" \
    --set-env-vars="AASIST_FT_CACHE_PATH=/tmp/aasist_finetuned_best.pth" \
    --set-env-vars="AASIST_MAX_WINDOWS=3" \
    --set-env-vars="AASIST_THRESHOLD=0.5" \
    --set-env-vars="ENABLE_DOCS=0"

echo "✓ Deployed to Cloud Run"

# =============================================================================
# GET SERVICE URL
# =============================================================================
echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format="value(status.url)")

echo ""
echo "Service Name:  $SERVICE_NAME"
echo "Image URI:     $IMAGE_URI"
echo "Service URL:   $SERVICE_URL"
echo ""
echo "Test with:"
echo "  export SERVICE_URL=$SERVICE_URL"
echo "  bash scripts/test_curl.sh sample.mp3"
echo "========================================"
