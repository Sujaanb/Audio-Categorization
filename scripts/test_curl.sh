#!/bin/bash
# =============================================================================
# Test script for Voice Detection API
# Works with local server or Cloud Run deployment
# =============================================================================

set -e

AUDIO_FILE="${1:-sample.mp3}"
API_KEY="${2:-${VOICE_API_KEYS:-test-key-123}}"

# Use SERVICE_URL env var if set, otherwise default to localhost
if [ -z "$SERVICE_URL" ]; then
    echo "SERVICE_URL not set. Using localhost:8001 for local testing."
    echo ""
    echo "For Cloud Run testing, set SERVICE_URL:"
    echo "  export SERVICE_URL=\$(gcloud run services describe voice-detect-api --region asia-south1 --format='value(status.url)')"
    echo ""
    BASE_URL="http://localhost:8001"
else
    BASE_URL="$SERVICE_URL"
fi

if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found: $AUDIO_FILE"
    echo ""
    echo "Usage: $0 <audio.mp3> [api-key]"
    echo ""
    echo "Examples:"
    echo "  $0 sample.mp3                    # Local test"
    echo "  SERVICE_URL=https://... $0 sample.mp3  # Cloud Run test"
    exit 1
fi

echo "========================================"
echo "Testing Voice Detection API"
echo "========================================"
echo "URL:   $BASE_URL"
echo "Audio: $AUDIO_FILE"
echo ""

# Encode audio to base64 (handle both Linux and macOS)
if command -v base64 &> /dev/null; then
    if base64 --help 2>&1 | grep -q "GNU"; then
        # GNU base64 (Linux)
        AUDIO_BASE64=$(base64 -w 0 "$AUDIO_FILE")
    else
        # BSD base64 (macOS)
        AUDIO_BASE64=$(base64 -i "$AUDIO_FILE")
    fi
else
    echo "Error: base64 command not found"
    exit 1
fi

echo "Sending request..."
echo ""

# Send request and format response
curl -s -X POST "${BASE_URL}/api/voice-detection" \
    -H "Content-Type: application/json" \
    -H "x-api-key: ${API_KEY}" \
    -d "{
        \"language\": \"English\",
        \"audioFormat\": \"mp3\",
        \"audioBase64\": \"${AUDIO_BASE64}\"
    }" | python3 -m json.tool

echo ""
echo "========================================"
