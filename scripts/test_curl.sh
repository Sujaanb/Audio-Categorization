#!/bin/bash
# Local test script using curl
# 
# Usage:
#   ./test_curl.sh path/to/audio.mp3 [API_KEY] [URL]
#
# Example:
#   ./test_curl.sh sample.mp3 test-key-123 http://localhost:8001

set -e

AUDIO_FILE="${1:-sample.mp3}"
API_KEY="${2:-test-key-123}"
BASE_URL="${3:-http://localhost:8001}"

if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found: $AUDIO_FILE"
    echo "Usage: $0 <audio.mp3> [api-key] [base-url]"
    exit 1
fi

echo "Testing Voice Detection API"
echo "==========================="
echo "Audio: $AUDIO_FILE"
echo "URL: $BASE_URL"
echo ""

# Encode audio to base64
AUDIO_BASE64=$(base64 -w 0 "$AUDIO_FILE" 2>/dev/null || base64 "$AUDIO_FILE")

# Build JSON payload
JSON_PAYLOAD=$(cat <<EOF
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "$AUDIO_BASE64"
}
EOF
)

# Send request
echo "Sending request..."
curl -s -X POST "${BASE_URL}/api/voice-detection" \
    -H "Content-Type: application/json" \
    -H "x-api-key: ${API_KEY}" \
    -d "$JSON_PAYLOAD" | python3 -m json.tool

echo ""
echo "Done."
