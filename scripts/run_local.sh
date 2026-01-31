#!/bin/bash
# Run the API locally with hot reload

set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Ensure VOICE_API_KEYS is set
if [ -z "$VOICE_API_KEYS" ]; then
    echo "Warning: VOICE_API_KEYS not set. Using default test key."
    export VOICE_API_KEYS="dev-test-key-123"
fi

# Enable docs for local development
export ENABLE_DOCS=1

echo "Starting API on http://localhost:8001"
echo "API docs available at http://localhost:8001/docs"
echo ""

# Run with uvicorn and hot reload
uvicorn api:app --reload --host 0.0.0.0 --port 8001
