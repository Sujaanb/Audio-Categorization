#!/usr/bin/env python3
"""
Local test script for the Voice Detection API.

Reads an MP3 file, encodes it to base64, and sends a request to the API.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Error: httpx not installed. Run: pip install httpx")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Test the Voice Detection API with a local MP3 file"
    )
    parser.add_argument(
        "--audio", 
        type=str, 
        required=True, 
        help="Path to MP3 audio file"
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        default="test-key-123",
        help="API key for authentication (default: test-key-123)"
    )
    parser.add_argument(
        "--url", 
        type=str, 
        default="http://localhost:8001",
        help="API base URL (default: http://localhost:8001)"
    )
    parser.add_argument(
        "--language", 
        type=str, 
        default="English",
        choices=["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
        help="Language of the audio (default: English)"
    )
    args = parser.parse_args()

    # Read and encode audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"Reading audio file: {audio_path}")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    print(f"Audio size: {len(audio_bytes)} bytes ({len(audio_base64)} base64 chars)")

    # Prepare request
    endpoint = f"{args.url.rstrip('/')}/api/voice-detection"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": args.api_key,
    }
    payload = {
        "language": args.language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64,
    }

    # Send request
    print(f"\nSending request to: {endpoint}")
    print(f"Language: {args.language}")
    print("-" * 50)

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(endpoint, json=payload, headers=headers)

        print(f"Status: {response.status_code}")
        
        result = response.json()
        print(f"\nResponse:")
        print(json.dumps(result, indent=2))

        if response.status_code == 200:
            print("-" * 50)
            print(f"Classification: {result.get('classification', 'N/A')}")
            print(f"Confidence: {result.get('confidenceScore', 'N/A')}")
            print(f"Explanation: {result.get('explanation', 'N/A')}")

    except httpx.ConnectError:
        print(f"Error: Could not connect to {args.url}")
        print("Make sure the server is running: uvicorn api:app --port 8001")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
