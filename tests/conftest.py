"""
Pytest fixtures and configuration for Voice Detection API tests.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment variables BEFORE importing app
os.environ["VOICE_API_KEYS"] = "test-key-123,test-key-456"
os.environ["DETECTOR_BACKEND"] = "qc_fallback"
os.environ["ENABLE_DOCS"] = "1"

from api import app
from detectors import reset_detector


@pytest.fixture(scope="session")
def test_client():
    """Create a test client for the API."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def valid_api_key():
    """Return a valid API key for testing."""
    return "test-key-123"


@pytest.fixture
def invalid_api_key():
    """Return an invalid API key for testing."""
    return "invalid-key-999"


@pytest.fixture
def valid_headers(valid_api_key):
    """Return valid headers for API requests."""
    return {
        "Content-Type": "application/json",
        "x-api-key": valid_api_key,
    }


@pytest.fixture
def invalid_headers(invalid_api_key):
    """Return headers with invalid API key."""
    return {
        "Content-Type": "application/json",
        "x-api-key": invalid_api_key,
    }


# Minimal valid MP3 file in base64 (a very short silent MP3)
# This is a minimal valid MP3 frame that ffmpeg can decode
MINIMAL_MP3_BASE64 = (
    "//uQxAAAAAANIAAAAAExBTUUzLjEwMFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAw"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
)


@pytest.fixture
def minimal_mp3_base64():
    """Return minimal valid MP3 base64 for testing."""
    return MINIMAL_MP3_BASE64


@pytest.fixture
def valid_request_body(minimal_mp3_base64):
    """Return a valid request body for testing."""
    return {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": minimal_mp3_base64,
    }
