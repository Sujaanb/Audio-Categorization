"""
Pytest fixtures and configuration for Voice Detection API tests.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment variables BEFORE importing app
os.environ["VOICE_API_KEYS"] = "test-key-123,test-key-456"
os.environ["ENABLE_DOCS"] = "1"
os.environ["AASIST_MODEL_PATH"] = "models/test_model.pt"  # Doesn't need to exist - we mock
os.environ["AASIST_DEVICE"] = "cpu"
os.environ["AASIST_THRESHOLD"] = "0.5"

# Import base classes before mocking
from detectors.base import PredictionResult

# Create mock detector instance
_mock_detector = MagicMock()
_mock_detector.name = "aasist"
_mock_detector.load = MagicMock()
_mock_detector.predict = MagicMock(return_value=PredictionResult(
    classification="HUMAN",
    confidenceScore=0.75,
    explanation="[TEST] Mocked AASIST detector result for testing."
))

# Import the aasist_detector module to ensure it's loaded
import detectors.aasist_detector

# Mock the AASISTDetector class
_mock_aasist_class = MagicMock(return_value=_mock_detector)

# Patch at module level so it's applied before app import
_patcher = patch.object(detectors.aasist_detector, 'AASISTDetector', _mock_aasist_class)
_patcher.start()

# Now import the app (detector will be mocked)
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


def pytest_sessionfinish(session, exitstatus):
    """Cleanup after all tests."""
    _patcher.stop()
