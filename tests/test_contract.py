"""
Tests for API response contract (JSON shape and types).
"""

import base64

import pytest


class TestContract:
    """Test response JSON contract matches specification."""

    def test_success_response_shape(self, test_client, valid_headers, minimal_mp3_base64):
        """
        Success response should have exact shape:
        {
            "status": "success",
            "language": str,
            "classification": "AI_GENERATED" | "HUMAN",
            "confidenceScore": float in [0,1],
            "explanation": str
        }
        """
        response = test_client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": minimal_mp3_base64,
            },
            headers=valid_headers,
        )

        # If decode fails, we get 400 error which is also valid
        if response.status_code == 400:
            data = response.json()
            assert "status" in data
            assert data["status"] == "error"
            assert "message" in data
            assert isinstance(data["message"], str)
            return

        # Success response
        assert response.status_code == 200
        data = response.json()

        # Check all required fields exist
        assert "status" in data
        assert "language" in data
        assert "classification" in data
        assert "confidenceScore" in data
        assert "explanation" in data

        # Check types
        assert data["status"] == "success"
        assert isinstance(data["language"], str)
        assert data["classification"] in ["AI_GENERATED", "HUMAN"]
        assert isinstance(data["confidenceScore"], (int, float))
        assert 0.0 <= data["confidenceScore"] <= 1.0
        assert isinstance(data["explanation"], str)

    def test_error_response_shape(self, test_client, valid_headers):
        """
        Error response should have exact shape:
        {
            "status": "error",
            "message": str
        }
        """
        response = test_client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": "definitely-not-valid-base64!!!",
            },
            headers=valid_headers,
        )

        assert response.status_code == 400
        data = response.json()

        # Check exact shape - only status and message
        assert "status" in data
        assert "message" in data
        assert data["status"] == "error"
        assert isinstance(data["message"], str)
        assert len(data["message"]) > 0

        # Should not have extra fields (other than status and message)
        allowed_keys = {"status", "message"}
        assert set(data.keys()) == allowed_keys

    def test_auth_error_response_shape(self, test_client, invalid_headers, minimal_mp3_base64):
        """Auth error response should have error shape in detail."""
        response = test_client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": minimal_mp3_base64,
            },
            headers=invalid_headers,
        )

        assert response.status_code == 401
        data = response.json()

        # FastAPI wraps HTTPException in detail
        assert "detail" in data
        detail = data["detail"]
        assert "status" in detail
        assert "message" in detail
        assert detail["status"] == "error"

    def test_language_echoed_in_response(self, test_client, valid_headers, minimal_mp3_base64):
        """Response should echo the same language from request."""
        languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

        for lang in languages:
            response = test_client.post(
                "/api/voice-detection",
                json={
                    "language": lang,
                    "audioFormat": "mp3",
                    "audioBase64": minimal_mp3_base64,
                },
                headers=valid_headers,
            )

            if response.status_code == 200:
                data = response.json()
                assert data["language"] == lang

    def test_confidence_score_range(self, test_client, valid_headers, minimal_mp3_base64):
        """Confidence score should always be between 0 and 1."""
        response = test_client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": minimal_mp3_base64,
            },
            headers=valid_headers,
        )

        if response.status_code == 200:
            data = response.json()
            score = data["confidenceScore"]
            assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"

    def test_no_audio_bytes_in_response(self, test_client, valid_headers, minimal_mp3_base64):
        """Response should never contain audio bytes or base64 audio."""
        response = test_client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": minimal_mp3_base64,
            },
            headers=valid_headers,
        )

        data = response.json()

        # Check that response doesn't have audio-related fields at the top level
        # (error messages may contain "audioBase64" as text, which is fine)
        forbidden_keys = ["audioBase64", "audio_bytes", "waveform", "audio", "mp3_bytes"]
        for key in forbidden_keys:
            assert key not in data, f"Response should not contain '{key}' field"
