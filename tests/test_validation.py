"""
Tests for request validation (audioFormat, language, base64).
"""

import pytest


class TestValidation:
    """Test request body validation."""

    def test_invalid_base64_returns_400(self, test_client, valid_headers):
        """Invalid base64 string should return 400."""
        response = test_client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": "not-valid-base64!!!",
            },
            headers=valid_headers,
        )
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "base64" in data["message"].lower()

    def test_wrong_audio_format_returns_422(self, test_client, valid_headers, minimal_mp3_base64):
        """Non-mp3 audioFormat should return 422 (validation error)."""
        response = test_client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "wav",
                "audioBase64": minimal_mp3_base64,
            },
            headers=valid_headers,
        )
        # Pydantic validation returns 422 for invalid field values
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_unsupported_language_returns_422(self, test_client, valid_headers, minimal_mp3_base64):
        """Unsupported language should return 422 (validation error)."""
        response = test_client.post(
            "/api/voice-detection",
            json={
                "language": "French",
                "audioFormat": "mp3",
                "audioBase64": minimal_mp3_base64,
            },
            headers=valid_headers,
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_empty_audio_base64_returns_422(self, test_client, valid_headers):
        """Empty audioBase64 should return 422."""
        response = test_client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": "",
            },
            headers=valid_headers,
        )
        assert response.status_code == 422

    def test_missing_language_returns_422(self, test_client, valid_headers, minimal_mp3_base64):
        """Missing language field should return 422."""
        response = test_client.post(
            "/api/voice-detection",
            json={
                "audioFormat": "mp3",
                "audioBase64": minimal_mp3_base64,
            },
            headers=valid_headers,
        )
        assert response.status_code == 422

    def test_uppercase_audio_format_accepted(self, test_client, valid_headers, minimal_mp3_base64):
        """Uppercase 'MP3' should be accepted."""
        response = test_client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "MP3",
                "audioBase64": minimal_mp3_base64,
            },
            headers=valid_headers,
        )
        # Should not be 422 for audioFormat (might be 400 if decode fails)
        assert response.status_code != 422

    def test_mixed_case_audio_format_accepted(self, test_client, valid_headers, minimal_mp3_base64):
        """Mixed case 'Mp3' should be accepted."""
        response = test_client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "Mp3",
                "audioBase64": minimal_mp3_base64,
            },
            headers=valid_headers,
        )
        assert response.status_code != 422

    def test_all_languages_accepted(self, test_client, valid_headers, minimal_mp3_base64):
        """All five supported languages should be accepted."""
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
            # Should not be 422 (validation passes)
            assert response.status_code != 422, f"Language '{lang}' should be accepted"
