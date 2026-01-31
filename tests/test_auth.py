"""
Tests for API authentication (x-api-key header).
"""

import pytest


class TestAuthentication:
    """Test API key authentication."""

    def test_missing_api_key_returns_401(self, test_client, valid_request_body):
        """Request without x-api-key header should return 401."""
        response = test_client.post(
            "/api/voice-detection",
            json=valid_request_body,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 401
        data = response.json()
        assert data["detail"]["status"] == "error"
        assert "API key" in data["detail"]["message"]

    def test_invalid_api_key_returns_401(self, test_client, invalid_headers, valid_request_body):
        """Request with invalid x-api-key should return 401."""
        response = test_client.post(
            "/api/voice-detection",
            json=valid_request_body,
            headers=invalid_headers,
        )
        assert response.status_code == 401
        data = response.json()
        assert data["detail"]["status"] == "error"
        assert "Invalid" in data["detail"]["message"]

    def test_valid_api_key_passes(self, test_client, valid_headers, valid_request_body):
        """Request with valid x-api-key should not return 401."""
        response = test_client.post(
            "/api/voice-detection",
            json=valid_request_body,
            headers=valid_headers,
        )
        # Should not be 401 (might be 400 if MP3 decoding fails, but not auth error)
        assert response.status_code != 401

    def test_second_valid_api_key_passes(self, test_client, valid_request_body):
        """Request with second valid API key should also pass."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": "test-key-456",
        }
        response = test_client.post(
            "/api/voice-detection",
            json=valid_request_body,
            headers=headers,
        )
        assert response.status_code != 401

    def test_empty_api_key_returns_401(self, test_client, valid_request_body):
        """Request with empty x-api-key should return 401."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": "",
        }
        response = test_client.post(
            "/api/voice-detection",
            json=valid_request_body,
            headers=headers,
        )
        assert response.status_code == 401
