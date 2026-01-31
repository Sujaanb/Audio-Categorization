"""
Pydantic request and response models for the Voice Detection API.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# Supported languages
SUPPORTED_LANGUAGES = Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"]


class VoiceDetectionRequest(BaseModel):
    """Request model for voice detection endpoint."""

    language: SUPPORTED_LANGUAGES = Field(
        ...,
        description="Language of the audio. Must be one of: Tamil, English, Hindi, Malayalam, Telugu",
    )
    audioFormat: str = Field(
        ...,
        description="Audio format. Only 'mp3' is supported (case-insensitive)",
    )
    audioBase64: str = Field(
        ...,
        description="Base64-encoded MP3 audio bytes",
        min_length=1,
    )

    @field_validator("audioFormat")
    @classmethod
    def validate_audio_format(cls, v: str) -> str:
        """Validate audioFormat is mp3 (case-insensitive)."""
        if v.lower() != "mp3":
            raise ValueError("Only 'mp3' audio format is supported")
        return v.lower()


class VoiceDetectionSuccessResponse(BaseModel):
    """Success response model for voice detection."""

    status: Literal["success"] = "success"
    language: str = Field(..., description="Language from the request")
    classification: Literal["AI_GENERATED", "HUMAN"] = Field(
        ...,
        description="Classification result: AI_GENERATED or HUMAN",
    )
    confidenceScore: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1",
    )
    explanation: str = Field(
        ...,
        max_length=200,
        description="Brief explanation of the classification",
    )


class VoiceDetectionErrorResponse(BaseModel):
    """Error response model for voice detection."""

    status: Literal["error"] = "error"
    message: str = Field(..., description="Human-readable error message")