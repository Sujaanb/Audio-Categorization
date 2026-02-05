"""
Configuration settings for AI-Generated Voice Detection API.
Uses pydantic-settings to load from environment variables and .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Project metadata
    PROJECT_NAME: str = "AI-Generated Voice Detection API"

    # API documentation (enabled by default)
    ENABLE_DOCS: bool = True

    # API authentication (comma-separated list of valid API keys)
    VOICE_API_KEYS: str = ""

    # ==========================================================================
    # AASIST Model Configuration
    # ==========================================================================

    # Device for inference: "cpu" or "cuda"
    AASIST_DEVICE: str = "cpu"

    # Path to model weights file
    AASIST_WEIGHTS_PATH: str = "./model_weights/aasist_original.pth"

    # Classification threshold (spoof probability > threshold = AI_GENERATED)
    AASIST_THRESHOLD: float = 0.5

    # Multi-window inference: max windows to sample from long audio
    AASIST_MAX_WINDOWS: int = 3

    # ==========================================================================
    # Audio Limits
    # ==========================================================================

    MAX_MP3_BYTES: int = 15_000_000  # 15 MB
    MAX_DURATION_SECONDS: float = 300.0  # 5 minutes
    MIN_DURATION_SECONDS: float = 0.5  # 0.5 seconds

    # Quality control thresholds
    SILENCE_RATIO_THRESHOLD: float = 0.80

    # ==========================================================================
    # Server Configuration
    # ==========================================================================

    PORT: int = 8080

    def get_api_keys(self) -> list[str]:
        """Parse comma-separated API keys into a list."""
        if not self.VOICE_API_KEYS:
            return []
        return [key.strip() for key in self.VOICE_API_KEYS.split(",") if key.strip()]

    def get_max_base64_length(self) -> int:
        """Calculate max base64 string length based on MAX_MP3_BYTES."""
        return int(self.MAX_MP3_BYTES * 4 / 3) + 100


settings = Settings()