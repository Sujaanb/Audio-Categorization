"""
External API detector stub.
Placeholder for calling external detection services.
"""

from typing import Dict

import numpy as np

from .base import BaseDetector, PredictionResult


class ExternalDetector(BaseDetector):
    """
    External API based voice detection stub.

    This detector would call external services like:
    - Third-party AI detection APIs
    - Custom ML model servers
    - Cloud-hosted inference endpoints

    Useful for:
    - Using proprietary models
    - Scaling inference separately
    - A/B testing different backends

    NOT IMPLEMENTED - raises NotImplementedError.
    """

    @property
    def name(self) -> str:
        return "external"

    def load(self) -> None:
        """
        Would initialize HTTP client and validate connectivity.
        Currently raises NotImplementedError.
        """
        raise NotImplementedError(
            "External detector not implemented. "
            "Implement HTTP client for external inference API."
        )

    def predict(
        self,
        language: str,
        mp3_bytes: bytes,
        waveform: np.ndarray,
        sr: int,
        qc: Dict[str, float],
    ) -> PredictionResult:
        """Not implemented."""
        raise NotImplementedError("External detector predict() not implemented.")
