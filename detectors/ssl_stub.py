"""
SSL-based detector stub.
Placeholder for Self-Supervised Learning based detection approach.
"""

from typing import Dict

import numpy as np

from .base import BaseDetector, PredictionResult


class SSLDetector(BaseDetector):
    """
    Self-Supervised Learning based voice detection stub.

    This detector would use pre-trained SSL models like:
    - Wav2Vec 2.0
    - HuBERT
    - WavLM

    These models learn representations from unlabeled audio and can be
    fine-tuned for AI-generated voice detection.

    NOT IMPLEMENTED - raises NotImplementedError.
    """

    @property
    def name(self) -> str:
        return "ssl"

    def load(self) -> None:
        """
        Would load the pre-trained SSL model and any fine-tuned weights.
        Currently raises NotImplementedError.
        """
        raise NotImplementedError(
            "SSL detector not implemented. "
            "Implement Wav2Vec2/HuBERT-based feature extraction and classification."
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
        raise NotImplementedError("SSL detector predict() not implemented.")
