"""
Ensemble detector stub.
Placeholder for combining multiple detection approaches.
"""

from typing import Dict

import numpy as np

from .base import BaseDetector, PredictionResult


class EnsembleDetector(BaseDetector):
    """
    Ensemble voice detection stub.

    This detector would combine multiple approaches:
    - DSP features
    - SSL embeddings
    - Anti-spoofing predictions

    Using ensemble methods like:
    - Weighted voting
    - Stacking
    - Late fusion

    NOT IMPLEMENTED - raises NotImplementedError.
    """

    @property
    def name(self) -> str:
        return "ensemble"

    def load(self) -> None:
        """
        Would load all component models and ensemble weights.
        Currently raises NotImplementedError.
        """
        raise NotImplementedError(
            "Ensemble detector not implemented. "
            "Implement multi-model ensemble with weighted fusion."
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
        raise NotImplementedError("Ensemble detector predict() not implemented.")
