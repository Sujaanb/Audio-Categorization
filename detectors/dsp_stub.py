"""
DSP-based detector stub.
Placeholder for Digital Signal Processing based detection approach.
"""

from typing import Dict

import numpy as np

from .base import BaseDetector, PredictionResult


class DSPDetector(BaseDetector):
    """
    DSP-based voice detection stub.

    This detector would use traditional DSP features like:
    - Spectral analysis (MFCC, spectral centroid, rolloff)
    - Pitch analysis (F0, jitter, shimmer)
    - Formant analysis
    - Voice activity detection

    NOT IMPLEMENTED - raises NotImplementedError.
    """

    @property
    def name(self) -> str:
        return "dsp"

    def load(self) -> None:
        """
        Would load any pre-computed statistics or thresholds.
        Currently raises NotImplementedError.
        """
        raise NotImplementedError(
            "DSP detector not implemented. "
            "Implement spectral and prosodic feature extraction in this class."
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
        raise NotImplementedError("DSP detector predict() not implemented.")
