"""
Anti-spoofing detector stub.
Placeholder for speaker verification anti-spoofing based detection.
"""

from typing import Dict

import numpy as np

from .base import BaseDetector, PredictionResult


class AntispoofDetector(BaseDetector):
    """
    Anti-spoofing based voice detection stub.

    This detector would use anti-spoofing models from ASVspoof challenges:
    - LCNN (Light CNN)
    - ResNet variants
    - AASIST
    - RawNet2

    These models are trained to distinguish bonafide from spoofed speech.

    NOT IMPLEMENTED - raises NotImplementedError.
    """

    @property
    def name(self) -> str:
        return "antispoof"

    def load(self) -> None:
        """
        Would load the anti-spoofing model weights.
        Currently raises NotImplementedError.
        """
        raise NotImplementedError(
            "Antispoof detector not implemented. "
            "Implement AASIST/RawNet2-based spoofing detection."
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
        raise NotImplementedError("Antispoof detector predict() not implemented.")
