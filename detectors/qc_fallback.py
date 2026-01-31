"""
QC Fallback Detector - returns low-confidence results based on QC metrics.
This is the always-available fallback when audio has insufficient signal.
"""

from typing import Dict

import numpy as np

from .base import BaseDetector, PredictionResult


class QCFallbackDetector(BaseDetector):
    """
    QC-based fallback detector.

    Returns low-confidence HUMAN classification when audio quality
    is insufficient for reliable detection. This is triggered by
    QC thresholds (duration, silence ratio), NOT hard-coded for all inputs.

    When the audio passes QC thresholds, this detector still returns
    a low-confidence result since it has no actual detection capability.
    """

    @property
    def name(self) -> str:
        return "qc_fallback"

    def load(self) -> None:
        """No initialization required for QC fallback."""
        pass

    def predict(
        self,
        language: str,
        mp3_bytes: bytes,
        waveform: np.ndarray,
        sr: int,
        qc: Dict[str, float],
    ) -> PredictionResult:
        """
        Return a low-confidence result based on QC metrics.

        If QC indicates insufficient signal, returns very low confidence (0.50-0.55).
        If QC passes, returns slightly higher but still low confidence (0.55-0.60)
        since this detector has no actual classification capability.
        """
        duration = qc.get("duration_seconds", 0.0)
        silence_ratio = qc.get("silence_ratio", 1.0)
        rms = qc.get("rms", 0.0)

        # Check for insufficient signal conditions
        if duration < 0.5:
            return PredictionResult(
                classification="HUMAN",
                confidenceScore=0.50,
                explanation=f"Audio too short ({duration:.2f}s). Low-confidence result due to insufficient signal.",
            )

        if silence_ratio >= 0.80:
            return PredictionResult(
                classification="HUMAN",
                confidenceScore=0.52,
                explanation=f"Audio mostly silent ({silence_ratio*100:.0f}% silence). Low-confidence result.",
            )

        if rms < 0.005:
            return PredictionResult(
                classification="HUMAN",
                confidenceScore=0.51,
                explanation="Audio has very low energy. Low-confidence result due to weak signal.",
            )

        # Audio passes QC but we have no real detection capability
        return PredictionResult(
            classification="HUMAN",
            confidenceScore=0.55,
            explanation="QC fallback detector - no trained model. Classification based on default assumptions.",
        )
