"""
QC Fallback Detector - returns dummy/test data for development.
TODO: Remove dummy logic when actual model is implemented.
"""

import random
from typing import Dict

import numpy as np

from .base import BaseDetector, PredictionResult


class QCFallbackDetector(BaseDetector):
    """
    QC-based fallback detector with dummy data for testing.

    TODO: This currently returns DUMMY DATA for testing purposes.
    Replace with actual detection logic when model is ready.
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
        Return DUMMY test data for development.
        TODO: Replace with actual model inference when ready.
        """
        duration = qc.get("duration_seconds", 0.0)
        silence_ratio = qc.get("silence_ratio", 1.0)
        rms = qc.get("rms", 0.0)

        # Check for insufficient signal conditions - return low-confidence
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

        # =============================================================
        # DUMMY DATA FOR TESTING - TODO: Remove when model is ready
        # =============================================================
        # Randomly classify as AI or HUMAN with varying confidence
        is_ai = random.random() > 0.5
        
        if is_ai:
            confidence = round(random.uniform(0.70, 0.95), 2)
            explanations = [
                f"[DUMMY] Detected synthetic speech patterns in {language} audio.",
                f"[DUMMY] Spectral analysis indicates AI-generated voice characteristics.",
                f"[DUMMY] Neural TTS artifacts detected with {confidence*100:.0f}% confidence.",
                f"[DUMMY] Voice synthesis markers found in frequency domain.",
            ]
        else:
            confidence = round(random.uniform(0.65, 0.92), 2)
            explanations = [
                f"[DUMMY] Natural speech patterns detected in {language} audio.",
                f"[DUMMY] Human voice characteristics confirmed.",
                f"[DUMMY] No synthetic artifacts detected. Appears to be human speech.",
                f"[DUMMY] Audio analysis indicates genuine human voice.",
            ]

        return PredictionResult(
            classification="AI_GENERATED" if is_ai else "HUMAN",
            confidenceScore=confidence,
            explanation=random.choice(explanations),
        )
