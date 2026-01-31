"""
Base detector interface and prediction result dataclass.
All detector implementations must inherit from BaseDetector.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np


@dataclass
class PredictionResult:
    """Result from a detector prediction."""

    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float  # 0.0 to 1.0
    explanation: str  # max 200 chars

    def __post_init__(self):
        """Validate prediction result fields."""
        if not 0.0 <= self.confidenceScore <= 1.0:
            raise ValueError(f"confidenceScore must be between 0 and 1, got {self.confidenceScore}")
        if len(self.explanation) > 200:
            self.explanation = self.explanation[:197] + "..."


class BaseDetector(ABC):
    """
    Abstract base class for voice detection backends.

    Subclasses must implement:
    - name: str property identifying the detector
    - predict(): perform classification on audio

    Optionally override:
    - load(): initialize models/resources at startup
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this detector."""
        pass

    def load(self) -> None:
        """
        Initialize the detector (load models, weights, etc.).
        Called once at application startup.
        Override in subclass if needed.
        """
        pass

    @abstractmethod
    def predict(
        self,
        language: str,
        mp3_bytes: bytes,
        waveform: np.ndarray,
        sr: int,
        qc: Dict[str, float],
    ) -> PredictionResult:
        """
        Classify audio as AI-generated or human.

        Args:
            language: Detected language (Tamil, English, Hindi, Malayalam, Telugu).
            mp3_bytes: Original MP3 file bytes.
            waveform: Decoded audio as float32 numpy array (mono, normalized).
            sr: Sample rate of the waveform.
            qc: Quality control metrics dictionary.

        Returns:
            PredictionResult with classification, confidence, and explanation.
        """
        pass
