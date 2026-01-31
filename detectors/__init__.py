"""Detector plugin system for voice detection."""

from .base import BaseDetector, PredictionResult
from .registry import get_detector, initialize_detector, reset_detector

__all__ = [
    "BaseDetector",
    "PredictionResult",
    "get_detector",
    "initialize_detector",
    "reset_detector",
]
