"""
Detector registry - singleton pattern for AASIST detector initialization.
"""

import logging
from typing import Optional

from .base import BaseDetector

logger = logging.getLogger(__name__)

# Singleton detector instance
_detector_instance: Optional[BaseDetector] = None


def get_detector() -> BaseDetector:
    """
    Get the initialized AASIST detector instance.

    Returns:
        The current detector instance.

    Raises:
        RuntimeError: If detector has not been initialized.
    """
    if _detector_instance is None:
        raise RuntimeError(
            "Detector not initialized. Call initialize_detector() at app startup."
        )
    return _detector_instance


def initialize_detector() -> BaseDetector:
    """
    Initialize the AASIST detector.

    Loads the AASIST model and stores it as a singleton for the application lifetime.
    This should be called once at application startup.

    Returns:
        Initialized AASIST detector instance.

    Raises:
        RuntimeError: If model loading fails.
        FileNotFoundError: If model file doesn't exist.
    """
    global _detector_instance

    logger.info("Initializing AASIST detector")

    from .aasist_detector import AASISTDetector

    detector = AASISTDetector()

    # Load the model (will raise on failure)
    detector.load()

    _detector_instance = detector
    logger.info(f"Detector '{detector.name}' initialized successfully")

    return detector


def reset_detector() -> None:
    """Reset the detector instance (primarily for testing)."""
    global _detector_instance
    _detector_instance = None
