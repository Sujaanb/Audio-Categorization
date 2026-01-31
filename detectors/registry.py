"""
Detector registry - singleton pattern for detector initialization.
"""

import logging
from typing import Optional

from .base import BaseDetector

logger = logging.getLogger(__name__)

# Singleton detector instance
_detector_instance: Optional[BaseDetector] = None


def get_detector() -> BaseDetector:
    """
    Get the initialized detector instance.

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


def initialize_detector(backend: str) -> BaseDetector:
    """
    Initialize the detector based on backend name.

    Args:
        backend: Name of the detector backend to use.
                 Options: "qc_fallback", "dsp", "ssl", "antispoof", "ensemble", "external"

    Returns:
        Initialized detector instance.

    Raises:
        ValueError: If backend name is not recognized.
    """
    global _detector_instance

    logger.info(f"Initializing detector backend: {backend}")

    # Import detectors lazily to avoid circular imports
    if backend == "qc_fallback":
        from .qc_fallback import QCFallbackDetector
        detector = QCFallbackDetector()

    elif backend == "dsp":
        from .dsp_stub import DSPDetector
        detector = DSPDetector()

    elif backend == "ssl":
        from .ssl_stub import SSLDetector
        detector = SSLDetector()

    elif backend == "antispoof":
        from .antispoof_stub import AntispoofDetector
        detector = AntispoofDetector()

    elif backend == "ensemble":
        from .ensemble_stub import EnsembleDetector
        detector = EnsembleDetector()

    elif backend == "external":
        from .external_stub import ExternalDetector
        detector = ExternalDetector()

    else:
        raise ValueError(
            f"Unknown detector backend: {backend}. "
            f"Valid options: qc_fallback, dsp, ssl, antispoof, ensemble, external"
        )

    # Call load() to initialize the detector
    detector.load()

    _detector_instance = detector
    logger.info(f"Detector '{detector.name}' initialized successfully")

    return detector


def reset_detector() -> None:
    """Reset the detector instance (primarily for testing)."""
    global _detector_instance
    _detector_instance = None
