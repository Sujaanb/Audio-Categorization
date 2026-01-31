"""
Quality Control (QC) metrics for audio analysis.
Computes duration, RMS, silence ratio, and clipping ratio.
"""

from typing import Dict

import numpy as np


def compute_qc_metrics(waveform: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Compute quality control metrics for an audio waveform.

    Args:
        waveform: Audio samples as float32 numpy array (normalized to [-1, 1]).
        sr: Sample rate in Hz.

    Returns:
        Dictionary with QC metrics:
        - duration_seconds: Total duration of the audio
        - rms: Root Mean Square energy level
        - silence_ratio: Fraction of samples below silence threshold
        - clipping_ratio: Fraction of samples at or near max amplitude
    """
    if len(waveform) == 0:
        return {
            "duration_seconds": 0.0,
            "rms": 0.0,
            "silence_ratio": 1.0,
            "clipping_ratio": 0.0,
        }

    # Duration
    duration_seconds = len(waveform) / sr

    # RMS (Root Mean Square) - measure of overall energy
    rms = float(np.sqrt(np.mean(waveform**2)))

    # Silence detection
    # Threshold: samples with absolute value < 0.01 are considered silence
    silence_threshold = 0.01
    silent_samples = np.sum(np.abs(waveform) < silence_threshold)
    silence_ratio = float(silent_samples / len(waveform))

    # Clipping detection
    # Samples with absolute value > 0.99 are considered clipped
    clipping_threshold = 0.99
    clipped_samples = np.sum(np.abs(waveform) > clipping_threshold)
    clipping_ratio = float(clipped_samples / len(waveform))

    return {
        "duration_seconds": duration_seconds,
        "rms": rms,
        "silence_ratio": silence_ratio,
        "clipping_ratio": clipping_ratio,
    }


def is_insufficient_signal(
    qc_metrics: Dict[str, float],
    min_duration_seconds: float,
    silence_ratio_threshold: float,
) -> bool:
    """
    Determine if the audio has insufficient signal for reliable detection.

    Args:
        qc_metrics: Dictionary from compute_qc_metrics.
        min_duration_seconds: Minimum required duration.
        silence_ratio_threshold: Maximum allowed silence ratio.

    Returns:
        True if the audio is insufficient for detection.
    """
    # Too short
    if qc_metrics["duration_seconds"] < min_duration_seconds:
        return True

    # Too much silence
    if qc_metrics["silence_ratio"] >= silence_ratio_threshold:
        return True

    return False
