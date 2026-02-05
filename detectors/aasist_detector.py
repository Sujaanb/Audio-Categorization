"""
AASIST Detector - Single model for AI-generated voice detection.

Uses the original AASIST model for inference across all supported languages.
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from config import settings
from models.AASIST import Model as AASISTModel

from .base import BaseDetector, PredictionResult

logger = logging.getLogger(__name__)

# AASIST model configuration (fixed for all weights)
AASIST_MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


class AASISTDetector(BaseDetector):
    """
    AASIST-based detector for AI-generated voice detection.
    
    Uses single model inference with multi-window sampling for long audio.
    """

    def __init__(self):
        self.model = None
        self.device = None

    @property
    def name(self) -> str:
        return "aasist"

    def load(self) -> None:
        """Load AASIST model from local path."""
        device_str = settings.AASIST_DEVICE
        logger.info(f"Initializing AASIST detector on device: {device_str}")

        # Set device
        if device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            if device_str == "cuda":
                logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = torch.device("cpu")

        # Load model
        weights_path = Path(settings.AASIST_WEIGHTS_PATH)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")
        
        logger.info(f"Loading AASIST model from: {weights_path}")
        self.model = AASISTModel(AASIST_MODEL_CONFIG).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        logger.info("AASIST model loaded successfully")

    def predict(
        self,
        language: str,
        mp3_bytes: bytes,
        waveform: np.ndarray,
        sr: int,
        qc: Dict[str, float],
    ) -> PredictionResult:
        """
        Run inference on audio waveform.

        Args:
            language: Input language (Tamil, English, Hindi, Malayalam, Telugu)
            mp3_bytes: Original MP3 bytes (not used)
            waveform: Audio as float32 numpy array (mono, 16kHz)
            sr: Sample rate (expected 16000)
            qc: Quality control metrics

        Returns:
            PredictionResult with classification, confidence, and explanation
        """
        # Get windows for multi-window inference
        windows = self._get_audio_windows(waveform)

        # Collect scores from all windows
        all_fake_scores = []
        all_real_scores = []

        for window in windows:
            audio_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, output = self.model(audio_tensor)
                probs = torch.softmax(output, dim=1)
                fake_prob = probs[0, 0].item()  # index 0 = spoof/fake
                real_prob = probs[0, 1].item()  # index 1 = real

            all_fake_scores.append(fake_prob)
            all_real_scores.append(real_prob)

        # Average scores across windows
        avg_fake = np.mean(all_fake_scores)
        avg_real = np.mean(all_real_scores)

        logger.info(f"Model scores - fake={avg_fake:.2%}, real={avg_real:.2%}")

        # Classification based on threshold
        threshold = settings.AASIST_THRESHOLD
        is_fake = avg_fake > threshold

        # Build result
        if is_fake:
            classification = "AI_GENERATED"
            confidence = round(avg_fake, 2)
            explanation = f"Detected AI-generated speech. Spoof probability: {avg_fake:.0%}"
        else:
            classification = "HUMAN"
            confidence = round(avg_real, 2)
            explanation = f"Detected human speech. Real probability: {avg_real:.0%}"

        return PredictionResult(
            classification=classification,
            confidenceScore=confidence,
            explanation=explanation,
        )

    def _get_audio_windows(self, waveform: np.ndarray) -> List[np.ndarray]:
        """Extract audio windows for multi-window inference."""
        nb_samp = AASIST_MODEL_CONFIG["nb_samp"]  # 64600
        max_windows = settings.AASIST_MAX_WINDOWS

        if len(waveform) <= nb_samp:
            if len(waveform) >= nb_samp:
                window = waveform[:nb_samp]
            else:
                num_repeats = int(np.ceil(nb_samp / len(waveform)))
                window = np.tile(waveform, num_repeats)[:nb_samp]
            return [window.astype(np.float32)]

        # Long audio: sample evenly-spaced windows
        total_len = len(waveform)
        n_windows = min(max_windows, total_len // nb_samp)
        n_windows = max(1, n_windows)

        windows = []
        step = (total_len - nb_samp) // max(1, n_windows - 1) if n_windows > 1 else 0

        for i in range(n_windows):
            start = i * step
            end = start + nb_samp
            if end <= total_len:
                windows.append(waveform[start:end].astype(np.float32))

        return windows if windows else [waveform[:nb_samp].astype(np.float32)]
