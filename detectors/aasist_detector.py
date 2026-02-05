"""
AASIST Ensemble Detector - Two-model ensemble for AI-generated voice detection.

Uses original AASIST and fine-tuned AASIST models with heuristic pattern rules
for improved multilingual detection (English + Indian languages).
"""

import logging
from typing import Dict, List, Tuple

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


class AASISTEnsembleDetector(BaseDetector):
    """
    Ensemble of original and fine-tuned AASIST models for deepfake detection.

    Uses pattern-based heuristics to handle inverted detection for Indian languages
    where the original English-trained model gives reversed predictions.
    """

    def __init__(self):
        self.original_model = None
        self.finetuned_model = None
        self.device = None
        self._demo_mode = False

    @property
    def name(self) -> str:
        return "aasist_ensemble"

    def load(self) -> None:
        """
        Load both AASIST models from GCS or local cache.

        Downloads weights from GCS if URIs are provided and files don't exist locally.
        Falls back to demo mode if weights are not available.
        """
        from services.gcs_weights import download_weights

        device_str = settings.AASIST_DEVICE
        logger.info(f"Initializing AASIST ensemble detector on device: {device_str}")

        # Set device
        if device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            if device_str == "cuda":
                logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = torch.device("cpu")

        # Download/load original model weights
        orig_weights_path = settings.AASIST_ORIG_CACHE_PATH
        orig_gcs_uri = settings.AASIST_ORIG_WEIGHTS_GCS_URI

        # Download/load finetuned model weights
        ft_weights_path = settings.AASIST_FT_CACHE_PATH
        ft_gcs_uri = settings.AASIST_FT_WEIGHTS_GCS_URI

        try:
            # Download weights if needed
            if orig_gcs_uri:
                orig_weights_path = download_weights(orig_gcs_uri, orig_weights_path)
            if ft_gcs_uri:
                ft_weights_path = download_weights(ft_gcs_uri, ft_weights_path)

            # Load original model
            logger.info(f"Loading original AASIST model from: {orig_weights_path}")
            self.original_model = self._load_model(orig_weights_path)

            # Load finetuned model
            logger.info(f"Loading fine-tuned AASIST model from: {ft_weights_path}")
            self.finetuned_model = self._load_model(ft_weights_path)

            self._demo_mode = False
            logger.info("AASIST ensemble models loaded successfully")

        except Exception as e:
            logger.warning(
                f"Failed to load AASIST models: {e}. Running in DEMO MODE."
            )
            self._demo_mode = True
            self.original_model = None
            self.finetuned_model = None

    def _load_model(self, weights_path: str) -> torch.nn.Module:
        """Load a single AASIST model from weights file."""
        model = AASISTModel(AASIST_MODEL_CONFIG).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def predict(
        self,
        language: str,
        mp3_bytes: bytes,
        waveform: np.ndarray,
        sr: int,
        qc: Dict[str, float],
    ) -> PredictionResult:
        """
        Run ensemble inference on audio waveform.

        Applies multi-window sampling for long audio and uses heuristic
        pattern rules to handle inverted detection for Indian languages.

        Args:
            language: Input language (Tamil, English, Hindi, Malayalam, Telugu)
            mp3_bytes: Original MP3 bytes (not used)
            waveform: Audio as float32 numpy array (mono, 16kHz)
            sr: Sample rate (expected 16000)
            qc: Quality control metrics

        Returns:
            PredictionResult with classification, confidence, and explanation
        """
        if self._demo_mode:
            return self._predict_demo(language, qc)

        # Get windows for multi-window inference
        windows = self._get_audio_windows(waveform)

        # Aggregate scores from all windows
        all_orig_scores = []
        all_ft_scores = []

        for window in windows:
            audio_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Original model scores
                _, orig_output = self.original_model(audio_tensor)
                orig_probs = torch.softmax(orig_output, dim=1)
                orig_fake = orig_probs[0, 0].item()  # index 0 = spoof/fake
                orig_real = orig_probs[0, 1].item()  # index 1 = real

                # Finetuned model scores
                _, ft_output = self.finetuned_model(audio_tensor)
                ft_probs = torch.softmax(ft_output, dim=1)
                ft_fake = ft_probs[0, 0].item()
                ft_real = ft_probs[0, 1].item()

            all_orig_scores.append((orig_real, orig_fake))
            all_ft_scores.append((ft_real, ft_fake))

        # Average scores across windows
        avg_orig_real = np.mean([s[0] for s in all_orig_scores])
        avg_orig_fake = np.mean([s[1] for s in all_orig_scores])
        avg_ft_real = np.mean([s[0] for s in all_ft_scores])
        avg_ft_fake = np.mean([s[1] for s in all_ft_scores])

        # Apply ensemble heuristics
        result = self._apply_ensemble_heuristics(
            avg_orig_real, avg_orig_fake, avg_ft_real, avg_ft_fake, language
        )

        return result

    def _get_audio_windows(self, waveform: np.ndarray) -> List[np.ndarray]:
        """
        Extract audio windows for multi-window inference.

        For short audio, pads/tiles to target length.
        For long audio, samples N evenly-spaced windows.

        Args:
            waveform: Input waveform as numpy array

        Returns:
            List of audio windows, each of length nb_samp
        """
        nb_samp = AASIST_MODEL_CONFIG["nb_samp"]  # 64600
        max_windows = settings.AASIST_MAX_WINDOWS

        if len(waveform) <= nb_samp:
            # Pad or tile short audio
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

    def _apply_ensemble_heuristics(
        self,
        orig_real: float,
        orig_fake: float,
        ft_real: float,
        ft_fake: float,
        language: str,
    ) -> PredictionResult:
        """
        Apply heuristic pattern rules for ensemble classification.

        Handles the inverted detection pattern for Indian languages where
        the original English-trained model gives reversed predictions.

        Pattern logic:
        1. orig_fake > 95% + ft_real > 95% → REAL Indian (inverted)
        2. orig_fake < 15% + ft_real > 80% → FAKE Indian (inverted)
        3. orig_fake > 92% → FAKE English
        4. orig_fake 85-92% + ft_real > 95% → REAL English
        5. Default fallback
        """
        detected_pattern = ""
        threshold = settings.AASIST_THRESHOLD

        # Pattern 1: Very high fake from original (>95%) + Fine-tuned confident real
        # This indicates REAL Indian language audio (inverted detection)
        if orig_fake > 0.95 and ft_real > 0.95:
            is_fake = False
            confidence = ft_real
            detected_pattern = "Indian REAL (inverted detection)"

        # Pattern 2: Very low fake from original (<15%) + Fine-tuned says real
        # This indicates FAKE Indian language audio (inverted detection)
        elif orig_fake < 0.15 and ft_real > 0.80:
            is_fake = True
            confidence = 1 - orig_fake  # Invert the score
            detected_pattern = "Indian FAKE (inverted detection)"

        # Pattern 3: Original confident fake (>92%) - likely English fake
        elif orig_fake > 0.92:
            is_fake = True
            confidence = orig_fake
            detected_pattern = "English FAKE (direct detection)"

        # Pattern 4: Original moderate-high (85-92%) + Fine-tuned very confident real
        # Likely English real audio
        elif orig_fake > 0.85 and ft_real > 0.95:
            is_fake = False
            confidence = ft_real
            detected_pattern = "English REAL (threshold)"

        # Pattern 5: Default fallback
        else:
            # Use fine-tuned model's confidence if very high
            if ft_real > 0.99 and orig_fake < 0.91:
                is_fake = False
                confidence = ft_real
                detected_pattern = "Default REAL (fine-tuned confident)"
            else:
                # Weighted decision
                is_fake = orig_fake > threshold
                confidence = orig_fake if is_fake else orig_real
                detected_pattern = "Default (weighted)"

        # Build classification and explanation
        classification = "AI_GENERATED" if is_fake else "HUMAN"
        confidence = max(0.0, min(1.0, confidence))

        explanation = (
            f"Ensemble: {detected_pattern}. "
            f"Orig(fake={orig_fake:.0%}), FT(real={ft_real:.0%})"
        )

        # Truncate explanation to 200 chars
        if len(explanation) > 200:
            explanation = explanation[:197] + "..."

        return PredictionResult(
            classification=classification,
            confidenceScore=confidence,
            explanation=explanation,
        )

    def _predict_demo(self, language: str, qc: Dict[str, float]) -> PredictionResult:
        """Return dummy prediction for demo mode."""
        import random

        # Use RMS to seed randomness for consistent results per audio
        rms = qc.get("rms", 0.1)
        random.seed(int(rms * 10000))

        is_ai = random.random() > 0.5

        if is_ai:
            confidence = round(random.uniform(0.70, 0.92), 2)
            explanation = (
                f"[DEMO] Detected synthetic speech patterns in {language} audio. "
                f"Models not loaded - using dummy response."
            )
            classification = "AI_GENERATED"
        else:
            confidence = round(random.uniform(0.68, 0.90), 2)
            explanation = (
                f"[DEMO] Natural speech patterns in {language} audio. "
                f"Models not loaded - using dummy response."
            )
            classification = "HUMAN"

        return PredictionResult(
            classification=classification,
            confidenceScore=confidence,
            explanation=explanation,
        )
