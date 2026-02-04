"""
AASIST Detector - Anti-spoofing model for AI-generated voice detection.

Uses a fine-tuned AASIST model exported as TorchScript for inference.
The model distinguishes between bona fide (human) and spoofed (AI-generated) speech.
"""

import logging
from typing import Dict

import numpy as np
import torch

from config import settings

from .base import BaseDetector, PredictionResult

logger = logging.getLogger(__name__)


class AASISTDetector(BaseDetector):
    """
    AASIST anti-spoofing detector for AI-generated voice detection.

    Loads a TorchScript model at startup and runs inference on audio waveforms.
    The model outputs logits for [spoof, bonafide] classes.
    """

    def __init__(self):
        self.model = None
        self.device = None
        self._demo_mode = False

    @property
    def name(self) -> str:
        return "aasist"

    def load(self) -> None:
        """
        Load the AASIST TorchScript model from disk.

        If the model file doesn't exist, enables demo mode with dummy responses.

        Raises:
            RuntimeError: If model file is .pth (requires architecture code)
        """
        model_path = settings.AASIST_MODEL_PATH
        device_str = settings.AASIST_DEVICE

        logger.info(f"Loading AASIST model from: {model_path}")
        logger.info(f"Using device: {device_str}")

        # Validate model file format
        if model_path.endswith(".pth"):
            raise RuntimeError(
                f"Model file '{model_path}' is a .pth checkpoint. "
                "Provide a TorchScript .pt file for deployment, "
                "or vendor the AASIST architecture code in this repo."
            )

        # Set device
        if device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            if device_str == "cuda":
                logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = torch.device("cpu")

        # Load TorchScript model (or enable demo mode if not found)
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            self._demo_mode = False
            logger.info(f"AASIST model loaded successfully on {self.device}")
        except (FileNotFoundError, ValueError) as e:
            # Model file doesn't exist - enable demo mode
            self._demo_mode = True
            self.model = None
            logger.warning(
                f"AASIST model not found at '{model_path}'. "
                "Running in DEMO MODE with dummy responses. "
                "Provide a valid model file for production use."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load AASIST model: {e}")

    def predict(
        self,
        language: str,
        mp3_bytes: bytes,
        waveform: np.ndarray,
        sr: int,
        qc: Dict[str, float],
    ) -> PredictionResult:
        """
        Run AASIST inference on the audio waveform.

        Args:
            language: Detected language (passed through, not used for inference).
            mp3_bytes: Original MP3 bytes (not used, waveform is preferred).
            waveform: Audio samples as float32 numpy array (mono, 16kHz).
            sr: Sample rate (expected 16000 Hz).
            qc: Quality control metrics dictionary.

        Returns:
            PredictionResult with classification, confidence, and explanation.
        """
        # Demo mode: return dummy responses
        if self._demo_mode or self.model is None:
            return self._predict_demo(language, qc)

        # Convert waveform to tensor: shape [1, T] (batch=1, time=T)
        # Waveform is already float32 normalized to [-1, 1] from audio_io.py
        audio_tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            # Model outputs logits: [batch, 2] where index 0=spoof, 1=bonafide
            logits = self.model(audio_tensor)

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)

            # Extract probabilities
            # Index 0 = spoof (AI-generated), Index 1 = bonafide (human)
            p_spoof = probs[0, 0].item()
            p_bonafide = probs[0, 1].item()

        # Classification based on threshold
        threshold = settings.AASIST_THRESHOLD

        if p_spoof > threshold:
            classification = "AI_GENERATED"
            confidence = p_spoof
            explanation = (
                f"AASIST anti-spoof model detected patterns consistent with "
                f"synthetic speech (confidence: {confidence:.0%})."
            )
        else:
            classification = "HUMAN"
            confidence = p_bonafide
            explanation = (
                f"AASIST anti-spoof model indicates bona fide human speech "
                f"(confidence: {confidence:.0%})."
            )

        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))

        return PredictionResult(
            classification=classification,
            confidenceScore=confidence,
            explanation=explanation,
        )

    def _predict_demo(self, language: str, qc: Dict[str, float]) -> PredictionResult:
        """
        Return dummy prediction for demo mode.
        
        Uses a simple heuristic based on audio RMS to vary responses.
        """
        import random
        
        # Use RMS to seed randomness for consistent results per audio
        rms = qc.get("rms", 0.1)
        random.seed(int(rms * 10000))
        
        is_ai = random.random() > 0.5
        
        if is_ai:
            confidence = round(random.uniform(0.70, 0.92), 2)
            explanation = (
                f"[DEMO] Detected synthetic speech patterns in {language} audio. "
                f"Model not loaded - using dummy response."
            )
            classification = "AI_GENERATED"
        else:
            confidence = round(random.uniform(0.68, 0.90), 2)
            explanation = (
                f"[DEMO] Natural speech patterns detected in {language} audio. "
                f"Model not loaded - using dummy response."
            )
            classification = "HUMAN"
        
        return PredictionResult(
            classification=classification,
            confidenceScore=confidence,
            explanation=explanation,
        )

