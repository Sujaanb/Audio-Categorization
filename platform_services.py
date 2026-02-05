"""
Platform Services - API endpoints for Voice Detection.
"""

import base64
import hmac
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from base_requests import (
    VoiceDetectionErrorResponse,
    VoiceDetectionRequest,
    VoiceDetectionSuccessResponse,
)
from config import settings
from detectors import get_detector
from services import AudioDecodeError, compute_qc_metrics, decode_mp3_to_waveform
from services.qc import is_insufficient_signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API Router - no prefix, tags for documentation
api_router = APIRouter(tags=["Voice Detection API"])

# API Key security scheme
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def create_error_response(status_code: int, message: str) -> JSONResponse:
    """Create a standardized error JSON response."""
    return JSONResponse(
        status_code=status_code,
        content=VoiceDetectionErrorResponse(message=message).model_dump(),
    )


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)) -> str:
    """
    Validate the API key from x-api-key header.

    Uses constant-time comparison to prevent timing attacks.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail=VoiceDetectionErrorResponse(
                message="Missing API key. Include x-api-key header."
            ).model_dump(),
        )

    valid_keys = settings.get_api_keys()

    if not valid_keys:
        # No keys configured - block all requests in production
        logger.warning("No API keys configured. Blocking request.")
        raise HTTPException(
            status_code=401,
            detail=VoiceDetectionErrorResponse(
                message="API authentication not configured."
            ).model_dump(),
        )

    # Constant-time comparison for each valid key (case-insensitive)
    # Normalize both incoming key and stored keys to lowercase
    is_valid = False
    api_key_lower = api_key.lower()
    for valid_key in valid_keys:
        if hmac.compare_digest(api_key_lower.encode("utf-8"), valid_key.lower().encode("utf-8")):
            is_valid = True
            break

    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail=VoiceDetectionErrorResponse(
                message="Invalid API key."
            ).model_dump(),
        )

    return api_key


@api_router.post(
    "/api/voice-detection",
    response_model=VoiceDetectionSuccessResponse,
    responses={
        400: {"model": VoiceDetectionErrorResponse, "description": "Bad Request"},
        401: {"model": VoiceDetectionErrorResponse, "description": "Unauthorized"},
        413: {"model": VoiceDetectionErrorResponse, "description": "Payload Too Large"},
        500: {"model": VoiceDetectionErrorResponse, "description": "Internal Server Error"},
    },
)
async def voice_detection(
    request: Request,
    body: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key),
) -> VoiceDetectionSuccessResponse:
    """
    Detect if audio is AI-generated or human speech.

    Accepts Base64-encoded MP3 audio and returns classification with confidence.

    Supported languages: Tamil, English, Hindi, Malayalam, Telugu
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Log request start (never log audio content)
    logger.info(
        f"[{request_id}] Voice detection request: language={body.language}"
    )

    try:
        # Step 1: Check base64 string length to prevent memory issues
        max_base64_len = settings.get_max_base64_length()
        if len(body.audioBase64) > max_base64_len:
            logger.warning(f"[{request_id}] Base64 string too long: {len(body.audioBase64)}")
            return create_error_response(
                413, f"Audio too large. Maximum base64 length: {max_base64_len}"
            )

        # Step 2: Decode base64 to bytes
        try:
            mp3_bytes = base64.b64decode(body.audioBase64, validate=True)
        except Exception as e:
            logger.warning(f"[{request_id}] Invalid base64: {str(e)}")
            return create_error_response(400, "Invalid base64 encoding in audioBase64 field.")

        # Step 3: Check decoded bytes size
        if len(mp3_bytes) > settings.MAX_MP3_BYTES:
            logger.warning(f"[{request_id}] MP3 too large: {len(mp3_bytes)} bytes")
            return create_error_response(
                413, f"Audio file too large. Maximum size: {settings.MAX_MP3_BYTES} bytes."
            )

        if len(mp3_bytes) == 0:
            logger.warning(f"[{request_id}] Empty audio data")
            return create_error_response(400, "Empty audio data provided.")

        # Step 4: Decode MP3 to waveform
        try:
            waveform, sr, duration = decode_mp3_to_waveform(mp3_bytes)
        except AudioDecodeError as e:
            logger.warning(f"[{request_id}] Audio decode failed: {str(e)}")
            return create_error_response(400, f"Failed to decode MP3: {str(e)}")

        # Step 5: Check duration limits
        if duration > settings.MAX_DURATION_SECONDS:
            logger.warning(f"[{request_id}] Audio too long: {duration:.2f}s")
            return create_error_response(
                400, f"Audio too long ({duration:.1f}s). Maximum: {settings.MAX_DURATION_SECONDS}s."
            )

        # Step 6: Compute QC metrics
        qc_metrics = compute_qc_metrics(waveform, sr)

        logger.info(
            f"[{request_id}] Audio decoded: duration={duration:.2f}s, "
            f"rms={qc_metrics['rms']:.4f}, silence_ratio={qc_metrics['silence_ratio']:.2f}"
        )

        # Step 7: Check if audio has insufficient signal (QC fallback)
        if is_insufficient_signal(
            qc_metrics,
            min_duration_seconds=settings.MIN_DURATION_SECONDS,
            silence_ratio_threshold=settings.SILENCE_RATIO_THRESHOLD,
        ):
            # Return low-confidence success response
            logger.info(f"[{request_id}] Insufficient signal - returning low-confidence result")

            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                f"[{request_id}] Completed: classification=HUMAN (low-conf), "
                f"latency={latency_ms:.0f}ms"
            )

            explanation = "Insufficient audio signal for reliable detection. "
            if qc_metrics["duration_seconds"] < settings.MIN_DURATION_SECONDS:
                explanation += f"Audio too short ({qc_metrics['duration_seconds']:.2f}s)."
            else:
                explanation += f"High silence ratio ({qc_metrics['silence_ratio']*100:.0f}%)."

            return VoiceDetectionSuccessResponse(
                language=body.language,
                classification="HUMAN",
                confidenceScore=round(0.50 + (qc_metrics["rms"] * 0.5), 2),  # 0.50-0.55 range
                explanation=explanation,
            )

        # Step 8: Call detector for classification
        detector = get_detector()
        result = detector.predict(
            language=body.language,
            mp3_bytes=mp3_bytes,
            waveform=waveform,
            sr=sr,
            qc=qc_metrics,
        )

        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[{request_id}] Completed: classification={result.classification}, "
            f"confidence={result.confidenceScore:.2f}, latency={latency_ms:.0f}ms"
        )

        return VoiceDetectionSuccessResponse(
            language=body.language,
            classification=result.classification,
            confidenceScore=round(result.confidenceScore, 2),
            explanation=result.explanation,
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.exception(f"[{request_id}] Unhandled error after {latency_ms:.0f}ms: {str(e)}")
        return create_error_response(500, "Internal server error. Please try again later.")