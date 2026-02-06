"""
Audio I/O utilities for decoding MP3 to waveform using ffmpeg.
Uses asyncio.to_thread() to run subprocess without blocking the event loop.
"""

import asyncio
import subprocess
from typing import Tuple

import numpy as np


class AudioDecodeError(Exception):
    """Raised when audio decoding fails."""

    pass


def _decode_mp3_sync(mp3_bytes: bytes, target_sr: int) -> Tuple[np.ndarray, int, float]:
    """
    Synchronous MP3 decode - runs in thread pool.
    """
    if not mp3_bytes:
        raise AudioDecodeError("Empty audio bytes provided")

    # ffmpeg command to decode MP3 to raw PCM
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]

    try:
        process = subprocess.run(
            cmd,
            input=mp3_bytes,
            capture_output=True,
            timeout=60,
        )

        if process.returncode != 0:
            error_msg = process.stderr.decode("utf-8", errors="replace").strip()
            raise AudioDecodeError(f"ffmpeg decode failed: {error_msg or 'Unknown error'}")

        pcm_bytes = process.stdout

        if len(pcm_bytes) == 0:
            raise AudioDecodeError("ffmpeg produced no output")

        # Convert bytes to numpy array
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        waveform = audio_int16.astype(np.float32) / 32768.0
        duration_seconds = len(waveform) / target_sr

        return waveform, target_sr, duration_seconds

    except subprocess.TimeoutExpired:
        raise AudioDecodeError("ffmpeg decode timed out after 60 seconds")
    except FileNotFoundError:
        raise AudioDecodeError("ffmpeg not found. Please install ffmpeg.")
    except AudioDecodeError:
        raise
    except Exception as e:
        raise AudioDecodeError(f"Unexpected error during decode: {type(e).__name__}: {str(e)}")


async def decode_mp3_to_waveform(
    mp3_bytes: bytes,
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int, float]:
    """
    Decode MP3 bytes to mono PCM float32 waveform using ffmpeg (async).

    Runs the blocking subprocess in a thread pool to avoid blocking the event loop.

    Args:
        mp3_bytes: Raw MP3 file bytes.
        target_sr: Target sample rate (default 16000 Hz).

    Returns:
        Tuple of (waveform, sample_rate, duration_seconds)
        - waveform: numpy array of float32 samples normalized to [-1, 1]
        - sample_rate: the output sample rate (target_sr)
        - duration_seconds: audio duration in seconds

    Raises:
        AudioDecodeError: If ffmpeg fails to decode the audio.
    """
    # Run blocking subprocess in thread pool to avoid blocking event loop
    return await asyncio.to_thread(_decode_mp3_sync, mp3_bytes, target_sr)
