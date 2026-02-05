"""
GCS weights download utility for AASIST ensemble models.

Downloads model weights from Google Cloud Storage to local cache,
with simple locking to prevent concurrent downloads in the same instance.
"""

import logging
import os
import time
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    """
    Parse a GCS URI into bucket and blob path.

    Args:
        gcs_uri: URI in format gs://bucket-name/path/to/file

    Returns:
        Tuple of (bucket_name, blob_path)

    Raises:
        ValueError: If URI format is invalid
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI format: {gcs_uri}. Must start with 'gs://'")

    # Remove gs:// prefix
    path = gcs_uri[5:]

    # Split bucket and blob path
    parts = path.split("/", 1)
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"Invalid GCS URI format: {gcs_uri}. Must include bucket and blob path")

    bucket_name = parts[0]
    blob_path = parts[1]

    return bucket_name, blob_path


def download_weights(gcs_uri: str, local_path: str) -> str:
    """
    Download model weights from GCS to local cache.

    Features:
    - Skip download if local file exists and is non-empty
    - Download to temp file first, then atomic rename
    - Simple lockfile to prevent concurrent downloads

    Args:
        gcs_uri: GCS URI (gs://bucket/path) or empty string to skip
        local_path: Local path to save weights

    Returns:
        Local path to the weights file

    Raises:
        RuntimeError: If download fails
    """
    if not gcs_uri:
        logger.info(f"No GCS URI provided, expecting local file at: {local_path}")
        return local_path

    local_path = Path(local_path)
    lock_path = Path(f"{local_path}.lock")
    tmp_path = Path(f"{local_path}.tmp")

    # Check if already downloaded
    if local_path.exists() and local_path.stat().st_size > 0:
        logger.info(f"Weights already cached at: {local_path}")
        return str(local_path)

    # Ensure parent directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Simple lockfile mechanism
    max_wait = 300  # 5 minutes
    wait_time = 0

    while lock_path.exists() and wait_time < max_wait:
        logger.info(f"Waiting for lock: {lock_path}")
        time.sleep(5)
        wait_time += 5

        # Check if download completed while waiting
        if local_path.exists() and local_path.stat().st_size > 0:
            logger.info(f"Weights downloaded by another process: {local_path}")
            return str(local_path)

    # Acquire lock
    try:
        lock_path.touch()

        # Check again after acquiring lock
        if local_path.exists() and local_path.stat().st_size > 0:
            logger.info(f"Weights already cached: {local_path}")
            return str(local_path)

        # Parse GCS URI
        bucket_name, blob_path = parse_gcs_uri(gcs_uri)
        logger.info(f"Downloading weights from gs://{bucket_name}/{blob_path}")

        # Import GCS client (lazy import to avoid startup overhead if not needed)
        try:
            from google.cloud import storage
        except ImportError:
            raise RuntimeError(
                "google-cloud-storage package not installed. "
                "Run: pip install google-cloud-storage"
            )

        # Download from GCS
        start_time = time.time()
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Download to temp file
        blob.download_to_filename(str(tmp_path))

        # Atomic rename
        tmp_path.rename(local_path)

        download_time = time.time() - start_time
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Downloaded {file_size_mb:.1f}MB in {download_time:.1f}s: {local_path}"
        )

        return str(local_path)

    except Exception as e:
        # Clean up temp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download weights from {gcs_uri}: {e}")

    finally:
        # Release lock
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass
