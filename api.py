"""
AI-Generated Voice Detection API

FastAPI application for detecting AI-generated vs human speech.
Supports Tamil, English, Hindi, Malayalam, and Telugu languages.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import settings
from detectors import initialize_detector
from platform_services import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes the detector at startup.
    """
    logger.info("Starting AI-Generated Voice Detection API")
    logger.info(f"Detector backend: {settings.DETECTOR_BACKEND}")
    logger.info(f"Docs enabled: {settings.ENABLE_DOCS}")

    # Initialize detector at startup
    try:
        initialize_detector(settings.DETECTOR_BACKEND)
        logger.info("Detector initialized successfully")
    except NotImplementedError as e:
        logger.warning(f"Detector not fully implemented: {e}")
        # Fall back to qc_fallback if the selected detector is not implemented
        if settings.DETECTOR_BACKEND != "qc_fallback":
            logger.info("Falling back to qc_fallback detector")
            initialize_detector("qc_fallback")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        raise

    yield

    logger.info("Shutting down AI-Generated Voice Detection API")


# Conditional docs/openapi configuration
docs_url = "/docs" if settings.ENABLE_DOCS else None
openapi_url = "/openapi.json" if settings.ENABLE_DOCS else None
redoc_url = None  # Always disabled

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "API for detecting AI-generated speech vs human speech. "
        "Supports Tamil, English, Hindi, Malayalam, and Telugu."
    ),
    version="1.0.0",
    docs_url=docs_url,
    openapi_url=openapi_url,
    redoc_url=redoc_url,
    lifespan=lifespan,
)

# Include the API router without prefix (endpoint is /api/voice-detection)
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    # Use port 8001 for local development, 8080 for production
    port = 8001
    logger.info(f"Starting server on port {port}")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
