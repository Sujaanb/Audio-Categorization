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
    Initializes the AASIST detector at startup.
    """
    logger.info("Starting AI-Generated Voice Detection API")
    logger.info(f"Docs enabled: {settings.ENABLE_DOCS}")

    # Initialize AASIST detector at startup (fail fast on error)
    try:
        initialize_detector()
        logger.info("AASIST detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AASIST detector: {e}")
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
