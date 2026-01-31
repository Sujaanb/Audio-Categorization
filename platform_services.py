from fastapi import APIRouter, HTTPException, status
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_router = APIRouter(tags=["AC API Services"])

"""
This is where the final endpoint will be written
"""