"""
Health check endpoints
"""
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends
import structlog

from ...core.config import settings
from ...schemas.validation import HealthCheck
from ...services.validation_service import ValidationService, get_validation_service

router = APIRouter(prefix="/health", tags=["health"])
logger = structlog.get_logger(__name__)


@router.get("/", response_model=HealthCheck)
async def health_check():
    """
    Basic health check endpoint.
    Returns the API status and version information.
    """
    return HealthCheck(
        status="healthy",
        version=settings.app_version,
        models_loaded={},
        dependencies={}
    )


@router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health_check(
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Detailed health check that verifies all AI services and dependencies.
    
    This endpoint checks:
    - OmniParser service (YOLO + Florence-2 models)
    - CLIP service (semantic filtering)
    - GPT service (OpenAI API)
    - Redis cache connectivity
    - File system access
    """
    try:
        health_status = await validation_service.get_health_status()
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/models")
async def model_status(
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Get the loading status of all AI models.
    
    Returns information about:
    - YOLO detection model
    - Florence-2 captioning model  
    - CLIP similarity model
    - GPT service connectivity
    """
    try:
        health_status = await validation_service.get_health_status()
        
        return {
            "models": health_status.get("services", {}),
            "timestamp": health_status.get("timestamp"),
            "overall_status": health_status.get("status")
        }
        
    except Exception as e:
        logger.error("Model status check failed", error=str(e))
        return {
            "status": "error",
            "error": str(e)
        }