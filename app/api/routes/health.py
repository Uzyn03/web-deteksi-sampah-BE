from fastapi import APIRouter
from app.models.response import HealthResponse
from app.services.detection_service import detection_service
from app.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_status = detection_service.is_model_loaded()
    
    status = "healthy" if model_status else "degraded"
    message = "Service is running normally" if model_status else "Model not loaded"
    
    return HealthResponse(
        status=status,
        message=message,
        model_loaded=model_status,
        version=settings.app_version
    )

@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "status": "running"
    }