from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from .detection import Detection

class DetectionResponse(BaseModel):
    success: bool = Field(..., description="Request success status")
    message: str = Field(..., description="Response message")
    detections: List[Detection] = Field(..., description="List of detections")
    detection_count: int = Field(..., description="Total number of detections")
    processing_time: float = Field(..., description="Processing time in seconds")
    image_info: Dict[str, Any] = Field(..., description="Original image information")
    annotated_image: Optional[str] = Field(None, description="Base64 encoded annotated image")
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    status: str = Field(..., description="API status")
    message: str = Field(..., description="Health message")
    timestamp: datetime = Field(default_factory=datetime.now)
    model_loaded: bool = Field(..., description="Model loading status")
    version: str = Field(..., description="API version")

class ErrorResponse(BaseModel):
    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now)