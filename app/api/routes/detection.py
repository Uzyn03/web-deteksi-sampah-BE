import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
from app.models.response import DetectionResponse, ErrorResponse
from app.models.detection import DetectionRequest, Detection, BoundingBox
from app.services.detection_service import detection_service
from app.services.image_service import image_service
from app.utils.logger import get_logger
from app.utils.image_utils import load_image_from_bytes

router = APIRouter()
logger = get_logger()

@router.post("/detect", response_model=DetectionResponse)
async def detect_waste(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = Form(None, ge=0.0, le=1.0),
    iou_threshold: Optional[float] = Form(None, ge=0.0, le=1.0),
    return_image: bool = Form(True, description="Return annotated image")
):
    """
    Detect waste objects in uploaded image
    
    - **file**: Image file (JPG, PNG, BMP, WebP)
    - **confidence_threshold**: Minimum confidence score (0.0-1.0)
    - **iou_threshold**: IoU threshold for NMS (0.0-1.0)
    - **return_image**: Whether to return annotated image
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing detection request for file: {file.filename}")
        
        # Process uploaded file
        image_bytes, image_info = await image_service.process_upload_file(file)
        
        # Load image for detection
        image = load_image_from_bytes(image_bytes)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to load image")
        
        # Run detection
        detections = detection_service.detect(
            image, 
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        
        # Convert detections to response format
        detection_objects = []
        for det in detections:
            bbox = BoundingBox(
                x1=det['bbox'][0],
                y1=det['bbox'][1], 
                x2=det['bbox'][2],
                y2=det['bbox'][3]
            )
            detection_obj = Detection(
                class_id=det['class_id'],
                class_name=det['class_name'],
                confidence=det['confidence'],
                bbox=bbox
            )
            detection_objects.append(detection_obj)
        
        # Create annotated image if requested
        annotated_image_b64 = None
        if return_image and detections:
            annotated_image_b64 = image_service.create_annotated_image(image_bytes, detections)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"Detection completed in {processing_time:.3f}s, found {len(detections)} objects")
        
        return DetectionResponse(
            success=True,
            message=f"Detection completed successfully. Found {len(detections)} objects.",
            detections=detection_objects,
            detection_count=len(detections),
            processing_time=processing_time,
            image_info=image_info,
            annotated_image=annotated_image_b64
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Detection failed after {processing_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_loaded": detection_service.is_model_loaded(),
        "model_path": settings.model_path,
        "image_size": settings.image_size,
        "confidence_threshold": settings.confidence_threshold,
        "iou_threshold": settings.iou_threshold,
        "class_names": detection_service.class_names
    }