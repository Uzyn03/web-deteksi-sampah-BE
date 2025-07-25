import os
import aiofiles
from fastapi import UploadFile
from typing import Tuple, Optional
from app.config import settings
from app.utils.logger import get_logger
from app.utils.image_utils import (
    validate_image, load_image_from_bytes, 
    draw_detections, encode_image_to_base64, get_image_info
)
from app.core.exceptions import (
    InvalidImageException, FileSizeExceededException, 
    UnsupportedFileTypeException
)

logger = get_logger()

class ImageService:
    def __init__(self):
        self.upload_dir = settings.upload_dir
        self.max_file_size = settings.max_file_size
        self.allowed_extensions = settings.allowed_extensions
    
    async def validate_upload_file(self, file: UploadFile) -> None:
        """Validate uploaded file"""
        # Check file size
        if file.size > self.max_file_size:
            raise FileSizeExceededException(self.max_file_size)
        
        # Check file extension
        if file.filename:
            extension = file.filename.split('.')[-1].lower()
            if extension not in self.allowed_extensions:
                raise UnsupportedFileTypeException(self.allowed_extensions)
    
    async def process_upload_file(self, file: UploadFile) -> Tuple[bytes, dict]:
        """Process uploaded file and return image bytes and info"""
        try:
            # Validate file
            await self.validate_upload_file(file)
            
            # Read file content
            file_content = await file.read()
            
            # Validate image content
            if not validate_image(file_content):
                raise InvalidImageException("Invalid or corrupted image file")
            
            # Load image to get info
            image = load_image_from_bytes(file_content)
            if image is None:
                raise InvalidImageException("Failed to load image")
            
            # Get image info
            image_info = get_image_info(image)
            image_info.update({
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size": file.size
            })
            
            return file_content, image_info
            
        except Exception as e:
            logger.error(f"Failed to process upload file: {e}")
            if isinstance(e, (InvalidImageException, FileSizeExceededException, UnsupportedFileTypeException)):
                raise
            raise InvalidImageException(f"File processing error: {str(e)}")
    
    def create_annotated_image(self, image_bytes: bytes, detections: list) -> Optional[str]:
        """Create annotated image with detection results"""
        try:
            # Load image
            image = load_image_from_bytes(image_bytes)
            if image is None:
                return None
            
            # Draw detections
            annotated_image = draw_detections(image, detections, {})
            
            # Encode to base64
            return encode_image_to_base64(annotated_image)
            
        except Exception as e:
            logger.error(f"Failed to create annotated image: {e}")
            return None
    
    async def save_temp_file(self, file_content: bytes, filename: str) -> str:
        """Save file temporarily (optional for debugging)"""
        try:
            file_path = os.path.join(self.upload_dir, filename)
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            return file_path
        except Exception as e:
            logger.error(f"Failed to save temp file: {e}")
            raise
    
    def cleanup_temp_file(self, file_path: str) -> None:
        """Clean up temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Failed to cleanup temp file: {e}")

# Global image service instance
image_service = ImageService()