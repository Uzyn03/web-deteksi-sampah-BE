import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import Tuple, Optional
from app.utils.logger import get_logger

logger = get_logger()

def validate_image(image_bytes: bytes) -> bool:
    """Validate if bytes represent a valid image"""
    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        return True
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False

def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Load image from bytes and convert to OpenCV format"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {e}")
        return None

def preprocess_image(image: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float]:
    """Preprocess image for YOLO model"""
    try:
        # Get original dimensions
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_size / width, target_size / height)
        
        # Resize image
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Create padded image
        padded_image = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2
        
        # Place resized image in center
        padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        
        # Normalize and convert to float32
        normalized_image = padded_image.astype(np.float32) / 255.0
        
        # Convert to CHW format and add batch dimension
        model_input = np.transpose(normalized_image, (2, 0, 1))
        model_input = np.expand_dims(model_input, axis=0)
        
        return model_input, scale
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise

def draw_detections(image: np.ndarray, detections: list, class_names: dict) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    try:
        annotated_image = image.copy()
        
        # Define colors for different classes
        colors = {
            'organic': (0, 255, 0),      # Green
            'anorganik': (0, 0, 255),    # Red
            'inorganic': (0, 0, 255),    # Red (alternative)
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name'].lower()
            confidence = detection['confidence']
            
            # Get color for this class
            color = colors.get(class_name, (255, 0, 0))  # Default blue
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            
            # Create label
            label = f"{detection['class_name']}: {confidence:.2f}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                annotated_image,
                (int(x1), int(y1) - text_height - 10),
                (int(x1) + text_width, int(y1)),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return annotated_image
    except Exception as e:
        logger.error(f"Failed to draw detections: {e}")
        return image

def encode_image_to_base64(image: np.ndarray, format: str = 'JPEG') -> str:
    """Encode OpenCV image to base64 string"""
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Save to BytesIO
        buffer = BytesIO()
        pil_image.save(buffer, format=format, quality=85)
        
        # Encode to base64
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/{format.lower()};base64,{base64_string}"
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}")
        raise

def get_image_info(image: np.ndarray) -> dict:
    """Get basic information about an image"""
    height, width, channels = image.shape
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "size_bytes": image.nbytes
    }