import onnxruntime as ort
import numpy as np
from typing import List, Dict, Tuple, Optional
from app.config import settings
from app.utils.logger import get_logger
from app.core.exceptions import ModelNotLoadedException
import cv2

logger = get_logger()

class WasteDetectionService:
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_name: str = ""
        self.class_names = {
            0: "organic",      # Sampah organik
            1: "inorganic"     # Sampah anorganik
        }
        self.load_model()
    
    def load_model(self) -> bool:
        """Load ONNX model"""
        try:
            logger.info(f"Loading model from: {settings.model_path}")
            
            # Use GPU if available, else CPU
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
                logger.info("Using CUDAExecutionProvider (GPU)")
            else:
                logger.info("Using CPUExecutionProvider")
            
            self.session = ort.InferenceSession(settings.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"Model loaded successfully. Input: {self.input_name}, Output: {self.output_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.session = None
            return False
    
    def is_model_loaded(self) -> bool:
        return self.session is not None
    
    def detect(self, image: np.ndarray, confidence_threshold: float = None, iou_threshold: float = None) -> List[Dict]:
        """Run YOLO ONNX detection"""
        if not self.is_model_loaded():
            raise ModelNotLoadedException()
        
        conf_thresh = confidence_threshold or settings.confidence_threshold
        iou_thresh = iou_threshold or settings.iou_threshold
        
        try:
            model_input, scale = self._preprocess_image(image)
            
            outputs = self.session.run([self.output_name], {self.input_name: model_input})
            
            detections = self._postprocess_outputs(outputs[0], scale, conf_thresh, iou_thresh)
            logger.info(f"Detected {len(detections)} objects")
            return detections
        
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize and normalize image for YOLO ONNX"""
        height, width = image.shape[:2]
        target_size = settings.image_size
        
        scale = min(target_size / width, target_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized_image = cv2.resize(image, (new_width, new_height))
        padded_image = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2
        padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        
        normalized_image = padded_image.astype(np.float32) / 255.0
        model_input = np.transpose(normalized_image, (2, 0, 1))
        model_input = np.expand_dims(model_input, axis=0)
        
        return model_input, scale
    
    def _postprocess_outputs(self, outputs: np.ndarray, scale: float, conf_thresh: float, iou_thresh: float) -> List[Dict]:
        """Convert YOLO raw outputs [1, 6, 8400] to final boxes"""
        detections = []
        output = np.squeeze(outputs)  # Shape becomes [6, 8400]
        preds = output.T              # Transpose to [8400, 6]
        
        logger.debug(f"Shape of preds after transpose: {preds.shape}")
        
        # Assuming the 6 features are: x_center, y_center, w, h, class_prob_0, class_prob_1
        boxes_raw = preds[:, :4] # x_center, y_center, w, h
        class_probs = preds[:, 4:] # Probabilities for each class (e.g., [prob_organic, prob_inorganic])
        
        # Get the maximum probability (confidence) and the corresponding class ID
        scores = np.max(class_probs, axis=1)
        class_ids = np.argmax(class_probs, axis=1).astype(int)
        
        logger.debug(f"Raw scores (first 10): {scores[:10]}")
        logger.debug(f"Raw class_ids (first 10): {class_ids[:10]}")
        logger.debug(f"Number of initial proposals: {len(scores)}")
        
        # Filter by confidence
        valid_indices = np.where(scores > conf_thresh)[0]
        boxes_raw = boxes_raw[valid_indices]
        scores = scores[valid_indices]
        class_ids = class_ids[valid_indices]
        
        logger.debug(f"Number of proposals after confidence filter ({conf_thresh}): {len(valid_indices)}")
        
        if len(boxes_raw) == 0:
            return []
        
        # Convert to corner format (x1, y1, x2, y2)
        x_center, y_center, w, h = boxes_raw.T
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        boxes_corner = np.stack([x1, y1, x2, y2], axis=1)
        
        # Apply NMS
        # cv2.dnn.NMSBoxes expects boxes as list of lists, not numpy array
        indices = cv2.dnn.NMSBoxes(boxes_corner.tolist(), scores.tolist(), conf_thresh, iou_thresh)
        
        logger.debug(f"Number of detections after NMS ({iou_thresh}): {len(indices) if len(indices) > 0 else 0}")
        
        if len(indices) > 0:
            for idx in indices.flatten():
                # Scale back bounding box coordinates to original image size
                # The scale factor was applied during preprocessing, so we divide by it here
                x1_orig, y1_orig, x2_orig, y2_orig = boxes_corner[idx] / scale
                
                detections.append({
                    "class_id": int(class_ids[idx]),
                    "class_name": self.class_names.get(int(class_ids[idx]), "unknown"),
                    "confidence": float(scores[idx]),
                    "bbox": [float(x1_orig), float(y1_orig), float(x2_orig), float(y2_orig)]
                })
        
        return detections


# Global detection service instance
detection_service = WasteDetectionService()
