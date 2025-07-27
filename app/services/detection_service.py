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
        # PERBAIKAN: Class mapping berdasarkan hasil evaluasi Google Colab yang benar:
        # Berdasarkan output evaluasi Anda:
        # - Sampah Anorganik (Class 0)
        # - Sampah Organik (Class 1)  
        # - Sampah-Anorganik (Class 2)
        self.class_names = {
            0: "Sampah Anorganik",    # Class 0 dari model
            1: "Sampah Organik",      # Class 1 dari model              
            2: "Sampah-Anorganik"     # Class 2 dari model (mungkin duplikat atau subkategori)
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
            
            # Log model input/output shapes untuk debugging
            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape
            logger.info(f"Model loaded successfully. Input: {self.input_name} {input_shape}, Output: {self.output_name} {output_shape}")
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
            model_input, scale, pad_info = self._preprocess_image(image)
            
            outputs = self.session.run([self.output_name], {self.input_name: model_input})
            
            detections = self._postprocess_outputs(outputs[0], scale, pad_info, conf_thresh, iou_thresh)
            logger.info(f"Detected {len(detections)} objects")
            return detections
        
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize and normalize image for YOLO ONNX - sesuai dengan YOLOv8 standard"""
        original_height, original_width = image.shape[:2]
        target_size = settings.image_size
        
        # Hitung scaling factor yang sama seperti YOLOv8
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize dengan INTER_LINEAR (default YOLOv8)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image dengan nilai 114 (YOLOv8 default)
        padded_image = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        
        # Calculate padding - center the image
        pad_x = (target_size - new_width) // 2
        pad_y = (target_size - new_height) // 2
        
        # Place resized image in center
        padded_image[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized_image
        
        # IMPORTANT: YOLOv8 expects RGB format, OpenCV loads as BGR
        # Convert BGR to RGB
        padded_image_rgb = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] seperti training
        normalized_image = padded_image_rgb.astype(np.float32) / 255.0
        
        # Transpose ke CHW format (Channel, Height, Width)
        model_input = np.transpose(normalized_image, (2, 0, 1))
        model_input = np.expand_dims(model_input, axis=0)  # Add batch dimension: [1, 3, 640, 640]
        
        logger.debug(f"Preprocessing: original({original_width}x{original_height}) -> resized({new_width}x{new_height}) -> padded({target_size}x{target_size})")
        logger.debug(f"Scale: {scale}, Padding: ({pad_x}, {pad_y})")
        
        return model_input, scale, (pad_x, pad_y)
    
    def _postprocess_outputs(self, outputs: np.ndarray, scale: float, pad_info: Tuple[int, int],
                           conf_thresh: float, iou_thresh: float) -> List[Dict]:
        """Convert YOLO raw outputs to final boxes - YOLOv8 format"""
        detections = []
        pad_x, pad_y = pad_info
        
        # Log shape untuk debugging
        logger.debug(f"Raw output shape: {outputs.shape}")
        
        # YOLOv8 output format: [1, 4+num_classes, num_anchors]
        # Untuk 3 kelas: [1, 7, 8400] -> [4 bbox coords + 3 class scores]
        output = np.squeeze(outputs)  # Remove batch dimension -> [7, 8400]
        
        if output.shape[0] == 7:  # [7, 8400] format untuk 3 kelas
            preds = output.T  # -> [8400, 7]
            boxes_raw = preds[:, :4]  # x_center, y_center, w, h
            class_probs = preds[:, 4:7]  # 3 class probabilities
        elif output.shape[0] == 6:  # [6, 8400] format untuk 2 kelas
            preds = output.T  # -> [8400, 6]
            boxes_raw = preds[:, :4]  # x_center, y_center, w, h
            class_probs = preds[:, 4:6]  # 2 class probabilities
        else:
            logger.error(f"Unexpected output shape: {output.shape}")
            return []
        
        logger.debug(f"Boxes shape: {boxes_raw.shape}, Class probs shape: {class_probs.shape}")
        
        # Get confidence scores dan class IDs
        scores = np.max(class_probs, axis=1)
        class_ids = np.argmax(class_probs, axis=1).astype(int)
        
        # DEBUGGING: Print statistik untuk diagnosa
        logger.info(f"Class probabilities stats:")
        for class_idx in range(class_probs.shape[1]):
            class_max = np.max(class_probs[:, class_idx])
            class_mean = np.mean(class_probs[:, class_idx])
            class_count = np.sum(class_ids == class_idx)
            logger.info(f"  Class {class_idx} ({self.class_names.get(class_idx, 'unknown')}): max={class_max:.3f}, mean={class_mean:.3f}, predicted_count={class_count}")
        
        logger.debug(f"Overall - Max score: {np.max(scores):.3f}, Min score: {np.min(scores):.3f}")
        logger.debug(f"Unique class IDs: {np.unique(class_ids)}")
        
        # Tampilkan top 10 predictions untuk debugging
        top_indices = np.argsort(scores)[-10:][::-1]  # Top 10 by confidence
        logger.info("Top 10 predictions:")
        for i, idx in enumerate(top_indices):
            logger.info(f"  {i+1}. Class {class_ids[idx]} ({self.class_names.get(class_ids[idx], 'unknown')}): {scores[idx]:.3f}")
            logger.info(f"      Raw probs: {class_probs[idx]}")
        
        # Filter by confidence threshold
        valid_indices = np.where(scores >= conf_thresh)[0]
        
        if len(valid_indices) == 0:
            logger.debug(f"No detections above confidence threshold {conf_thresh}")
            return []
        
        boxes_raw = boxes_raw[valid_indices]
        scores = scores[valid_indices]
        class_ids = class_ids[valid_indices]
        
        logger.debug(f"After confidence filter: {len(valid_indices)} detections")
        
        # Convert dari center format ke corner format dan remove padding
        x_center, y_center, w, h = boxes_raw.T
        
        # Remove padding offset
        x_center = x_center - pad_x
        y_center = y_center - pad_y
        
        # Convert to corner coordinates
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        
        # Scale back ke original image size
        x1 = x1 / scale
        y1 = y1 / scale
        x2 = x2 / scale
        y2 = y2 / scale
        
        boxes_corner = np.stack([x1, y1, x2, y2], axis=1)
        
        # Apply Non-Maximum Suppression
        if len(boxes_corner) > 0:
            # Convert ke format yang dibutuhkan cv2.dnn.NMSBoxes
            boxes_list = []
            for box in boxes_corner:
                # Convert [x1, y1, x2, y2] ke [x, y, w, h]
                x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                boxes_list.append([float(x), float(y), float(w), float(h)])
            
            indices = cv2.dnn.NMSBoxes(boxes_list, scores.tolist(), conf_thresh, iou_thresh)
            
            logger.debug(f"After NMS: {len(indices) if len(indices) > 0 else 0} detections")
            
            if len(indices) > 0:
                for idx in indices.flatten():
                    x1_final, y1_final, x2_final, y2_final = boxes_corner[idx]
                    
                    # Pastikan koordinat tidak negatif
                    x1_final = max(0, x1_final)
                    y1_final = max(0, y1_final)
                    
                    detection = {
                        "class_id": int(class_ids[idx]),
                        "class_name": self.class_names.get(int(class_ids[idx]), "unknown"),
                        "confidence": float(scores[idx]),
                        "bbox": [float(x1_final), float(y1_final), float(x2_final), float(y2_final)]
                    }
                    
                    # DEBUGGING: Log raw class probabilities untuk class ini
                    raw_probs = class_probs[valid_indices[idx]]
                    logger.info(f"Final detection - Raw class probabilities: {raw_probs}")
                    logger.info(f"Predicted class_id: {int(class_ids[idx])}, mapped to: {detection['class_name']}")
                    
                    detections.append(detection)
                    logger.debug(f"Detection: {detection}")
        
        return detections

# Global detection service instance
detection_service = WasteDetectionService()
