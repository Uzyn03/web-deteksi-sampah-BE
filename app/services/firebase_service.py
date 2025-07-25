from typing import Optional, Dict, Any, List
import json
from datetime import datetime
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger()

# Uncomment and implement if using Firebase
"""
import firebase_admin
from firebase_admin import credentials, firestore, storage

class FirebaseService:
    def __init__(self):
        self.db = None
        self.bucket = None
        self.initialize_firebase()
    
    def initialize_firebase(self):
        try:
            if not firebase_admin._apps:
                if settings.firebase_credentials_path:
                    cred = credentials.Certificate(settings.firebase_credentials_path)
                    firebase_admin.initialize_app(cred, {
                        'storageBucket': settings.firebase_storage_bucket
                    })
                else:
                    # Use default credentials (for Cloud Run, etc.)
                    firebase_admin.initialize_app()
            
            self.db = firestore.client()
            self.bucket = storage.bucket()
            logger.info("Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
    
    def save_detection_result(self, detection_data: Dict[str, Any]) -> Optional[str]:
        try:
            doc_ref = self.db.collection('detections').document()
            detection_data['timestamp'] = datetime.now()
            detection_data['id'] = doc_ref.id
            
            doc_ref.set(detection_data)
            logger.info(f"Detection result saved with ID: {doc_ref.id}")
            return doc_ref.id
            
        except Exception as e:
            logger.error(f"Failed to save detection result: {e}")
            return None
    
    def get_detection_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            docs = (self.db.collection('detections')
                   .order_by('timestamp', direction=firestore.Query.DESCENDING)
                   .limit(limit)
                   .stream())
            
            results = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                results.append(data)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get detection history: {e}")
            return []
    
    def upload_image(self, image_bytes: bytes, filename: str) -> Optional[str]:
        try:
            blob = self.bucket.blob(f"images/{filename}")
            blob.upload_from_string(image_bytes, content_type='image/jpeg')
            blob.make_public()
            
            logger.info(f"Image uploaded: {filename}")
            return blob.public_url
            
        except Exception as e:
            logger.error(f"Failed to upload image: {e}")
            return None

# Global Firebase service instance (uncomment if using Firebase)
# firebase_service = FirebaseService()
"""

# Placeholder class if not using Firebase
class FirebaseService:
    def __init__(self):
        logger.info("Firebase service disabled")
    
    def save_detection_result(self, detection_data: Dict[str, Any]) -> Optional[str]:
        logger.info("Firebase save disabled - detection result not saved")
        return None
    
    def get_detection_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        logger.info("Firebase get disabled - returning empty history")
        return []
    
    def upload_image(self, image_bytes: bytes, filename: str) -> Optional[str]:
        logger.info("Firebase upload disabled - image not uploaded")
        return None

firebase_service = FirebaseService()