from fastapi import HTTPException
from typing import Any, Dict, Optional

class WasteDetectionException(HTTPException):
    def __init__(
        self,
        status_code: int,
        message: str,
        error_code: str = "GENERIC_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.details = details or {}
        super().__init__(status_code=status_code, detail={
            "message": message,
            "error_code": error_code,
            "details": self.details
        })

class ModelNotLoadedException(WasteDetectionException):
    def __init__(self):
        super().__init__(
            status_code=503,
            message="Detection model is not loaded",
            error_code="MODEL_NOT_LOADED"
        )

class InvalidImageException(WasteDetectionException):
    def __init__(self, details: str = ""):
        super().__init__(
            status_code=400,
            message="Invalid image format or corrupted image",
            error_code="INVALID_IMAGE",
            details={"details": details}
        )

class FileSizeExceededException(WasteDetectionException):
    def __init__(self, max_size: int):
        super().__init__(
            status_code=413,
            message=f"File size exceeds maximum allowed size of {max_size} bytes",
            error_code="FILE_SIZE_EXCEEDED",
            details={"max_size": max_size}
        )

class UnsupportedFileTypeException(WasteDetectionException):
    def __init__(self, allowed_types: list):
        super().__init__(
            status_code=415,
            message="Unsupported file type",
            error_code="UNSUPPORTED_FILE_TYPE",
            details={"allowed_types": allowed_types}
        )