"""
Custom exceptions for the UI Validation API
"""
from typing import Any, Dict, Optional


class UIValidationException(Exception):
    """Base exception for UI validation errors"""
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ModelLoadError(UIValidationException):
    """Raised when AI models fail to load"""
    
    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Failed to load model: {model_name}",
            status_code=503,
            details=details
        )


class ImageProcessingError(UIValidationException):
    """Raised when image processing fails"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Image processing error: {message}",
            status_code=400,
            details=details
        )


class ValidationError(UIValidationException):
    """Raised when validation process fails"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Validation error: {message}",
            status_code=422,
            details=details
        )


class AIServiceError(UIValidationException):
    """Raised when AI services (OpenAI, etc.) fail"""
    
    def __init__(self, service: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"{service} service error: {message}",
            status_code=502,
            details=details
        )


class RateLimitError(UIValidationException):
    """Raised when rate limits are exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message,
            status_code=429
        )


class FileUploadError(UIValidationException):
    """Raised when file upload fails"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"File upload error: {message}",
            status_code=400,
            details=details
        )