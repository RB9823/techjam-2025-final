"""
Pydantic schemas for UI validation API
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x: int = Field(..., description="X coordinate of top-left corner")
    y: int = Field(..., description="Y coordinate of top-left corner") 
    width: int = Field(..., description="Width of bounding box")
    height: int = Field(..., description="Height of bounding box")


class UIElement(BaseModel):
    """Detected UI element"""
    id: str = Field(..., description="Unique element identifier")
    bbox: BoundingBox = Field(..., description="Element bounding box")
    caption: str = Field(..., description="AI-generated caption")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    detection_method: str = Field(..., description="Detection method used (yolo, gpt4v, florence2)")
    clip_similarity: Optional[float] = Field(None, description="CLIP similarity score to query")


class ChangeType(str, Enum):
    """Types of detected changes"""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified" 
    MOVED = "moved"
    STATE_CHANGED = "state_changed"


class DetectedChange(BaseModel):
    """Detected change between UI states"""
    element_id: str = Field(..., description="Element identifier")
    change_type: ChangeType = Field(..., description="Type of change detected")
    before_element: Optional[UIElement] = Field(None, description="Element state before")
    after_element: Optional[UIElement] = Field(None, description="Element state after")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Change detection confidence")
    details: str = Field(..., description="Human-readable change description")


class ExceptionType(str, Enum):
    """Types of UI exceptions"""
    EXPECTED = "expected"
    UNEXPECTED = "unexpected"


class ExceptionCategory(str, Enum):
    """Categories of UI exceptions"""
    LOADING = "loading"
    POPUP = "popup"
    CRASH = "crash"
    RENDERING_ERROR = "rendering_error"
    LAYOUT_ISSUE = "layout_issue"
    STATE_CHANGE = "state_change"


class DetectedException(BaseModel):
    """Detected UI exception or anomaly"""
    exception_type: ExceptionType = Field(..., description="Whether exception was expected")
    category: ExceptionCategory = Field(..., description="Category of exception")
    description: str = Field(..., description="Human-readable description")
    location: Optional[BoundingBox] = Field(None, description="Location if applicable")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")


class ValidationRequest(BaseModel):
    """Request for UI validation"""
    qa_prompt: str = Field(..., description="Question to validate (e.g., 'Does button turn green when clicked?')")
    before_image_url: Optional[str] = Field(None, description="URL/path to before image")
    after_image_url: Optional[str] = Field(None, description="URL/path to after image")
    
    # Processing options
    enable_streaming: bool = Field(default=True, description="Enable real-time progress streaming")
    max_gpt_calls: Optional[int] = Field(None, description="Limit GPT API calls")
    detection_confidence: Optional[float] = Field(None, description="YOLO detection confidence threshold")
    clip_max_elements: Optional[int] = Field(None, description="Max elements after CLIP filtering")
    
    @validator('qa_prompt')
    def validate_qa_prompt(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError('QA prompt must be at least 5 characters')
        return v.strip()


class ValidationJobStatus(str, Enum):
    """Status of validation job"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationProgress(BaseModel):
    """Real-time validation progress"""
    stage: str = Field(..., description="Current processing stage")
    progress_percent: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    message: str = Field(..., description="Progress message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional fields for intermediate stage data
    details: Optional[str] = Field(None, description="Additional stage details")
    before_elements: Optional[List[UIElement]] = Field(None, description="UI elements in before image")
    after_elements: Optional[List[UIElement]] = Field(None, description="UI elements in after image")
    filtered_elements: Optional[List[UIElement]] = Field(None, description="Filtered UI elements (before)")
    filtered_after_elements: Optional[List[UIElement]] = Field(None, description="Filtered UI elements (after)")
    detected_changes: Optional[List[DetectedChange]] = Field(None, description="Detected changes")
    clip_analysis: Optional[Dict[str, Any]] = Field(None, description="CLIP analysis results")
    elements_detected: Optional[int] = Field(None, description="Number of elements detected")
    changes_detected: Optional[int] = Field(None, description="Number of changes detected")


class ValidationResponse(BaseModel):
    """Response from UI validation"""
    job_id: str = Field(..., description="Unique job identifier")
    status: ValidationJobStatus = Field(..., description="Validation status")
    
    # Core validation result
    is_valid: Optional[bool] = Field(None, description="Whether validation passed")
    reasoning: Optional[str] = Field(None, description="Professional QA reasoning")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Validation confidence")
    
    # Detailed analysis
    detected_changes: List[DetectedChange] = Field(default=[], description="Detected UI changes")
    detected_exceptions: List[DetectedException] = Field(default=[], description="Detected exceptions")
    before_elements: List[UIElement] = Field(default=[], description="UI elements in before image")
    after_elements: List[UIElement] = Field(default=[], description="UI elements in after image")
    
    # Metadata
    processing_time_seconds: Optional[float] = Field(None, description="Total processing time")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    # Error details
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
    
    # Processing statistics
    stats: Optional[Dict[str, Any]] = Field(None, description="Processing statistics")


class BatchValidationItem(BaseModel):
    """Single item in batch validation"""
    item_id: str = Field(..., description="Unique identifier for this validation item")
    qa_prompt: str = Field(..., description="Question to validate")
    before_image_url: str = Field(..., description="URL/path to before image")
    after_image_url: str = Field(..., description="URL/path to after image")


class BatchValidationRequest(BaseModel):
    """Request for batch UI validation"""
    items: List[BatchValidationItem] = Field(..., description="List of validation items")
    enable_streaming: bool = Field(default=True, description="Enable progress streaming")
    max_concurrent: int = Field(default=3, description="Max concurrent validations")
    
    # Global processing options
    max_gpt_calls: Optional[int] = Field(None, description="Global limit for GPT calls")
    detection_confidence: Optional[float] = Field(None, description="YOLO confidence threshold")
    
    @validator('items')
    def validate_items(cls, v):
        if not v:
            raise ValueError('At least one validation item is required')
        if len(v) > 50:  # Reasonable limit
            raise ValueError('Maximum 50 items per batch')
        return v


class BatchValidationResponse(BaseModel):
    """Response from batch UI validation"""
    job_id: str = Field(..., description="Unique batch job identifier")
    status: ValidationJobStatus = Field(..., description="Overall batch status")
    
    # Results
    results: List[ValidationResponse] = Field(default=[], description="Individual validation results")
    completed_count: int = Field(default=0, description="Number of completed validations")
    failed_count: int = Field(default=0, description="Number of failed validations")
    total_count: int = Field(..., description="Total number of validations")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
    processing_time_seconds: Optional[float] = Field(None)


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="API version")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    dependencies: Dict[str, str] = Field(..., description="Dependency health status")


class StreamMessage(BaseModel):
    """WebSocket/SSE stream message"""
    job_id: str = Field(..., description="Job identifier")
    type: str = Field(..., description="Message type (progress, result, error)")
    data: Union[ValidationProgress, ValidationResponse, Dict[str, Any]] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)