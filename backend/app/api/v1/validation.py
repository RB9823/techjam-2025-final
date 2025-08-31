"""
Validation API endpoints
"""
import os
import uuid
import asyncio
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import structlog

from ...core.config import settings
from ...core.exceptions import ValidationError, FileUploadError
from ...schemas.validation import (
    ValidationRequest,
    ValidationResponse,
    BatchValidationRequest,
    BatchValidationResponse,
    ValidationJobStatus,
    ValidationProgress
)
from ...services.validation_service import ValidationService, get_validation_service
from ...utils.file_handler import FileHandler, get_file_handler

router = APIRouter(prefix="/validate", tags=["validation"])
logger = structlog.get_logger(__name__)


@router.post("/", response_model=ValidationResponse)
async def validate_ui_change(
    qa_prompt: str = Form(..., description="QA prompt (e.g., 'Does button turn green when clicked?')"),
    before_image: UploadFile = File(..., description="Before screenshot"),
    after_image: UploadFile = File(..., description="After screenshot"),
    enable_streaming: bool = Form(default=True, description="Enable real-time progress"),
    max_gpt_calls: Optional[int] = Form(default=None, description="Limit GPT API calls"),
    detection_confidence: Optional[float] = Form(default=None, description="YOLO confidence threshold"),
    validation_service: ValidationService = Depends(get_validation_service),
    file_handler: FileHandler = Depends(get_file_handler)
):
    """
    Validate UI changes between before and after screenshots.
    
    This endpoint processes two images and a QA prompt to validate whether
    the expected UI change occurred. Uses AI pipeline with OmniParser, CLIP, and GPT.
    
    - **qa_prompt**: Question to validate (e.g., "Does button turn green when clicked?")
    - **before_image**: Screenshot before the action
    - **after_image**: Screenshot after the action
    - **enable_streaming**: Whether to provide real-time progress (for WebSocket)
    - **max_gpt_calls**: Optional limit on expensive GPT API calls
    - **detection_confidence**: Optional YOLO detection confidence threshold
    """
    session_logger = logger.bind(
        endpoint="validate_ui_change",
        qa_prompt=qa_prompt,
        request_id=str(uuid.uuid4())
    )
    session_logger.info("Received validation request")
    
    try:
        # Validate file uploads
        for upload_file in [before_image, after_image]:
            if not upload_file.content_type.startswith('image/'):
                raise FileUploadError(f"Invalid file type: {upload_file.content_type}")
            
            if upload_file.size > settings.max_file_size:
                raise FileUploadError(f"File too large: {upload_file.size} bytes")
        
        # Save uploaded files
        before_path = await file_handler.save_upload(before_image, "before")
        after_path = await file_handler.save_upload(after_image, "after")
        
        session_logger.info("Files uploaded successfully", 
                           before_path=before_path, 
                           after_path=after_path)
        
        # Create validation request
        request = ValidationRequest(
            qa_prompt=qa_prompt,
            enable_streaming=enable_streaming,
            max_gpt_calls=max_gpt_calls,
            detection_confidence=detection_confidence
        )
        
        # Process validation
        result = await validation_service.validate_ui_change(
            request=request,
            before_image_path=before_path,
            after_image_path=after_path
        )
        
        session_logger.info("Validation completed", 
                           job_id=result.job_id,
                           is_valid=result.is_valid,
                           processing_time=result.processing_time_seconds)
        
        return result
        
    except FileUploadError as e:
        session_logger.error("File upload error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        session_logger.error("Validation error", error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        session_logger.error("Unexpected error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup uploaded files
        try:
            if 'before_path' in locals() and os.path.exists(before_path):
                os.unlink(before_path)
            if 'after_path' in locals() and os.path.exists(after_path):
                os.unlink(after_path)
        except Exception as e:
            session_logger.warning("Failed to cleanup files", error=str(e))


@router.get("/{job_id}", response_model=ValidationResponse)
async def get_validation_status(
    job_id: str,
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Get the status and results of a validation job.
    
    - **job_id**: Unique identifier for the validation job
    """
    job_data = await validation_service.get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Return the response if available, otherwise return status info
    if "response" in job_data:
        return job_data["response"]
    
    return ValidationResponse(
        job_id=job_id,
        status=job_data["status"],
        created_at=job_data["start_time"]
    )


@router.delete("/{job_id}")
async def cancel_validation(
    job_id: str,
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Cancel a running validation job.
    
    - **job_id**: Unique identifier for the validation job to cancel
    """
    cancelled = await validation_service.cancel_job(job_id)
    
    if not cancelled:
        raise HTTPException(status_code=404, detail="Job not found or already completed")
    
    return {"message": "Job cancelled successfully", "job_id": job_id}


@router.post("/batch", response_model=BatchValidationResponse)
async def batch_validate_ui_changes(
    request: BatchValidationRequest,
    background_tasks: BackgroundTasks,
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Validate multiple UI changes in batch.
    
    This endpoint processes multiple validation requests concurrently.
    Use this for efficient processing of multiple test cases.
    
    - **items**: List of validation items with before/after images and QA prompts
    - **enable_streaming**: Whether to provide real-time progress updates
    - **max_concurrent**: Maximum number of concurrent validations (default: 3)
    """
    batch_id = str(uuid.uuid4())
    session_logger = logger.bind(endpoint="batch_validate", batch_id=batch_id)
    session_logger.info("Received batch validation request", items_count=len(request.items))
    
    try:
        # Convert items to validation requests and file paths
        # Note: For batch validation, we assume image URLs are accessible file paths
        # In a full implementation, you'd want file upload handling for batch as well
        validation_requests = []
        
        for item in request.items:
            if not item.before_image_url or not item.after_image_url:
                raise ValidationError(f"Missing image URLs for item {item.item_id}")
            
            validation_req = ValidationRequest(
                qa_prompt=item.qa_prompt,
                enable_streaming=request.enable_streaming,
                max_gpt_calls=request.max_gpt_calls,
                detection_confidence=request.detection_confidence
            )
            
            validation_requests.append((
                validation_req,
                item.before_image_url,
                item.after_image_url
            ))
        
        # Process batch validation
        results = await validation_service.batch_validate(
            requests=validation_requests,
            max_concurrent=request.max_concurrent
        )
        
        # Compile batch response
        completed_count = len([r for r in results if r.status == ValidationJobStatus.COMPLETED])
        failed_count = len([r for r in results if r.status == ValidationJobStatus.FAILED])
        
        batch_response = BatchValidationResponse(
            job_id=batch_id,
            status=ValidationJobStatus.COMPLETED,
            results=results,
            completed_count=completed_count,
            failed_count=failed_count,
            total_count=len(results)
        )
        
        session_logger.info("Batch validation completed",
                           total=len(results),
                           completed=completed_count,
                           failed=failed_count)
        
        return batch_response
        
    except ValidationError as e:
        session_logger.error("Batch validation error", error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        session_logger.error("Unexpected batch validation error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/jobs/cleanup")
async def cleanup_old_jobs(
    max_age_hours: int = 24,
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Clean up old completed validation jobs.
    
    - **max_age_hours**: Remove jobs older than this many hours (default: 24)
    """
    removed_count = await validation_service.cleanup_completed_jobs(max_age_hours)
    
    return {
        "message": f"Cleaned up {removed_count} old jobs",
        "removed_count": removed_count,
        "max_age_hours": max_age_hours
    }