"""
Advanced element detection endpoints
"""
import uuid
from typing import Optional, List
from pathlib import Path
from PIL import Image

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import structlog

from ...core.config import settings
from ...core.exceptions import ValidationError, FileUploadError
from ...schemas.validation import UIElement
from ...services.element_detection_service import ElementDetectionService, get_element_detection_service
from ...services.debug_service import DebugService, get_debug_service
from ...services import get_omniparser_service, get_clip_service
from ...utils.file_handler import FileHandler, get_file_handler

router = APIRouter(prefix="/element-detection", tags=["element-detection"])
logger = structlog.get_logger(__name__)


@router.post("/detect", response_model=List[UIElement])
async def detect_specific_elements(
    element_type: str = Form(..., description="Type of element to detect (e.g., 'heart icon', 'submit button')"),
    image: UploadFile = File(..., description="Screenshot to analyze"),
    max_elements: Optional[int] = Form(default=10, description="Maximum elements to return"),
    enable_debug: bool = Form(default=False, description="Enable debug crop saving"),
    element_detection_service: ElementDetectionService = Depends(get_element_detection_service),
    omniparser_service = Depends(get_omniparser_service),
    clip_service = Depends(get_clip_service),
    debug_service: DebugService = Depends(get_debug_service),
    file_handler: FileHandler = Depends(get_file_handler)
):
    """
    Detect specific UI elements in a screenshot using advanced AI pipeline.
    
    This endpoint:
    1. Uses OmniParser to detect all UI elements
    2. Applies CLIP filtering to find elements relevant to the specified type
    3. Uses GPT to identify which crops actually contain the target element
    4. Returns detailed results with confidence scores and reasoning
    
    - **element_type**: What to look for (e.g., "heart icon", "submit button", "menu icon")
    - **image**: Screenshot to analyze
    - **max_elements**: Limit number of results (after CLIP filtering)
    - **enable_debug**: Save debug crops for inspection (development only)
    """
    session_id = str(uuid.uuid4())[:8]
    session_logger = logger.bind(
        endpoint="detect_specific_elements",
        element_type=element_type,
        session_id=session_id
    )
    session_logger.info("Received element detection request")
    
    try:
        # Validate and save uploaded image
        if not image.content_type.startswith('image/'):
            raise FileUploadError(f"Invalid file type: {image.content_type}")
        
        image_path = await file_handler.save_upload(image, f"detect_{session_id}")
        session_logger.info("Image uploaded", image_path=image_path)
        
        # Step 1: Parse all UI elements with OmniParser
        session_logger.info("Parsing UI elements with OmniParser")
        all_elements = await omniparser_service.analyze_screenshot(image_path)
        session_logger.info("OmniParser completed", elements_found=len(all_elements))
        
        # Step 2: Apply CLIP filtering for relevance
        session_logger.info("Applying CLIP filtering for relevance")
        filtered_elements = await clip_service.filter_elements_async(
            image_path=image_path,
            elements=all_elements,
            qa_prompt=f"find {element_type}",
            max_elements=max_elements or settings.clip_max_elements
        )
        session_logger.info("CLIP filtering completed", filtered_count=len(filtered_elements))
        
        # Step 3: Debug crop saving (if enabled)
        debug_dir = None
        if enable_debug or settings.enable_debug_crops:
            debug_dir = await debug_service.save_clip_filtered_crops(
                image_path=image_path,
                elements=filtered_elements,
                qa_prompt=f"find {element_type}",
                session_id=session_id
            )
        
        # Step 4: Use advanced element detection on the filtered crops
        crops = []
        for element in filtered_elements:
            try:
                # Load image and crop element
                full_image = Image.open(image_path).convert('RGB')
                bbox = element.bbox
                left, top = bbox.x, bbox.y
                right, bottom = left + bbox.width, top + bbox.height
                crop = full_image.crop((left, top, right, bottom))
                crops.append(crop)
            except Exception as e:
                session_logger.warning("Failed to crop element", 
                                     element_id=element.id,
                                     error=str(e))
        
        if crops:
            session_logger.info("Running advanced element detection on crops")
            detection_results = await element_detection_service.detect_elements_in_crops(
                crops=crops,
                element_type=element_type,
                session_id=session_id
            )
            
            # Update elements with detection results
            target_elements = []
            for i, result in enumerate(detection_results):
                if result.is_target_element and i < len(filtered_elements):
                    element = filtered_elements[i]
                    # Add detection confidence and reasoning to element
                    element.confidence = result.confidence
                    element.caption = f"{element_type}: {result.reasoning}"
                    target_elements.append(element)
            
            session_logger.info("Advanced detection completed",
                              target_elements_found=len(target_elements))
            
            return target_elements
        else:
            session_logger.warning("No valid crops extracted")
            return []
        
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
        # Cleanup uploaded file
        try:
            if 'image_path' in locals() and Path(image_path).exists():
                Path(image_path).unlink()
        except Exception as e:
            session_logger.warning("Failed to cleanup image file", error=str(e))


@router.get("/debug/{session_id}")
async def get_debug_crops(
    session_id: str,
    debug_service: DebugService = Depends(get_debug_service)
):
    """
    Get debug information for a specific session.
    Returns metadata about saved crops and analysis results.
    
    - **session_id**: Session identifier from element detection request
    """
    if not settings.enable_debug_crops:
        raise HTTPException(status_code=404, detail="Debug mode not enabled")
    
    debug_dir = Path(settings.debug_output_dir) / f"session_{session_id}"
    metadata_path = debug_dir / "metadata.json"
    
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Debug session not found")
    
    try:
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read debug data: {e}")


@router.get("/debug/{session_id}/crop/{crop_filename}")
async def get_debug_crop_image(
    session_id: str,
    crop_filename: str,
    debug_service: DebugService = Depends(get_debug_service)
):
    """
    Download a specific debug crop image.
    
    - **session_id**: Session identifier
    - **crop_filename**: Name of the crop file to download
    """
    if not settings.enable_debug_crops:
        raise HTTPException(status_code=404, detail="Debug mode not enabled")
    
    debug_dir = Path(settings.debug_output_dir) / f"session_{session_id}"
    crop_path = debug_dir / crop_filename
    
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Debug crop not found")
    
    return FileResponse(
        path=str(crop_path),
        media_type="image/png",
        filename=crop_filename
    )


@router.delete("/debug/cleanup")
async def cleanup_debug_files(
    max_age_hours: int = 48,
    debug_service: DebugService = Depends(get_debug_service)
):
    """
    Clean up old debug files and directories.
    
    - **max_age_hours**: Remove files older than this many hours (default: 48)
    """
    removed_count = await debug_service.cleanup_old_debug_files(max_age_hours)
    
    return {
        "message": f"Cleaned up {removed_count} debug files",
        "removed_count": removed_count,
        "max_age_hours": max_age_hours
    }