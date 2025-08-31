"""
Main validation service that orchestrates the entire UI validation pipeline
Combines OmniParser, CLIP filtering, and GPT validation into a cohesive workflow
"""
import asyncio
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
from PIL import Image
import structlog

from ..core.config import settings
from ..core.exceptions import ValidationError, AIServiceError
from ..schemas.validation import (
    ValidationRequest,
    ValidationResponse,
    ValidationJobStatus,
    ValidationProgress,
    UIElement,
    DetectedChange,
    DetectedException,
    BoundingBox
)
from .omniparser_service import get_omniparser_service, OmniParserService
from .clip_service import get_clip_service, CLIPService  
from .gpt_service import get_gpt_service, GPTService
from .element_detection_service import get_element_detection_service, ElementDetectionService

logger = structlog.get_logger(__name__)


class ValidationService:
    """
    Main validation service that orchestrates the complete UI validation pipeline.
    
    Pipeline stages:
    1. Image preprocessing and validation
    2. OmniParser UI element detection
    3. CLIP-based semantic filtering  
    4. GPT-powered change analysis and validation
    5. Result compilation and reporting
    """
    
    def __init__(
        self,
        omniparser_service: Optional[OmniParserService] = None,
        clip_service: Optional[CLIPService] = None,
        gpt_service: Optional[GPTService] = None,
        element_detection_service: Optional[ElementDetectionService] = None
    ):
        self.omniparser_service = omniparser_service
        self.clip_service = clip_service
        self.gpt_service = gpt_service
        self.element_detection_service = element_detection_service
        self.logger = logger.bind(service="validation")
        
        # Job tracking
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
        self._job_lock = asyncio.Lock()
        
        # Element detection data storage for GPT Vision validation
        self._current_element_data: Optional[Dict[str, Any]] = None
    
    async def initialize(self) -> None:
        """Initialize all dependent services"""
        try:
            self.logger.info("Initializing validation service")
            
            # Initialize services if not provided
            if not self.omniparser_service:
                self.omniparser_service = get_omniparser_service()
            if not self.clip_service:
                self.clip_service = await get_clip_service()
            if not self.gpt_service:
                self.gpt_service = await get_gpt_service()
            if not self.element_detection_service:
                self.element_detection_service = await get_element_detection_service()
            
            # Ensure all services are initialized
            await self.omniparser_service.initialize_models()
            await self.clip_service.initialize()
            await self.gpt_service.initialize()
            await self.element_detection_service.initialize()
            
            self.logger.info("Validation service initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize validation service", error=str(e))
            raise ValidationError(f"Service initialization failed: {e}")
    
    async def validate_ui_change(
        self,
        request: ValidationRequest,
        before_image_path: str,
        after_image_path: str,
        progress_callback: Optional[Callable[[ValidationProgress], None]] = None
    ) -> ValidationResponse:
        """
        Main validation workflow that processes UI change validation request.
        
        Args:
            request: Validation request with QA prompt and options
            before_image_path: Path to before screenshot
            after_image_path: Path to after screenshot  
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete validation response with reasoning and detected changes
        """
        job_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Track job
        async with self._job_lock:
            self._active_jobs[job_id] = {
                "status": ValidationJobStatus.PROCESSING,
                "start_time": start_time,
                "request": request
            }
        
        session_logger = self.logger.bind(job_id=job_id, qa_prompt=request.qa_prompt)
        session_logger.info("Starting UI validation", 
                           before_image=before_image_path, 
                           after_image=after_image_path)
        
        try:
            # Helper function to report progress
            async def report_progress(stage: str, progress: float, message: str):
                if progress_callback:
                    progress_update = ValidationProgress(
                        stage=stage,
                        progress_percent=progress,
                        message=message
                    )
                    progress_callback(progress_update)
                session_logger.info(f"Progress: {stage}", progress=progress, message=message)
            
            await report_progress("initialization", 0, "Starting validation pipeline")
            
            # Stage 1: Parse QA prompt to extract region of interest and expected change
            await report_progress("parsing_prompt", 2, "Parsing QA prompt")
            prompt_parsed = await self.gpt_service.parse_qa_prompt(request.qa_prompt)
            region_of_interest = prompt_parsed.get("region_of_interest")
            expected_change = prompt_parsed.get("expected_change")
            
            session_logger.info("QA prompt parsed", 
                              region_of_interest=region_of_interest,
                              expected_change=expected_change)
            
            # Stage 2: Validate images
            await report_progress("validation", 5, "Validating input images")
            await self._validate_images(before_image_path, after_image_path)
            
            # Stage 3: Parse UI elements
            await report_progress("parsing", 10, "Detecting UI elements with OmniParser")
            before_elements, after_elements = await self._parse_ui_elements(
                before_image_path, after_image_path, session_logger, region_of_interest
            )
            
            # Stage 4: CLIP filtering (if enabled) - now uses region of interest
            filtered_before, filtered_after = before_elements, after_elements
            if settings.clip_max_elements and len(before_elements) > settings.clip_max_elements:
                await report_progress("filtering", 40, "Filtering relevant elements with CLIP")
                filtered_before, filtered_after = await self._filter_elements_with_clip(
                    region_of_interest or request.qa_prompt,  # Use region or fallback to full prompt
                    before_image_path, before_elements,
                    after_image_path, after_elements,
                    session_logger
                )
            
            # Stage 4: Change detection
            await report_progress("analysis", 60, "Analyzing UI changes")
            detected_changes = await self._detect_changes(
                filtered_before, filtered_after, session_logger
            )
            
            # Stage 5: Exception detection
            await report_progress("exceptions", 70, "Detecting UI exceptions")
            detected_exceptions = await self._detect_exceptions(
                filtered_before, filtered_after, session_logger
            )
            
            # Stage 6: GPT validation
            await report_progress("validation", 80, "Generating professional QA validation")
            validation_result = await self._validate_with_gpt(
                request.qa_prompt,
                detected_changes,
                detected_exceptions, 
                filtered_before,
                filtered_after,
                session_logger,
                expected_change
            )
            
            # Stage 7: Save final debug images
            await report_progress("saving_debug", 90, "Saving debug images with final elements")
            
            # Use element data if available, otherwise use filtered elements
            if self._current_element_data:
                # Save the actual element crops that were used in validation
                debug_paths = await self._save_element_debug_images(
                    self._current_element_data["before_element"],
                    self._current_element_data["after_element"],
                    job_id, session_logger
                )
            else:
                # Fallback to saving filtered elements (top 1 from each)
                final_before = filtered_before[:1] if filtered_before else []
                final_after = filtered_after[:1] if filtered_after else []
                
                debug_paths = await self._save_final_debug_images(
                    before_image_path, after_image_path,
                    final_before, final_after,
                    job_id, region_of_interest, session_logger
                )
            
            # Stage 8: Compile final response
            await report_progress("compilation", 95, "Compiling final validation report")
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            response = ValidationResponse(
                job_id=job_id,
                status=ValidationJobStatus.COMPLETED,
                is_valid=validation_result.get("is_valid"),
                reasoning=validation_result.get("reasoning"),
                confidence=validation_result.get("confidence"),
                detected_changes=detected_changes,
                detected_exceptions=detected_exceptions,
                before_elements=before_elements,
                after_elements=after_elements,
                processing_time_seconds=processing_time,
                completed_at=datetime.utcnow(),
                stats={
                    "before_elements_count": len(before_elements),
                    "after_elements_count": len(after_elements),
                    "filtered_before_count": len(filtered_before),
                    "filtered_after_count": len(filtered_after),
                    "changes_detected": len(detected_changes),
                    "exceptions_detected": len(detected_exceptions),
                    "debug_images": debug_paths
                }
            )
            
            await report_progress("completed", 100, "Validation completed successfully")
            
            # Update job status
            async with self._job_lock:
                if job_id in self._active_jobs:
                    self._active_jobs[job_id]["status"] = ValidationJobStatus.COMPLETED
                    self._active_jobs[job_id]["response"] = response
            
            session_logger.info("Validation completed successfully", 
                               processing_time=processing_time,
                               is_valid=response.is_valid)
            
            return response
            
        except Exception as e:
            session_logger.error("Validation failed", error=str(e), exc_info=True)
            
            # Create error response
            error_response = ValidationResponse(
                job_id=job_id,
                status=ValidationJobStatus.FAILED,
                error_message=str(e),
                error_details={"exception_type": type(e).__name__},
                processing_time_seconds=(datetime.utcnow() - start_time).total_seconds()
            )
            
            # Update job status
            async with self._job_lock:
                if job_id in self._active_jobs:
                    self._active_jobs[job_id]["status"] = ValidationJobStatus.FAILED
                    self._active_jobs[job_id]["response"] = error_response
            
            if progress_callback:
                progress_callback(ValidationProgress(
                    stage="error",
                    progress_percent=0,
                    message=f"Validation failed: {str(e)}"
                ))
            
            return error_response
    
    async def _validate_images(self, before_path: str, after_path: str) -> None:
        """Validate that image files exist and are readable"""
        for path in [before_path, after_path]:
            if not Path(path).exists():
                raise ValidationError(f"Image file not found: {path}")
            
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as e:
                raise ValidationError(f"Invalid image file {path}: {e}")
    
    async def _parse_ui_elements(
        self, 
        before_path: str, 
        after_path: str,
        session_logger,
        region_of_interest: Optional[str] = None
    ) -> tuple[List[UIElement], List[UIElement]]:
        """Parse UI elements from both images using OmniParser and optional ElementDetectionService"""
        session_logger.info("Parsing UI elements", 
                           stage="omniparser",
                           region_of_interest=region_of_interest)
        
        try:
            # Parse both images concurrently
            before_task = self.omniparser_service.analyze_screenshot(before_path)
            after_task = self.omniparser_service.analyze_screenshot(after_path)
            
            before_result, after_result = await asyncio.gather(before_task, after_task)
            before_elements, before_metadata = before_result
            after_elements, after_metadata = after_result
            
            session_logger.info("UI parsing completed",
                              before_count=len(before_elements),
                              after_count=len(after_elements))
            
            # If we have a region of interest, use GPT Vision to find specific elements
            if region_of_interest:
                session_logger.info("Using GPT Vision for element detection",
                                  region_of_interest=region_of_interest)
                
                try:
                    # Apply GPT Vision element detection to CLIP-filtered elements
                    best_before_element, best_after_element = await self._detect_elements_with_gpt_vision(
                        before_path, before_elements, after_path, after_elements,
                        region_of_interest, session_logger
                    )
                    
                    # If elements were found in both images, convert back to UIElement format for consistency
                    if best_before_element and best_after_element:
                        session_logger.info(f"{region_of_interest} elements found in both images",
                                          before_element_confidence=best_before_element.get("confidence", 0.0),
                                          after_element_confidence=best_after_element.get("confidence", 0.0))
                        
                        # Store the element data in the ValidationService for later use in GPT validation
                        self._current_element_data = {
                            "before_element": best_before_element,
                            "after_element": best_after_element,
                            "element_type": region_of_interest
                        }
                        
                        # Return the original elements associated with the detected elements for compatibility
                        before_ui_element = [best_before_element["original_element"]]
                        after_ui_element = [best_after_element["original_element"]]
                        return before_ui_element, after_ui_element
                    else:
                        session_logger.info(f"{region_of_interest} elements not found in both images, using CLIP filtering")
                        
                except Exception as e:
                    session_logger.warning("GPT Vision element detection failed, falling back to CLIP filtering",
                                         error=str(e))
            
            return before_elements, after_elements
            
        except Exception as e:
            raise ValidationError(f"UI parsing failed: {e}")
    
    async def _detect_elements_with_gpt_vision(
        self,
        before_path: str, before_elements: List[UIElement],
        after_path: str, after_elements: List[UIElement],
        element_type: str,
        session_logger
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Use GPT Vision to find and select the best UI elements from CLIP-filtered elements"""
        
        from PIL import Image
        
        # Convert UIElements to PIL Image crops with metadata
        def element_to_crop_data(element: UIElement, image_path: str, crop_id: int) -> Dict[str, Any]:
            # Load the full image
            full_image = Image.open(image_path).convert('RGB')
            
            # Get bounding box coordinates
            bbox = element.bbox
            x1, y1 = bbox.x, bbox.y
            x2, y2 = x1 + bbox.width, y1 + bbox.height
            
            # Clamp coordinates to image bounds
            img_width, img_height = full_image.size
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(x1, min(x2, img_width))
            y2 = max(y1, min(y2, img_height))
            
            # Skip if invalid bounding box
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Crop the element
            cropped = full_image.crop((x1, y1, x2, y2))
            
            return {
                "crop_id": crop_id,
                "crop_image": cropped,
                "original_element": element,
                "bbox": [x1, y1, x2, y2]
            }
        
        # Create crop data for GPT Vision processing
        before_crop_data = []
        for i, element in enumerate(before_elements):
            try:
                crop_data = element_to_crop_data(element, before_path, i)
                if crop_data:
                    before_crop_data.append(crop_data)
            except Exception as e:
                session_logger.warning(f"Failed to crop before element {element.id}", error=str(e))
        
        after_crop_data = []
        for i, element in enumerate(after_elements):
            try:
                crop_data = element_to_crop_data(element, after_path, i)
                if crop_data:
                    after_crop_data.append(crop_data)
            except Exception as e:
                session_logger.warning(f"Failed to crop after element {element.id}", error=str(e))
        
        if not before_crop_data and not after_crop_data:
            session_logger.warning("No valid crops created for element detection")
            return None, None
        
        # Parallel GPT Vision element detection on all crops
        session_id = f"element_detection_{int(time.time() * 1000) % 100000}"
        
        async def detect_elements_in_image(crop_data_list: List[Dict[str, Any]], image_type: str) -> List[Dict[str, Any]]:
            """Detect elements in all crops from one image using parallel GPT calls"""
            if not crop_data_list:
                return []
                
            session_logger.info(f"Starting parallel {element_type} detection for {len(crop_data_list)} {image_type} crops")
            
            # Create tasks for parallel execution
            tasks = []
            for crop_data in crop_data_list:
                task = self.gpt_service.detect_element_in_crop(
                    crop_data["crop_image"], 
                    crop_data["crop_id"], 
                    element_type,
                    f"{session_id}_{image_type}"
                )
                tasks.append(task)
            
            # Execute all tasks in parallel
            import time
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time
            
            # Process results and combine with crop data
            element_candidates = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    session_logger.error(f"{element_type} detection failed for {image_type} crop {i}: {result}")
                    continue
                    
                if result.get("is_element", False):
                    # Combine detection result with crop data
                    element_candidate = {
                        **crop_data_list[i],
                        **result
                    }
                    element_candidates.append(element_candidate)
            
            session_logger.info(f"Parallel {image_type} detection completed in {duration:.2f}s - {len(element_candidates)} {element_type} elements found")
            return element_candidates
        
        # Process both images in parallel
        before_task = detect_elements_in_image(before_crop_data, "before")
        after_task = detect_elements_in_image(after_crop_data, "after")
        
        before_elements, after_elements = await asyncio.gather(before_task, after_task)
        
        # Select best element from each image (if multiple found)
        best_before_element = None
        best_after_element = None
        
        if before_elements:
            if len(before_elements) == 1:
                best_before_element = before_elements[0]
                session_logger.info(f"Single {element_type} found in before image")
            else:
                # Multiple elements - use GPT to select the best one
                session_logger.info(f"Multiple {element_type} elements found in before image ({len(before_elements)}), selecting best")
                best_before_element = await self.gpt_service.select_best_element(
                    before_elements, element_type, f"{session_id}_before_selection"
                )
        
        if after_elements:
            if len(after_elements) == 1:
                best_after_element = after_elements[0]
                session_logger.info(f"Single {element_type} found in after image")
            else:
                # Multiple elements - use GPT to select the best one
                session_logger.info(f"Multiple {element_type} elements found in after image ({len(after_elements)}), selecting best")
                best_after_element = await self.gpt_service.select_best_element(
                    after_elements, element_type, f"{session_id}_after_selection"
                )
        
        session_logger.info("GPT Vision element detection completed",
                          element_type=element_type,
                          before_found=best_before_element is not None,
                          after_found=best_after_element is not None)
        
        return best_before_element, best_after_element
    
    async def _filter_elements_with_clip(
        self,
        qa_prompt: str,
        before_path: str, before_elements: List[UIElement],
        after_path: str, after_elements: List[UIElement],
        session_logger
    ) -> tuple[List[UIElement], List[UIElement]]:
        """Filter elements using CLIP semantic similarity"""
        session_logger.info("Filtering elements with CLIP", 
                           qa_prompt=qa_prompt,
                           before_count=len(before_elements),
                           after_count=len(after_elements))
        
        try:
            # Apply CLIP filtering to both image sets concurrently
            before_task = self.clip_service.filter_elements_async(
                image_path=before_path,
                elements=before_elements,
                qa_prompt=qa_prompt,
                max_elements=settings.clip_max_elements
            )
            
            after_task = self.clip_service.filter_elements_async(
                image_path=after_path,
                elements=after_elements,
                qa_prompt=qa_prompt,
                max_elements=settings.clip_max_elements
            )
            
            filtered_before, filtered_after = await asyncio.gather(before_task, after_task)
            
            session_logger.info("CLIP filtering completed",
                              filtered_before_count=len(filtered_before),
                              filtered_after_count=len(filtered_after))
            
            return filtered_before, filtered_after
            
        except Exception as e:
            session_logger.warning("CLIP filtering failed, using all elements", error=str(e))
            return before_elements, after_elements
    
    async def _detect_changes(
        self,
        before_elements: List[UIElement],
        after_elements: List[UIElement],
        session_logger
    ) -> List[DetectedChange]:
        """Detect changes between before and after UI states"""
        session_logger.info("Detecting UI changes")
        
        # Simple change detection logic
        # TODO: Implement more sophisticated change detection
        changes = []
        
        # For now, create a summary change if elements differ
        if len(before_elements) != len(after_elements):
            changes.append(DetectedChange(
                element_id="layout_change",
                change_type="modified",
                confidence=0.8,
                details=f"Element count changed from {len(before_elements)} to {len(after_elements)}"
            ))
        
        return changes
    
    async def _detect_exceptions(
        self,
        before_elements: List[UIElement],
        after_elements: List[UIElement], 
        session_logger
    ) -> List[DetectedException]:
        """Detect UI exceptions and anomalies"""
        session_logger.info("Detecting UI exceptions")
        
        # Simple exception detection
        # TODO: Implement more sophisticated exception detection
        exceptions = []
        
        return exceptions
    
    async def _validate_with_gpt(
        self,
        qa_prompt: str,
        detected_changes: List[DetectedChange],
        detected_exceptions: List[DetectedException],
        before_elements: List[UIElement],
        after_elements: List[UIElement],
        session_logger,
        expected_change: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate changes using GPT professional reasoning (Vision-based if elements detected)"""
        session_logger.info("Validating with GPT", qa_prompt=qa_prompt)
        
        try:
            # If we have element data from GPT Vision detection, use Vision-based validation
            if self._current_element_data and expected_change:
                session_logger.info("Using GPT Vision validation for element comparison")
                
                session_id = f"element_validation_{int(time.time() * 1000) % 100000}"
                vision_result = await self.gpt_service.validate_element_change(
                    self._current_element_data["before_element"],
                    self._current_element_data["after_element"],
                    expected_change,
                    session_id
                )
                
                # Convert vision result to standard format
                result = {
                    "is_valid": vision_result.get("is_valid", False),
                    "reasoning": vision_result.get("reasoning", "No reasoning provided"),
                    "confidence": vision_result.get("confidence", 0.0),
                    "validation_type": "gpt_vision",
                    "element_comparison_details": {
                        "visual_changes": vision_result.get("visual_changes", []),
                        "same_ui_element": vision_result.get("same_ui_element", True),
                        "change_category": vision_result.get("change_category", "unknown"),
                        "before_element_confidence": vision_result.get("before_confidence", 0.0),
                        "after_element_confidence": vision_result.get("after_confidence", 0.0)
                    }
                }
                
                session_logger.info("GPT Vision validation completed", 
                                  is_valid=result.get("is_valid"),
                                  confidence=result.get("confidence"),
                                  change_category=vision_result.get("change_category"))
                
                return result
            
            # Fallback to traditional text-based validation
            session_logger.info("Using traditional text-based GPT validation")
            
            # Helper function to convert elements to dict format
            def element_to_dict(elem):
                if hasattr(elem, 'dict'):
                    return elem.dict()
                elif isinstance(elem, dict):
                    return elem
                elif isinstance(elem, list):
                    # Handle nested list structure
                    return {"raw_data": elem}
                else:
                    return {"raw_data": str(elem)}
            
            # Prepare analysis summary for GPT
            analysis_summary = {
                "qa_prompt": qa_prompt,
                "expected_change": expected_change or qa_prompt,  # Use parsed expected change or fallback to full prompt
                "before_elements": [element_to_dict(elem) for elem in before_elements],
                "after_elements": [element_to_dict(elem) for elem in after_elements],
                "detected_changes": [element_to_dict(change) for change in detected_changes],
                "detected_exceptions": [element_to_dict(exc) for exc in detected_exceptions]
            }
            
            # Get traditional GPT validation
            result = await self.gpt_service.validate_ui_change_async(analysis_summary)
            result["validation_type"] = "gpt_text"
            
            session_logger.info("Traditional GPT validation completed", 
                              is_valid=result.get("is_valid"),
                              confidence=result.get("confidence"))
            
            return result
            
        except Exception as e:
            session_logger.error("GPT validation failed", error=str(e))
            raise ValidationError(f"GPT validation failed: {e}")
        
        finally:
            # Clear element data after validation
            self._current_element_data = None
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a validation job"""
        async with self._job_lock:
            return self._active_jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running validation job"""
        async with self._job_lock:
            if job_id in self._active_jobs:
                self._active_jobs[job_id]["status"] = ValidationJobStatus.CANCELLED
                return True
            return False
    
    async def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up old completed jobs"""
        cutoff = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        removed_count = 0
        
        async with self._job_lock:
            to_remove = []
            for job_id, job_data in self._active_jobs.items():
                if (job_data["start_time"].timestamp() < cutoff and 
                    job_data["status"] in [ValidationJobStatus.COMPLETED, ValidationJobStatus.FAILED]):
                    to_remove.append(job_id)
            
            for job_id in to_remove:
                del self._active_jobs[job_id]
                removed_count += 1
        
        if removed_count > 0:
            self.logger.info("Cleaned up old jobs", removed_count=removed_count)
        
        return removed_count
    
    async def batch_validate(
        self,
        requests: List[tuple[ValidationRequest, str, str]],  # (request, before_path, after_path)
        progress_callback: Optional[Callable[[str, ValidationProgress], None]] = None,
        max_concurrent: int = 3
    ) -> List[ValidationResponse]:
        """
        Process multiple validation requests concurrently.
        
        Args:
            requests: List of (request, before_path, after_path) tuples
            progress_callback: Optional callback for progress updates (job_id, progress)
            max_concurrent: Maximum concurrent validations
            
        Returns:
            List of validation responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(request_data):
            async with semaphore:
                request, before_path, after_path = request_data
                callback = None
                if progress_callback:
                    callback = lambda p: progress_callback(f"batch_{uuid.uuid4()}", p)
                
                return await self.validate_ui_change(request, before_path, after_path, callback)
        
        self.logger.info("Starting batch validation", 
                        batch_size=len(requests), 
                        max_concurrent=max_concurrent)
        
        results = await asyncio.gather(
            *[process_single(req) for req in requests],
            return_exceptions=True
        )
        
        # Convert exceptions to error responses
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_response = ValidationResponse(
                    job_id=str(uuid.uuid4()),
                    status=ValidationJobStatus.FAILED,
                    error_message=str(result),
                    error_details={"exception_type": type(result).__name__}
                )
                final_results.append(error_response)
            else:
                final_results.append(result)
        
        self.logger.info("Batch validation completed", 
                        total=len(final_results),
                        succeeded=len([r for r in final_results if r.status == ValidationJobStatus.COMPLETED]),
                        failed=len([r for r in final_results if r.status == ValidationJobStatus.FAILED]))
        
        return final_results
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the validation service"""
        try:
            # Check all dependent services
            omniparser_health = await self.omniparser_service.get_health_status()
            clip_health = await self.clip_service.get_health_status()
            gpt_health = await self.gpt_service.get_health_status()
            
            # Overall health status
            all_healthy = all([
                omniparser_health.get("status") == "healthy",
                clip_health.get("status") == "healthy", 
                gpt_health.get("status") == "healthy"
            ])
            
            async with self._job_lock:
                active_jobs_count = len(self._active_jobs)
                processing_jobs = len([j for j in self._active_jobs.values() 
                                     if j["status"] == ValidationJobStatus.PROCESSING])
            
            return {
                "status": "healthy" if all_healthy else "degraded",
                "timestamp": datetime.utcnow().isoformat(),
                "services": {
                    "omniparser": omniparser_health,
                    "clip": clip_health,
                    "gpt": gpt_health
                },
                "active_jobs": active_jobs_count,
                "processing_jobs": processing_jobs
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _save_final_debug_images(
        self,
        before_image_path: str,
        after_image_path: str,
        before_elements: List[UIElement],
        after_elements: List[UIElement],
        job_id: str,
        region_of_interest: Optional[str],
        session_logger
    ) -> Dict[str, str]:
        """Save debug images showing final elements used in validation"""
        debug_paths = {}
        
        try:
            # Create debug directory if it doesn't exist
            debug_dir = Path(settings.debug_output_dir) / "final_comparison" / job_id
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Import PIL for image processing
            from PIL import Image, ImageDraw, ImageFont
            
            # Load original images
            before_img = Image.open(before_image_path).convert('RGB')
            after_img = Image.open(after_image_path).convert('RGB')
            
            # Create copies for drawing bounding boxes
            before_debug = before_img.copy()
            after_debug = after_img.copy()
            
            # Draw bounding boxes on full images
            before_draw = ImageDraw.Draw(before_debug)
            after_draw = ImageDraw.Draw(after_debug)
            
            # Colors for bounding boxes
            box_color = "red"
            box_width = 3
            
            cropped_images = []
            
            # Process before elements
            for i, element in enumerate(before_elements):
                bbox = element.bbox
                x1, y1 = bbox.x, bbox.y
                x2, y2 = x1 + bbox.width, y1 + bbox.height
                
                # Clamp coordinates to image bounds
                img_width, img_height = before_img.size
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(x1, min(x2, img_width))
                y2 = max(y1, min(y2, img_height))
                
                # Skip if invalid bounding box
                if x2 <= x1 or y2 <= y1:
                    session_logger.warning(f"Skipping invalid before element {i+1}", bbox=[x1, y1, x2, y2])
                    continue
                
                # Draw bounding box on full image
                before_draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)
                before_draw.text((x1, max(0, y1-20)), f"Element {i+1}", fill=box_color)
                
                # Crop the element from original image
                cropped = before_img.crop((x1, y1, x2, y2))
                crop_path = debug_dir / f"before_element_{i+1}_crop.png"
                cropped.save(crop_path)
                cropped_images.append(str(crop_path))
                
            # Process after elements
            for i, element in enumerate(after_elements):
                bbox = element.bbox
                x1, y1 = bbox.x, bbox.y
                x2, y2 = x1 + bbox.width, y1 + bbox.height
                
                # Clamp coordinates to image bounds
                img_width, img_height = after_img.size
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(x1, min(x2, img_width))
                y2 = max(y1, min(y2, img_height))
                
                # Skip if invalid bounding box
                if x2 <= x1 or y2 <= y1:
                    session_logger.warning(f"Skipping invalid after element {i+1}", bbox=[x1, y1, x2, y2])
                    continue
                
                # Draw bounding box on full image
                after_draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)
                after_draw.text((x1, max(0, y1-20)), f"Element {i+1}", fill=box_color)
                
                # Crop the element from original image
                cropped = after_img.crop((x1, y1, x2, y2))
                crop_path = debug_dir / f"after_element_{i+1}_crop.png"
                cropped.save(crop_path)
                cropped_images.append(str(crop_path))
            
            # Save full images with bounding boxes
            before_debug_path = debug_dir / "before_with_boxes.png"
            after_debug_path = debug_dir / "after_with_boxes.png"
            before_debug.save(before_debug_path)
            after_debug.save(after_debug_path)
            
            # Create summary info file
            summary_path = debug_dir / "summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Debug Images for Job: {job_id}\n")
                f.write(f"Region of Interest: {region_of_interest or 'Not specified'}\n")
                f.write(f"Before Elements Count: {len(before_elements)}\n")
                f.write(f"After Elements Count: {len(after_elements)}\n")
                f.write(f"Total Cropped Images: {len(cropped_images)}\n")
                f.write(f"\nCropped Images:\n")
                for crop_path in cropped_images:
                    f.write(f"  - {crop_path}\n")
            
            debug_paths = {
                "before_full": str(before_debug_path),
                "after_full": str(after_debug_path),
                "cropped_images": cropped_images,
                "summary": str(summary_path),
                "debug_directory": str(debug_dir)
            }
            
            session_logger.info("Debug images saved successfully",
                              debug_dir=str(debug_dir),
                              cropped_count=len(cropped_images),
                              before_elements=len(before_elements),
                              after_elements=len(after_elements))
            
            return debug_paths
            
        except Exception as e:
            session_logger.error("Failed to save debug images", error=str(e))
            return {"error": f"Failed to save debug images: {str(e)}"}

    async def _save_element_debug_images(
        self,
        before_element_data: Dict[str, Any],
        after_element_data: Dict[str, Any],
        job_id: str,
        session_logger
    ) -> Dict[str, str]:
        """Save debug images for the specific element crops used in GPT Vision validation"""
        debug_paths = {}
        
        try:
            # Get element type from data
            element_type = before_element_data.get("element_type", "element")
            element_name = element_type.replace(" ", "_").lower()
            
            # Create debug directory if it doesn't exist
            debug_dir = Path(settings.debug_output_dir) / f"{element_name}_comparison" / job_id
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the element crop images directly
            before_crop_path = debug_dir / f"before_{element_name}_crop.png"
            after_crop_path = debug_dir / f"after_{element_name}_crop.png"
            
            before_element_data["crop_image"].save(before_crop_path)
            after_element_data["crop_image"].save(after_crop_path)
            
            # Create detailed summary with element detection info
            summary_path = debug_dir / f"{element_name}_analysis.txt"
            with open(summary_path, 'w') as f:
                f.write(f"{element_type.title()} Comparison Debug Images for Job: {job_id}\n")
                f.write(f"="*60 + "\n\n")
                
                f.write(f"BEFORE {element_type.upper()}:\n")
                f.write(f"- Crop ID: {before_element_data.get('crop_id', 'N/A')}\n")
                f.write(f"- Detection Confidence: {before_element_data.get('confidence', 0.0):.3f}\n")
                f.write(f"- Reasoning: {before_element_data.get('reasoning', 'N/A')}\n")
                f.write(f"- Bbox: {before_element_data.get('bbox', 'N/A')}\n")
                if 'selection_reasoning' in before_element_data:
                    f.write(f"- Selection Reasoning: {before_element_data['selection_reasoning']}\n")
                    f.write(f"- Total Candidates: {before_element_data.get('total_candidates', 1)}\n")
                
                f.write(f"\nAFTER {element_type.upper()}:\n")
                f.write(f"- Crop ID: {after_element_data.get('crop_id', 'N/A')}\n")
                f.write(f"- Detection Confidence: {after_element_data.get('confidence', 0.0):.3f}\n")
                f.write(f"- Reasoning: {after_element_data.get('reasoning', 'N/A')}\n")
                f.write(f"- Bbox: {after_element_data.get('bbox', 'N/A')}\n")
                if 'selection_reasoning' in after_element_data:
                    f.write(f"- Selection Reasoning: {after_element_data['selection_reasoning']}\n")
                    f.write(f"- Total Candidates: {after_element_data.get('total_candidates', 1)}\n")
                
                f.write(f"\nFILES SAVED:\n")
                f.write(f"- Before {element_type} crop: before_{element_name}_crop.png\n")
                f.write(f"- After {element_type} crop: after_{element_name}_crop.png\n")
            
            debug_paths = {
                f"before_{element_name}_crop": str(before_crop_path),
                f"after_{element_name}_crop": str(after_crop_path),
                f"{element_name}_analysis": str(summary_path),
                "debug_directory": str(debug_dir),
                "validation_type": f"gpt_vision_{element_name}_comparison"
            }
            
            session_logger.info("Element debug images saved successfully",
                              debug_dir=str(debug_dir),
                              element_type=element_type,
                              before_confidence=before_element_data.get('confidence', 0.0),
                              after_confidence=after_element_data.get('confidence', 0.0))
            
            return debug_paths
            
        except Exception as e:
            session_logger.error("Failed to save element debug images", error=str(e))
            return {"error": f"Failed to save element debug images: {str(e)}"}

    async def cleanup(self) -> None:
        """Cleanup service resources"""
        self.logger.info("Cleaning up validation service")
        
        # Cleanup dependent services
        if self.omniparser_service:
            await self.omniparser_service.cleanup()
        if self.clip_service:
            await self.clip_service.cleanup()
        if self.gpt_service:
            await self.gpt_service.cleanup()
        if self.element_detection_service:
            await self.element_detection_service.cleanup()
        
        # Clear job tracking
        async with self._job_lock:
            self._active_jobs.clear()


# Global service instance
_validation_service: Optional[ValidationService] = None
_service_lock = asyncio.Lock()


async def get_validation_service() -> ValidationService:
    """Get or create the global validation service instance"""
    global _validation_service
    
    if _validation_service is None:
        async with _service_lock:
            if _validation_service is None:
                _validation_service = ValidationService()
                await _validation_service.initialize()
    
    return _validation_service


async def validation_service_context():
    """Context manager for validation service lifecycle"""
    service = ValidationService()
    try:
        await service.initialize()
        yield service
    finally:
        await service.cleanup()