"""
Production-ready CLIP service for semantic filtering of UI elements.

This service provides async-compatible CLIP-based semantic filtering of UI elements
to identify the most relevant elements for QA validation before expensive GPT processing.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, AsyncGenerator
import tempfile
import os
import shutil

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import structlog

from ..core.config import settings
from ..core.logging import get_logger
from ..schemas.validation import UIElement, BoundingBox


logger = get_logger(__name__)


class CLIPServiceError(Exception):
    """Base exception for CLIP service errors"""
    pass


class CLIPModelLoadError(CLIPServiceError):
    """Raised when CLIP model fails to load"""
    pass


class CLIPProcessingError(CLIPServiceError):
    """Raised when CLIP processing fails"""
    pass


class CLIPService:
    """
    Production-ready async CLIP service for UI element semantic filtering.
    
    Features:
    - Async/await support for non-blocking operations
    - Proper resource management and cleanup
    - Structured logging with contextual information
    - Configuration integration
    - Error handling and recovery
    - Memory optimization for batch processing
    - Progress tracking for long-running operations
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 device: Optional[str] = None,
                 similarity_threshold: Optional[float] = None,
                 max_elements: Optional[int] = None):
        """
        Initialize CLIP service.
        
        Args:
            model_name: CLIP model to use (defaults to config)
            device: Device to run on (auto-detect if None)
            similarity_threshold: Minimum similarity score (defaults to config)
            max_elements: Maximum elements to return (defaults to config)
        """
        self.model_name = model_name or settings.clip_model
        self.similarity_threshold = similarity_threshold or settings.clip_similarity_threshold
        self.max_elements = max_elements or settings.clip_max_elements
        
        # Auto-detect device
        if device is None:
            if settings.use_gpu and torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available() and settings.use_gpu:
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None
        self._is_initialized = False
        self._initialization_lock = asyncio.Lock()
        self._session_counter = 0
        
        logger.info(
            "CLIP service initialized",
            model_name=self.model_name,
            device=self.device,
            similarity_threshold=self.similarity_threshold,
            max_elements=self.max_elements
        )
    
    async def initialize(self) -> None:
        """
        Initialize CLIP model and processor asynchronously.
        Thread-safe and idempotent.
        """
        if self._is_initialized:
            return
            
        async with self._initialization_lock:
            if self._is_initialized:
                return
                
            logger.info("Loading CLIP model", model=self.model_name, device=self.device)
            
            try:
                # Load model and processor in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                
                model_task = loop.run_in_executor(
                    None, 
                    lambda: CLIPModel.from_pretrained(
                        self.model_name,
                        cache_dir=settings.huggingface_cache_dir
                    )
                )
                
                processor_task = loop.run_in_executor(
                    None,
                    lambda: CLIPProcessor.from_pretrained(
                        self.model_name,
                        cache_dir=settings.huggingface_cache_dir
                    )
                )
                
                # Wait for both to complete
                model, processor = await asyncio.gather(model_task, processor_task)
                
                # Move model to device
                self._model = model.to(self.device)
                self._processor = processor
                self._is_initialized = True
                
                logger.info(
                    "CLIP model loaded successfully",
                    model=self.model_name,
                    device=self.device,
                    memory_allocated=torch.cuda.memory_allocated() if self.device == "cuda" else None
                )
                
            except Exception as e:
                error_msg = f"Failed to load CLIP model {self.model_name}: {str(e)}"
                logger.error(error_msg, error=str(e), model=self.model_name)
                raise CLIPModelLoadError(error_msg) from e
    
    async def cleanup(self) -> None:
        """Clean up resources and free memory."""
        logger.info("Cleaning up CLIP service resources")
        
        if self._model is not None:
            if self.device == "cuda":
                self._model.cpu()
                torch.cuda.empty_cache()
            del self._model
            self._model = None
            
        if self._processor is not None:
            del self._processor
            self._processor = None
            
        self._is_initialized = False
        logger.info("CLIP service cleanup completed")
    
    def _ensure_initialized(self) -> None:
        """Ensure service is initialized, raise error if not."""
        if not self._is_initialized:
            raise CLIPServiceError("CLIP service not initialized. Call initialize() first.")
    
    async def _crop_element_async(self, 
                                 image: Image.Image, 
                                 element: Union[UIElement, Dict[str, Any]]) -> Optional[Image.Image]:
        """
        Crop UI element from full image asynchronously.
        
        Args:
            image: Full PIL Image
            element: UIElement schema or dict with bbox
            
        Returns:
            Cropped PIL Image or None if cropping fails
        """
        try:
            # Extract bounding box from different formats
            if isinstance(element, UIElement):
                bbox = element.bbox
                left, top = bbox.x, bbox.y
                right, bottom = bbox.x + bbox.width, bbox.y + bbox.height
            elif isinstance(element, dict):
                bbox = element.get('bbox')
                if isinstance(bbox, dict):
                    # Schema format: {x, y, width, height} or {left, top, right, bottom}
                    if 'width' in bbox and 'height' in bbox:
                        left, top = bbox['x'], bbox['y']
                        right, bottom = bbox['x'] + bbox['width'], bbox['y'] + bbox['height']
                    else:
                        left, top, right, bottom = bbox['left'], bbox['top'], bbox['right'], bbox['bottom']
                elif isinstance(bbox, list) and len(bbox) >= 4:
                    # List format: [left, top, right, bottom]
                    left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
                else:
                    raise ValueError(f"Unsupported bbox format: {type(bbox)}")
            else:
                raise ValueError(f"Unsupported element format: {type(element)}")
            
            # Ensure coordinates are integers
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            
            # Validate bounding box is within image bounds
            img_width, img_height = image.size
            left = max(0, min(left, img_width - 1))
            top = max(0, min(top, img_height - 1))
            right = max(left + 1, min(right, img_width))
            bottom = max(top + 1, min(bottom, img_height))
            
            # Ensure minimum size
            if right <= left:
                right = left + 1
            if bottom <= top:
                bottom = top + 1
            
            # Perform crop in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            cropped = await loop.run_in_executor(
                None, 
                lambda: image.crop((left, top, right, bottom))
            )
            
            return cropped
            
        except Exception as e:
            element_id = getattr(element, 'id', element.get('id', 'unknown')) if isinstance(element, (UIElement, dict)) else 'unknown'
            logger.warning(
                "Failed to crop element",
                element_id=element_id,
                error=str(e),
                bbox=bbox if 'bbox' in locals() else None
            )
            # Return small fallback image
            return Image.new('RGB', (32, 32), color='white')
    
    async def _calculate_similarities_async(self, 
                                          cropped_images: List[Image.Image], 
                                          text_prompt: str,
                                          session_id: str) -> List[float]:
        """
        Calculate CLIP similarity scores asynchronously.
        
        Args:
            cropped_images: List of cropped UI element images
            text_prompt: QA prompt to compare against
            session_id: Session identifier for logging
            
        Returns:
            List of similarity scores (0-1 range)
        """
        self._ensure_initialized()
        
        if not cropped_images:
            return []
        
        try:
            logger.info(
                "Calculating CLIP similarities",
                session_id=session_id,
                num_images=len(cropped_images),
                prompt=text_prompt[:50] + "..." if len(text_prompt) > 50 else text_prompt
            )
            
            # Process inputs in thread pool
            loop = asyncio.get_event_loop()
            
            def process_inputs():
                return self._processor(
                    text=[text_prompt],
                    images=cropped_images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
            
            inputs = await loop.run_in_executor(None, process_inputs)
            
            # Run model inference in thread pool
            def run_inference():
                with torch.no_grad():
                    return self._model(**inputs)
            
            outputs = await loop.run_in_executor(None, run_inference)
            
            # Process results
            logits_per_image = outputs.logits_per_image.flatten()
            
            logger.debug(
                "CLIP inference completed",
                session_id=session_id,
                logits_shape=logits_per_image.shape,
                logits_range=f"{logits_per_image.min().item():.3f} to {logits_per_image.max().item():.3f}"
            )
            
            # Normalize similarities to 0-1 range using min-max scaling
            if len(logits_per_image) > 1:
                min_logit = logits_per_image.min()
                max_logit = logits_per_image.max()
                
                if max_logit - min_logit > 1e-6:
                    similarities = ((logits_per_image - min_logit) / (max_logit - min_logit)).cpu().numpy()
                else:
                    similarities = np.full(len(logits_per_image), 0.5)
            else:
                similarities = np.array([0.5])
            
            # Scale to 0.1-0.9 range to avoid extremes
            similarities = [0.1 + 0.8 * score for score in similarities.tolist()]
            
            logger.info(
                "CLIP similarities calculated",
                session_id=session_id,
                num_elements=len(similarities),
                score_range=f"{min(similarities):.3f} to {max(similarities):.3f}",
                top_scores=[f"{s:.3f}" for s in sorted(similarities, reverse=True)[:5]]
            )
            
            return similarities
            
        except Exception as e:
            error_msg = f"CLIP similarity calculation failed: {str(e)}"
            logger.error(
                error_msg,
                session_id=session_id,
                num_images=len(cropped_images),
                prompt=text_prompt,
                error=str(e)
            )
            
            # Return fallback scores
            fallback_scores = [0.1] * len(cropped_images)
            logger.warning(
                "Using fallback similarity scores",
                session_id=session_id,
                num_scores=len(fallback_scores)
            )
            return fallback_scores
    
    @asynccontextmanager
    async def _temp_crops_dir(self, session_id: str) -> AsyncGenerator[Optional[Path], None]:
        """Context manager for temporary crops directory."""
        temp_dir = None
        try:
            if settings.debug:
                temp_dir = Path(tempfile.mkdtemp(prefix=f"clip_crops_{session_id}_"))
                logger.debug("Created temporary crops directory", path=str(temp_dir))
            yield temp_dir
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug("Cleaned up temporary crops directory", path=str(temp_dir))
    
    async def filter_elements_async(self,
                                   image_path: Union[str, Path],
                                   elements: List[Union[UIElement, Dict[str, Any]]],
                                   qa_prompt: str,
                                   max_elements: Optional[int] = None,
                                   similarity_threshold: Optional[float] = None,
                                   save_debug_crops: bool = False) -> List[Union[UIElement, Dict[str, Any]]]:
        """
        Filter UI elements using CLIP semantic similarity asynchronously.
        
        Args:
            image_path: Path to the full screenshot image
            elements: List of UIElement schemas or dicts from parser
            qa_prompt: QA prompt to filter by relevance
            max_elements: Maximum number of elements to return
            similarity_threshold: Minimum similarity score to consider
            save_debug_crops: Whether to save debug crop images
            
        Returns:
            Filtered list of most relevant elements with similarity scores
            
        Raises:
            CLIPServiceError: If service not initialized
            CLIPProcessingError: If processing fails
        """
        self._ensure_initialized()
        
        # Use instance defaults if not provided
        max_elements = max_elements or self.max_elements
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        if not elements:
            logger.warning("No elements provided for CLIP filtering")
            return []
        
        # Generate session ID for tracking
        self._session_counter += 1
        session_id = f"{int(time.time() * 1000) % 100000}_{self._session_counter}"
        
        logger.info(
            "Starting CLIP filtering session",
            session_id=session_id,
            num_elements=len(elements),
            prompt=qa_prompt[:100] + "..." if len(qa_prompt) > 100 else qa_prompt,
            max_elements=max_elements,
            similarity_threshold=similarity_threshold
        )
        
        try:
            # Load image in thread pool
            loop = asyncio.get_event_loop()
            full_image = await loop.run_in_executor(
                None,
                lambda: Image.open(image_path).convert('RGB')
            )
            
            logger.info(
                "Image loaded",
                session_id=session_id,
                image_path=str(image_path),
                image_size=full_image.size
            )
            
            # Crop all elements concurrently
            crop_tasks = [
                self._crop_element_async(full_image, element)
                for element in elements
            ]
            
            cropped_results = await asyncio.gather(*crop_tasks, return_exceptions=True)
            
            # Filter out failed crops and build valid elements list
            cropped_images = []
            valid_elements = []
            
            for i, (element, crop_result) in enumerate(zip(elements, cropped_results)):
                if isinstance(crop_result, Exception):
                    element_id = getattr(element, 'id', element.get('id', f'element_{i}')) if hasattr(element, 'id') or isinstance(element, dict) else f'element_{i}'
                    logger.warning(
                        "Element crop failed",
                        session_id=session_id,
                        element_id=element_id,
                        error=str(crop_result)
                    )
                    continue
                    
                if crop_result is None or crop_result.size[0] <= 0 or crop_result.size[1] <= 0:
                    element_id = getattr(element, 'id', element.get('id', f'element_{i}')) if hasattr(element, 'id') or isinstance(element, dict) else f'element_{i}'
                    logger.warning(
                        "Invalid crop size",
                        session_id=session_id,
                        element_id=element_id,
                        size=crop_result.size if crop_result else None
                    )
                    continue
                
                cropped_images.append(crop_result)
                valid_elements.append(element)
            
            if not cropped_images:
                logger.warning(
                    "No valid cropped images to process",
                    session_id=session_id
                )
                return []
            
            logger.info(
                "Elements cropped successfully",
                session_id=session_id,
                valid_elements=len(valid_elements),
                failed_crops=len(elements) - len(valid_elements)
            )
            
            # Save debug crops if requested
            async with self._temp_crops_dir(session_id) as crops_dir:
                if save_debug_crops and crops_dir:
                    await self._save_debug_crops(
                        cropped_images, valid_elements, crops_dir, session_id
                    )
                
                # Calculate similarities
                similarities = await self._calculate_similarities_async(
                    cropped_images, qa_prompt, session_id
                )
                
                # Add similarity scores to elements
                for element, score in zip(valid_elements, similarities):
                    if isinstance(element, UIElement):
                        element.clip_similarity = score
                    elif isinstance(element, dict):
                        element['clip_similarity'] = score
                
                # Filter by threshold and sort by similarity
                filtered_elements = [
                    element for element, score in zip(valid_elements, similarities)
                    if score >= similarity_threshold
                ]
                
                # Sort by similarity (highest first) - handle different data types
                def get_similarity_score(x):
                    if hasattr(x, 'clip_similarity'):
                        return x.clip_similarity or 0
                    elif isinstance(x, dict) and 'clip_similarity' in x:
                        return x.get('clip_similarity', 0)
                    elif isinstance(x, list) and len(x) > 0:
                        # Handle list format - assume last element might be similarity score
                        return 0
                    else:
                        return 0
                
                filtered_elements.sort(key=get_similarity_score, reverse=True)
                
                # Limit to max_elements
                result = filtered_elements[:max_elements]
                
                logger.info(
                    "CLIP filtering completed",
                    session_id=session_id,
                    input_elements=len(elements),
                    valid_elements=len(valid_elements),
                    passed_threshold=len(filtered_elements),
                    final_count=len(result),
                    selected_scores=[
                        f"{x.clip_similarity if hasattr(x, 'clip_similarity') else x.get('clip_similarity', 0):.3f}"
                        for x in result[:5]
                    ]
                )
                
                return result
                
        except Exception as e:
            error_msg = f"CLIP filtering failed for session {session_id}: {str(e)}"
            logger.error(
                error_msg,
                session_id=session_id,
                error=str(e),
                num_elements=len(elements),
                qa_prompt=qa_prompt
            )
            raise CLIPProcessingError(error_msg) from e
    
    async def _save_debug_crops(self,
                               cropped_images: List[Image.Image],
                               elements: List[Union[UIElement, Dict[str, Any]]],
                               crops_dir: Path,
                               session_id: str) -> None:
        """Save cropped images for debugging purposes."""
        logger.debug(
            "Saving debug crops",
            session_id=session_id,
            num_crops=len(cropped_images),
            output_dir=str(crops_dir)
        )
        
        def save_crop(i: int, img: Image.Image, element: Union[UIElement, Dict[str, Any]]) -> None:
            try:
                element_id = (
                    element.id if isinstance(element, UIElement) 
                    else element.get('id', f'element_{i}')
                )
                
                # Create safe filename
                safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(element_id))
                filename = f"{i:03d}_{safe_id}.png"
                filepath = crops_dir / filename
                
                img.save(filepath)
                
            except Exception as e:
                logger.warning(
                    "Failed to save debug crop",
                    session_id=session_id,
                    crop_index=i,
                    error=str(e)
                )
        
        # Save crops concurrently in thread pool
        loop = asyncio.get_event_loop()
        save_tasks = [
            loop.run_in_executor(None, save_crop, i, img, element)
            for i, (img, element) in enumerate(zip(cropped_images, elements))
        ]
        
        await asyncio.gather(*save_tasks, return_exceptions=True)
        
        logger.debug(
            "Debug crops saved",
            session_id=session_id,
            output_dir=str(crops_dir)
        )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get service health status."""
        return {
            "service": "clip_service",
            "initialized": self._is_initialized,
            "model_name": self.model_name,
            "device": self.device,
            "memory_allocated": (
                torch.cuda.memory_allocated() / 1024**2  # MB
                if self.device == "cuda" and torch.cuda.is_available()
                else None
            ),
            "similarity_threshold": self.similarity_threshold,
            "max_elements": self.max_elements,
            "sessions_processed": self._session_counter
        }


# Global service instance
_clip_service: Optional[CLIPService] = None


async def get_clip_service() -> CLIPService:
    """
    Get global CLIP service instance for dependency injection.
    Initializes service if not already done.
    
    Returns:
        CLIPService: Initialized CLIP service instance
    """
    global _clip_service
    
    if _clip_service is None:
        _clip_service = CLIPService()
        await _clip_service.initialize()
    
    return _clip_service


async def cleanup_clip_service() -> None:
    """Clean up global CLIP service instance."""
    global _clip_service
    
    if _clip_service is not None:
        await _clip_service.cleanup()
        _clip_service = None


# Context manager for service lifecycle
@asynccontextmanager
async def clip_service_context() -> AsyncGenerator[CLIPService, None]:
    """
    Context manager for CLIP service lifecycle management.
    
    Usage:
        async with clip_service_context() as clip_service:
            results = await clip_service.filter_elements_async(...)
    """
    service = CLIPService()
    try:
        await service.initialize()
        yield service
    finally:
        await service.cleanup()