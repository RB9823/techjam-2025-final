"""
Generic UI Element Detection Service
Advanced service for detecting and validating any type of UI element changes.
Based on the enhanced generic_ui_validator.py from latest updates.
"""
import asyncio
import json
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
from PIL import Image
import structlog

from ..core.config import settings
from ..core.exceptions import ValidationError, AIServiceError
from ..schemas.validation import UIElement, BoundingBox, DetectedChange
from .gpt_service import GPTService, get_gpt_service

logger = structlog.get_logger(__name__)


@dataclass
class ElementDetectionResult:
    """Result from individual element detection"""
    crop_id: int
    is_target_element: bool
    confidence: float
    reasoning: str
    element_type: str
    crop_image: Image.Image


@dataclass
class BatchDetectionResult:
    """Result from batch element detection"""
    batch_id: int
    results: List[ElementDetectionResult]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ElementSelectionResult:
    """Result from element selection when multiple found"""
    selected_crop_id: int
    crop_image: Image.Image
    confidence: float
    reasoning: str
    element_type: str
    total_candidates: int


class UIElementDetectionPrompts:
    """Dynamic prompt generation for different UI element types"""
    
    # Element type definitions with specific characteristics
    ELEMENT_PATTERNS = {
        "heart icon": {
            "shapes": ["♥", "♡", "💖", "🤍", "❤️", "💕"],
            "visual_patterns": [
                "Two rounded bumps at the top, pointed bottom (classic heart shape)",
                "Symmetrical curved shape tapering to a point"
            ],
            "contexts": [
                "Like buttons in social media apps",
                "Favorite/love buttons",
                "Bookmark icons",
                "Save to favorites features"
            ],
            "variations": [
                "Outlined/hollow hearts (not liked state)",
                "Filled/solid hearts (liked state)",
                "Different colors (red, white, pink, gray)",
                "Different sizes (small icons to large buttons)"
            ]
        },
        "button": {
            "shapes": ["rectangular", "rounded rectangle", "pill-shaped", "circular"],
            "visual_patterns": [
                "Clickable rectangular or rounded element",
                "Often has background color or border",
                "May contain text or icons"
            ],
            "contexts": [
                "Submit buttons in forms",
                "Action buttons in dialogs",
                "Navigation buttons",
                "Call-to-action elements"
            ],
            "variations": [
                "Primary buttons (prominent styling)",
                "Secondary buttons (subtle styling)",
                "Disabled state (grayed out)",
                "Loading state (with spinner)"
            ]
        },
        "menu icon": {
            "shapes": ["three horizontal lines", "hamburger", "≡"],
            "visual_patterns": [
                "Three parallel horizontal lines",
                "Sometimes called 'hamburger menu'",
                "Usually in top corner of interface"
            ],
            "contexts": [
                "Navigation menu triggers",
                "Mobile app menus",
                "Sidebar toggles"
            ],
            "variations": [
                "Simple three lines",
                "With dots or decorations",
                "Animated states",
                "Different orientations"
            ]
        }
    }
    
    @classmethod
    def get_element_patterns(cls, element_type: str) -> Dict[str, Any]:
        """Get patterns for a specific element type, with fallback to generic"""
        normalized_type = element_type.lower().strip()
        
        # Try exact match first
        if normalized_type in cls.ELEMENT_PATTERNS:
            return cls.ELEMENT_PATTERNS[normalized_type]
        
        # Try partial matches
        for pattern_key, patterns in cls.ELEMENT_PATTERNS.items():
            if normalized_type in pattern_key or pattern_key in normalized_type:
                return patterns
        
        # Generic fallback
        return {
            "shapes": ["various shapes possible"],
            "visual_patterns": [
                f"Visual element that appears to be a {element_type}",
                "Interactive UI component"
            ],
            "contexts": [
                "Part of user interface",
                "Interactive element users can click/tap"
            ],
            "variations": [
                "Different visual states",
                "Various sizes and colors",
                "Different interaction states"
            ]
        }
    
    @classmethod
    def generate_detection_prompt(cls, element_type: str, batch_size: int = 1) -> str:
        """Generate dynamic prompt for detecting specific UI element types"""
        patterns = cls.get_element_patterns(element_type)
        
        if batch_size == 1:
            # Single image prompt
            prompt = f"""You are analyzing a cropped UI element to determine if it contains a {element_type} that users can interact with.

TASK: Determine if this image shows a {element_type}.

ELEMENT CHARACTERISTICS TO LOOK FOR:

Visual Shapes & Patterns:
{chr(10).join(f"- {pattern}" for pattern in patterns["visual_patterns"])}

Common Contexts:
{chr(10).join(f"- {context}" for context in patterns["contexts"])}

Variations to Consider:
{chr(10).join(f"- {variation}" for variation in patterns["variations"])}

BE VERY CAREFUL TO LOOK FOR:
- Elements that might be small or subtle
- Partially visible or cropped elements  
- Stylized or abstract representations
- Elements that blend into the background
- Interactive elements within larger UI groups

NOT {element_type}:
- Pure text without the actual element
- Numbers or counts alone
- Other UI elements that don't match the description
- Random decorative elements

RESPONSE FORMAT (JSON only):
{{
    "is_target_element": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Describe exactly what you see and why it matches/doesn't match"
}}

Analyze the image carefully and respond in JSON format only."""
            
        else:
            # Batch processing prompt
            prompt = f"""You are analyzing {batch_size} cropped UI elements to determine which contain {element_type}.

TASK: For each of the {batch_size} images, determine if it shows a {element_type} that users can interact with.

ELEMENT CHARACTERISTICS TO LOOK FOR:

Visual Shapes & Patterns:
{chr(10).join(f"- {pattern}" for pattern in patterns["visual_patterns"])}

Common Contexts:
{chr(10).join(f"- {context}" for context in patterns["contexts"])}

Variations to Consider:
{chr(10).join(f"- {variation}" for variation in patterns["variations"])}

ANALYSIS INSTRUCTIONS:
- Examine each image carefully for the specified element type
- Look for subtle, small, or partially visible elements
- Consider stylized or abstract representations
- Note interactive elements within larger UI groups

RESPONSE FORMAT (JSON only):
{{
    "results": [
        {{
            "image_index": 0,
            "is_target_element": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "What you see in image 0"
        }},
        {{
            "image_index": 1, 
            "is_target_element": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "What you see in image 1"
        }}
        // ... for all {batch_size} images
    ]
}}

Analyze all {batch_size} images and respond in JSON format only."""
        
        return prompt
    
    @classmethod
    def generate_comparison_prompt(cls, element_type: str, expected_change: str) -> str:
        """Generate prompt for comparing elements between before/after states"""
        
        return f"""You are validating a UI change between before/after states of a {element_type}.

EXPECTED CHANGE: {expected_change}

TASK: Compare the two {element_type} elements and determine if the expected change occurred.

ANALYSIS CRITERIA:
1. Visual Similarity: Do these appear to be the same UI element?
2. State Changes: What visual changes occurred (color, style, state, etc.)?
3. Expected vs Actual: Does the change match what was expected?

BEFORE STATE: [First image shows the before state]
AFTER STATE: [Second image shows the after state]

Consider common {element_type} changes:
- Visual state changes (color, fill, style)
- Interaction state changes (enabled/disabled, active/inactive)
- Content changes (text, icons, labels)
- Layout or position changes

RESPONSE FORMAT (JSON only):
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of what changed and whether it matches expectations",
    "detected_changes": ["list", "of", "specific", "changes"],
    "matches_expectation": true/false
}}

Compare the images and respond in JSON format only."""


class ElementDetectionService:
    """
    Advanced UI element detection service with support for any element type.
    Implements the enhanced detection capabilities from the latest updates.
    """
    
    def __init__(self, 
                 gpt_service: Optional[GPTService] = None,
                 batch_size: int = 4):
        """
        Initialize element detection service.
        
        Args:
            gpt_service: GPT service instance (will be auto-created if None)
            batch_size: Number of crops to process in each GPT batch call
        """
        self.gpt_service = gpt_service
        self.batch_size = batch_size or settings.gpt_batch_size
        self.prompt_generator = UIElementDetectionPrompts()
        self.logger = logger.bind(service="element_detection")
        
        # Session tracking
        self._session_counter = 0
        self._detection_stats = {
            "total_detections": 0,
            "successful_detections": 0,
            "failed_detections": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the detection service"""
        if not self.gpt_service:
            self.gpt_service = await get_gpt_service()
        
        await self.gpt_service.initialize()
        self.logger.info("Element detection service initialized")
    
    async def detect_elements_in_crops(self,
                                     crops: List[Image.Image],
                                     element_type: str,
                                     session_id: Optional[str] = None) -> List[ElementDetectionResult]:
        """
        Detect specific UI elements in a list of cropped images using batch GPT processing.
        
        Args:
            crops: List of cropped PIL images
            element_type: Type of element to detect (e.g., "heart icon", "submit button")
            session_id: Optional session identifier for logging
            
        Returns:
            List of detection results for each crop
        """
        if not crops:
            return []
        
        session_id = session_id or str(uuid.uuid4())[:8]
        session_logger = self.logger.bind(session_id=session_id, element_type=element_type)
        
        session_logger.info("Starting element detection", 
                           total_crops=len(crops),
                           batch_size=self.batch_size)
        
        all_results = []
        
        # Process crops in batches
        for batch_idx in range(0, len(crops), self.batch_size):
            batch_crops = crops[batch_idx:batch_idx + self.batch_size]
            batch_id = batch_idx // self.batch_size
            
            try:
                batch_result = await self._detect_batch(
                    batch_crops, element_type, batch_id, session_id
                )
                
                if batch_result.success:
                    all_results.extend(batch_result.results)
                    self._detection_stats["successful_detections"] += len(batch_result.results)
                else:
                    session_logger.warning("Batch detection failed", 
                                         batch_id=batch_id,
                                         error=batch_result.error_message)
                    self._detection_stats["failed_detections"] += len(batch_crops)
                
            except Exception as e:
                session_logger.error("Batch processing error", 
                                   batch_id=batch_id,
                                   error=str(e))
                self._detection_stats["failed_detections"] += len(batch_crops)
        
        session_logger.info("Element detection completed",
                           total_results=len(all_results),
                           found_elements=len([r for r in all_results if r.is_target_element]))
        
        return all_results
    
    async def _detect_batch(self,
                           crops_batch: List[Image.Image],
                           element_type: str,
                           batch_id: int,
                           session_id: str) -> BatchDetectionResult:
        """Detect UI elements in a batch of crops using a single GPT call"""
        start_time = time.time()
        batch_size = len(crops_batch)
        
        try:
            # Validate batch inputs
            if not crops_batch:
                raise ValueError("Empty crops batch")
            
            valid_crops = []
            crop_indices = []
            
            # Validate each crop and filter out invalid ones
            for i, crop in enumerate(crops_batch):
                if crop is None or crop.size[0] <= 5 or crop.size[1] <= 5:
                    continue
                
                valid_crops.append(crop)
                crop_indices.append(i)
            
            if not valid_crops:
                return BatchDetectionResult(
                    batch_id=batch_id,
                    results=[],
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="No valid crops in batch"
                )
            
            # Convert images to base64 for GPT
            image_data = []
            for crop in valid_crops:
                image_b64 = self._encode_image_base64(crop)
                image_data.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                })
            
            # Generate dynamic prompt for this element type and batch size
            prompt = self.prompt_generator.generate_detection_prompt(element_type, len(valid_crops))
            
            # Prepare GPT message content
            content = [{"type": "text", "text": prompt}] + image_data
            
            # Make batch GPT call through the GPT service
            gpt_request = {
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 800,
                "temperature": 0.1
            }
            
            response = await self.gpt_service._make_gpt_request(gpt_request, session_id)
            result_text = response.get("content", "")
            
            # Parse JSON response
            result_json = self._parse_gpt_json(result_text, session_id, f"batch {batch_id}")
            
            # Process batch results
            batch_results = []
            
            if len(valid_crops) == 1:
                # Single image response format
                is_target = result_json.get("is_target_element", False)
                confidence = float(result_json.get("confidence", 0.0))
                reasoning = result_json.get("reasoning", "No reasoning provided")
                
                batch_results.append(ElementDetectionResult(
                    crop_id=crop_indices[0],
                    is_target_element=is_target,
                    confidence=confidence,
                    reasoning=reasoning,
                    element_type=element_type,
                    crop_image=valid_crops[0]
                ))
            else:
                # Multi-image batch response format
                results_list = result_json.get("results", [])
                
                for i, result_data in enumerate(results_list):
                    if i >= len(valid_crops):
                        break
                    
                    is_target = result_data.get("is_target_element", False)
                    confidence = float(result_data.get("confidence", 0.0))
                    reasoning = result_data.get("reasoning", "No reasoning provided")
                    
                    batch_results.append(ElementDetectionResult(
                        crop_id=crop_indices[i],
                        is_target_element=is_target,
                        confidence=confidence,
                        reasoning=reasoning,
                        element_type=element_type,
                        crop_image=valid_crops[i]
                    ))
            
            return BatchDetectionResult(
                batch_id=batch_id,
                results=batch_results,
                processing_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Batch detection failed: {str(e)}"
            self.logger.error("Batch detection error",
                            batch_id=batch_id,
                            session_id=session_id,
                            error=str(e))
            
            return BatchDetectionResult(
                batch_id=batch_id,
                results=[],
                processing_time=time.time() - start_time,
                success=False,
                error_message=error_msg
            )
    
    async def select_best_element(self,
                                candidates: List[ElementDetectionResult],
                                element_type: str,
                                session_id: Optional[str] = None) -> Optional[ElementSelectionResult]:
        """
        Select the best element when multiple candidates are found.
        
        Args:
            candidates: List of detected element candidates
            element_type: Type of element being selected
            session_id: Session identifier for logging
            
        Returns:
            Selection result with the best candidate, or None if none suitable
        """
        if not candidates:
            return None
        
        # Filter to only target elements
        target_elements = [c for c in candidates if c.is_target_element]
        
        if not target_elements:
            return None
        
        if len(target_elements) == 1:
            # Only one candidate, use it
            candidate = target_elements[0]
            return ElementSelectionResult(
                selected_crop_id=candidate.crop_id,
                crop_image=candidate.crop_image,
                confidence=candidate.confidence,
                reasoning=f"Only {element_type} found: {candidate.reasoning}",
                element_type=element_type,
                total_candidates=len(candidates)
            )
        
        # Multiple candidates - use GPT to select the best one
        session_id = session_id or str(uuid.uuid4())[:8]
        session_logger = self.logger.bind(session_id=session_id)
        
        session_logger.info("Selecting best element from candidates",
                           candidates_count=len(target_elements),
                           element_type=element_type)
        
        try:
            # Prepare images for comparison
            comparison_images = []
            for i, candidate in enumerate(target_elements):
                image_b64 = self._encode_image_base64(candidate.crop_image)
                comparison_images.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                })
            
            # Generate selection prompt
            selection_prompt = f"""You are selecting the best {element_type} from {len(target_elements)} candidates.

TASK: Choose which image shows the most typical, clear, and interactable {element_type}.

SELECTION CRITERIA:
1. Clarity and visibility
2. Typical appearance for the element type
3. Likely to be the main interactive element
4. Best represents what users would click/interact with

RESPONSE FORMAT (JSON only):
{{
    "selected_index": 0-{len(target_elements)-1},
    "confidence": 0.0-1.0,
    "reasoning": "Why this {element_type} is the best choice"
}}

Analyze all images and select the best {element_type} in JSON format only."""
            
            # Prepare GPT request
            content = [{"type": "text", "text": selection_prompt}] + comparison_images
            gpt_request = {
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 300,
                "temperature": 0.1
            }
            
            response = await self.gpt_service._make_gpt_request(gpt_request, session_id)
            result_text = response.get("content", "")
            result_json = self._parse_gpt_json(result_text, session_id, "selection")
            
            # Parse selection result
            selected_index = int(result_json.get("selected_index", 0))
            if 0 <= selected_index < len(target_elements):
                selected_candidate = target_elements[selected_index]
                
                return ElementSelectionResult(
                    selected_crop_id=selected_candidate.crop_id,
                    crop_image=selected_candidate.crop_image,
                    confidence=float(result_json.get("confidence", selected_candidate.confidence)),
                    reasoning=result_json.get("reasoning", "Selected as best candidate"),
                    element_type=element_type,
                    total_candidates=len(candidates)
                )
            else:
                # Fallback to highest confidence
                best_candidate = max(target_elements, key=lambda x: x.confidence)
                return ElementSelectionResult(
                    selected_crop_id=best_candidate.crop_id,
                    crop_image=best_candidate.crop_image,
                    confidence=best_candidate.confidence,
                    reasoning="Selected highest confidence candidate after selection failed",
                    element_type=element_type,
                    total_candidates=len(candidates)
                )
                
        except Exception as e:
            session_logger.error("Element selection failed", error=str(e))
            # Fallback to highest confidence
            best_candidate = max(target_elements, key=lambda x: x.confidence)
            return ElementSelectionResult(
                selected_crop_id=best_candidate.crop_id,
                crop_image=best_candidate.crop_image,
                confidence=best_candidate.confidence,
                reasoning=f"Fallback selection due to error: {str(e)}",
                element_type=element_type,
                total_candidates=len(candidates)
            )
    
    async def validate_element_change(self,
                                    before_element: Image.Image,
                                    after_element: Image.Image,
                                    element_type: str,
                                    expected_change: str,
                                    session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate that an expected change occurred between before/after element states.
        
        Args:
            before_element: Cropped image of element before change
            after_element: Cropped image of element after change
            element_type: Type of element being validated
            expected_change: Description of expected change
            session_id: Session identifier for logging
            
        Returns:
            Validation result with is_valid, confidence, reasoning, etc.
        """
        session_id = session_id or str(uuid.uuid4())[:8]
        session_logger = self.logger.bind(session_id=session_id, element_type=element_type)
        
        session_logger.info("Validating element change", expected_change=expected_change)
        
        try:
            # Convert images to base64
            before_b64 = self._encode_image_base64(before_element)
            after_b64 = self._encode_image_base64(after_element)
            
            # Generate comparison prompt
            prompt = self.prompt_generator.generate_comparison_prompt(element_type, expected_change)
            
            # Prepare GPT request with both images
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_b64}"}}
            ]
            
            gpt_request = {
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            response = await self.gpt_service._make_gpt_request(gpt_request, session_id)
            result_text = response.get("content", "")
            result_json = self._parse_gpt_json(result_text, session_id, "validation")
            
            session_logger.info("Element change validation completed",
                              is_valid=result_json.get("is_valid"),
                              confidence=result_json.get("confidence"))
            
            return result_json
            
        except Exception as e:
            session_logger.error("Element change validation failed", error=str(e))
            raise ValidationError(f"Element change validation failed: {e}")
    
    def _encode_image_base64(self, image: Image.Image) -> str:
        """Encode PIL image as base64 string"""
        import io
        import base64
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode as base64
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _parse_gpt_json(self, response_text: str, session_id: str, context: str) -> Dict[str, Any]:
        """Parse GPT JSON response with error handling"""
        try:
            # Try to parse as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown or other formatting
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON object in text
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # Fallback response
            self.logger.warning("Failed to parse GPT JSON response",
                              session_id=session_id,
                              context=context,
                              response_text=response_text[:200])
            
            return {
                "is_target_element": False,
                "confidence": 0.0,
                "reasoning": f"Failed to parse GPT response: {response_text[:100]}..."
            }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the element detection service"""
        try:
            gpt_health = await self.gpt_service.get_health_status()
            
            return {
                "status": "healthy" if gpt_health.get("status") == "healthy" else "degraded",
                "gpt_service": gpt_health,
                "detection_stats": self._detection_stats,
                "batch_size": self.batch_size
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup service resources"""
        if self.gpt_service:
            await self.gpt_service.cleanup()


# Global service instance
_element_detection_service: Optional[ElementDetectionService] = None
_service_lock = asyncio.Lock()


async def get_element_detection_service() -> ElementDetectionService:
    """Get or create the global element detection service instance"""
    global _element_detection_service
    
    if _element_detection_service is None:
        async with _service_lock:
            if _element_detection_service is None:
                _element_detection_service = ElementDetectionService()
                await _element_detection_service.initialize()
    
    return _element_detection_service


@asynccontextmanager
async def element_detection_service_context():
    """Context manager for element detection service lifecycle"""
    service = ElementDetectionService()
    try:
        await service.initialize()
        yield service
    finally:
        await service.cleanup()