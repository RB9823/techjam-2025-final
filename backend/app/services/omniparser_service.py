"""
OmniParser Production Service
A production-ready async service for UI element detection and analysis.
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoProcessor
from ultralytics import YOLO

from app.core.config import settings
from app.schemas.validation import (
    BoundingBox,
    DetectedChange,
    DetectedException,
    UIElement,
    ValidationProgress
)

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class OmniParserService:
    """Production-ready OmniParser service with async support"""
    
    def __init__(self):
        """Initialize the OmniParser service"""
        self.device = "cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu"
        self.cache_dir = Path(settings.huggingface_cache_dir) if settings.huggingface_cache_dir else None
        self.detection_confidence = settings.detection_confidence
        self.max_gpt_calls = settings.max_gpt_calls
        self.use_nms = True
        self.nms_threshold = 0.1
        
        # Model instances (will be loaded lazily)
        self.icon_detector: Optional[YOLO] = None
        self.caption_processor: Optional[AutoProcessor] = None
        self.caption_model: Optional[AutoModelForCausalLM] = None
        self.openai_client = None
        
        # Caching for GPT results
        self.caption_cache: Dict[str, str] = {}
        
        # Thread pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Thread safety locks
        self._yolo_lock = threading.Lock()
        
        # Model loading status
        self.models_loaded = {
            "yolo": False,
            "florence2": False,
            "openai": False
        }
        
        logger.info(f"OmniParser service initialized on {self.device}")
    
    async def initialize_models(self) -> None:
        """Initialize all AI models asynchronously"""
        logger.info("Loading AI models...")
        
        try:
            # Load models in thread pool to avoid blocking
            await self._load_yolo_model()
            await self._load_florence2_model()
            await self._setup_openai_client()
            
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    async def _load_yolo_model(self) -> None:
        """Load YOLO detection model"""
        try:
            from huggingface_hub import hf_hub_download
            
            # Download model in thread pool
            model_path = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: hf_hub_download(
                    "microsoft/OmniParser-v2.0",
                    "icon_detect/model.pt",
                    cache_dir=self.cache_dir
                )
            )
            
            # Load YOLO model with thread safety
            self.icon_detector = YOLO(model_path)
            
            # Disable automatic model fusing to prevent multithreading issues
            # The fusing error occurs when multiple threads try to modify the same model
            if hasattr(self.icon_detector.model, 'fuse'):
                # Replace the fuse method with a no-op to prevent automatic fusing
                self.icon_detector.model.fuse = lambda verbose=True: self.icon_detector.model
            
            if self.device == "cuda":
                self.icon_detector.to('cuda')
            
            self.models_loaded["yolo"] = True
            logger.info("✓ YOLO detection model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    async def _load_florence2_model(self) -> None:
        """Load Florence-2 captioning model"""
        try:
            from huggingface_hub import snapshot_download
            
            # Download model in thread pool
            model_path = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: snapshot_download(
                    "microsoft/OmniParser-v2.0",
                    allow_patterns=["icon_caption/*"],
                    cache_dir=self.cache_dir
                )
            )
            
            florence_path = Path(model_path) / "icon_caption"
            
            # Load processor and model
            self.caption_processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base",
                trust_remote_code=True
            )
            
            self.caption_model = AutoModelForCausalLM.from_pretrained(
                florence_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(self.device)
            
            self.models_loaded["florence2"] = True
            logger.info("✓ Florence-2 captioning model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load Florence-2 model: {e}")
            raise
    
    async def _setup_openai_client(self) -> None:
        """Setup OpenAI client for GPT-4V"""
        try:
            if settings.openai_api_key:
                import openai
                self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
                self.models_loaded["openai"] = True
                logger.info("✓ OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not found, GPT-4V disabled")
        except ImportError:
            logger.warning("OpenAI library not installed, GPT-4V disabled")
        except Exception as e:
            logger.error(f"Failed to setup OpenAI client: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "models_loaded": self.models_loaded.copy(),
            "device": self.device,
            "cache_dir": str(self.cache_dir) if self.cache_dir else "default",
            "detection_confidence": self.detection_confidence
        }
    
    async def detect_ui_elements(
        self, 
        image: Image.Image,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect UI elements in image using YOLO
        
        Args:
            image: PIL Image object
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of detected elements with bounding boxes and confidence
        """
        if not self.icon_detector:
            await self._load_yolo_model()
        
        if progress_callback:
            await progress_callback(ValidationProgress(
                stage="detection",
                progress_percent=10.0,
                message="Running YOLO detection..."
            ))
        
        # Run detection in thread pool
        results = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self._run_yolo_detection,
            image
        )
        
        detections = self._extract_detections_from_results(results)
        
        if progress_callback:
            await progress_callback(ValidationProgress(
                stage="detection",
                progress_percent=30.0,
                message=f"Found {len(detections)} UI elements"
            ))
        
        logger.info(f"Detected {len(detections)} UI elements")
        return detections
    
    def _run_yolo_detection(self, image: Image.Image) -> Any:
        """Run YOLO detection synchronously with thread safety"""
        with self._yolo_lock:
            if self.use_nms:
                # Use YOLO's built-in NMS for cleaner detection
                result = self.icon_detector(
                    image, 
                    verbose=False, 
                    conf=self.detection_confidence,
                    iou=self.nms_threshold  # NMS IoU threshold
                )
            else:
                result = self.icon_detector(
                    image,
                    verbose=False,
                    conf=self.detection_confidence,
                )
        return result
    
    def _extract_detections_from_results(self, results: Any) -> List[Dict[str, Any]]:
        """Extract detection data from YOLO results"""
        detections = []
        
        for r in results:
            if r.boxes is not None:
                for i, box in enumerate(r.boxes):
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    confidence = float(box.conf.cpu().numpy())
                    
                    detection = {
                        "id": f"element_{i}",
                        "bbox": [int(coord) for coord in coords],  # [x1, y1, x2, y2]
                        "confidence": round(confidence, 3),
                        "area": int((coords[2] - coords[0]) * (coords[3] - coords[1]))
                    }
                    detections.append(detection)
        
        return detections
    
    async def generate_captions(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]],
        max_gpt_calls: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate captions for detected elements
        
        Args:
            image: PIL Image object
            detections: List of detection dictionaries
            max_gpt_calls: Maximum GPT-4V calls to make
            progress_callback: Optional callback for progress updates
            
        Returns:
            Updated detections with captions and methods
        """
        if not detections:
            return detections
        
        max_calls = max_gpt_calls or self.max_gpt_calls
        
        if progress_callback:
            await progress_callback(ValidationProgress(
                stage="captioning",
                progress_percent=40.0,
                message=f"Generating captions for {len(detections)} elements..."
            ))
        
        # Process captions with limited GPT calls
        gpt_calls_made = 0
        total_detections = len(detections)
        
        for i, detection in enumerate(detections):
            bbox = detection["bbox"]
            
            # Decide whether to use GPT-4V
            use_gpt = (
                self.openai_client and 
                gpt_calls_made < max_calls and
                self._should_use_gpt_for_element(detection, image.size)
            )
            
            if use_gpt:
                gpt_calls_made += 1
            
            # Generate caption
            caption_result = await self._generate_single_caption(
                image, bbox, use_gpt
            )
            
            detection.update(caption_result)
            
            # Progress update
            if progress_callback and i % 5 == 0:
                progress = 40.0 + (50.0 * (i + 1) / total_detections)
                await progress_callback(ValidationProgress(
                    stage="captioning",
                    progress_percent=progress,
                    message=f"Captioned {i + 1}/{total_detections} elements"
                ))
        
        logger.info(f"Generated captions: {gpt_calls_made} GPT-4V, {total_detections - gpt_calls_made} Florence-2")
        return detections
    
    def _should_use_gpt_for_element(self, detection: Dict[str, Any], image_size: Tuple[int, int]) -> bool:
        """Determine if element should use GPT-4V based on importance heuristics"""
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        
        x1, y1, x2, y2 = bbox
        width, height = image_size
        
        # Calculate element properties
        elem_width = x2 - x1
        elem_height = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        area = elem_width * elem_height
        
        # Prioritize GPT-4V for:
        # 1. High confidence detections
        # 2. Elements in interactive areas (right side, navigation)
        # 3. Medium to large elements
        # 4. Elements with good aspect ratios
        
        high_confidence = confidence > 0.3
        interactive_area = center_x > width * 0.7 or center_y < height * 0.2 or center_y > height * 0.8
        good_size = area > 1000 and elem_width > 20 and elem_height > 20
        
        return high_confidence and (interactive_area or good_size)
    
    async def _generate_single_caption(
        self,
        image: Image.Image,
        bbox: List[int],
        use_gpt: bool
    ) -> Dict[str, Any]:
        """Generate caption for a single element"""
        try:
            # Skip if bbox is too small
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                return {
                    "caption": "small_element",
                    "detection_method": "fallback"
                }
            
            # Try GPT-4V first if enabled
            if use_gpt and self.openai_client:
                gpt_result = await self._caption_with_gpt4v(image, bbox)
                if gpt_result:
                    return {
                        "caption": gpt_result,
                        "detection_method": "gpt4v"
                    }
            
            # Fallback to Florence-2
            florence_result = await self._caption_with_florence2(image, bbox)
            return {
                "caption": florence_result,
                "detection_method": "florence2"
            }
            
        except Exception as e:
            logger.warning(f"Caption generation failed for {bbox}: {e}")
            return {
                "caption": self._get_fallback_caption(bbox, image.size),
                "detection_method": "fallback"
            }
    
    async def _caption_with_gpt4v(self, image: Image.Image, bbox: List[int]) -> Optional[str]:
        """Generate caption using GPT-4V"""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(image, bbox)
            if cache_key in self.caption_cache:
                return self.caption_cache[cache_key]
            
            # Crop and prepare image
            cropped_image = self._crop_and_prepare_image(image, bbox)
            base64_image = self._encode_image_for_gpt4v(cropped_image)
            
            # Create context-aware prompt
            prompt = self._create_ui_caption_prompt(bbox, image.size)
            
            # Call GPT-4V
            response = await self.openai_client.chat.completions.create(
                model=settings.gpt_vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0
            )
            
            caption = response.choices[0].message.content.strip()
            
            # Cache result
            self.caption_cache[cache_key] = caption
            
            return caption
            
        except Exception as e:
            logger.warning(f"GPT-4V failed for {bbox}: {e}")
            return None
    
    async def _caption_with_florence2(self, image: Image.Image, bbox: List[int]) -> str:
        """Generate caption using Florence-2"""
        if not self.caption_model or not self.caption_processor:
            return self._get_fallback_caption(bbox, image.size)
        
        try:
            # Crop image
            cropped_image = self._crop_and_prepare_image(image, bbox, padding=5)
            
            # Run Florence-2 in thread pool
            caption = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self._run_florence2_inference,
                cropped_image
            )
            
            return caption if caption else self._get_fallback_caption(bbox, image.size)
            
        except Exception as e:
            logger.warning(f"Florence-2 failed for {bbox}: {e}")
            return self._get_fallback_caption(bbox, image.size)
    
    def _run_florence2_inference(self, image: Image.Image) -> str:
        """Run Florence-2 inference synchronously"""
        try:
            prompt = "<CAPTION>"
            inputs = self.caption_processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.caption_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=50,
                    num_beams=3,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=self.caption_processor.tokenizer.pad_token_id
                )
            
            generated_text = self.caption_processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]
            
            # Extract caption
            caption = generated_text.split(prompt)[-1].strip()
            caption = caption.replace("</s>", "").strip()
            
            return caption
            
        except Exception as e:
            logger.error(f"Florence-2 inference error: {e}")
            return ""
    
    def _crop_and_prepare_image(
        self, 
        image: Image.Image, 
        bbox: List[int], 
        padding: int = 15,
        min_size: int = 100
    ) -> Image.Image:
        """Crop and prepare image for captioning"""
        x1, y1, x2, y2 = bbox
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.width, x2 + padding)
        y2 = min(image.height, y2 + padding)
        
        # Crop image
        cropped = image.crop((x1, y1, x2, y2))
        
        # Ensure minimum size for GPT-4V
        if cropped.width < min_size or cropped.height < min_size:
            ratio = max(min_size / cropped.width, min_size / cropped.height)
            new_width = int(cropped.width * ratio)
            new_height = int(cropped.height * ratio)
            cropped = cropped.resize((new_width, new_height), Image.LANCZOS)
        
        return cropped
    
    def _encode_image_for_gpt4v(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string for GPT-4V"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _generate_cache_key(self, image: Image.Image, bbox: List[int]) -> str:
        """Generate cache key for GPT results"""
        x1, y1, x2, y2 = bbox
        cropped = image.crop((x1, y1, x2, y2))
        
        # Create hash from image data and bbox
        image_hash = hashlib.md5(cropped.tobytes()).hexdigest()[:8]
        bbox_hash = hashlib.md5(str(bbox).encode()).hexdigest()[:8]
        
        return f"{image_hash}_{bbox_hash}"
    
    def _create_ui_caption_prompt(self, bbox: List[int], image_size: Tuple[int, int]) -> str:
        """Create context-aware prompt for GPT-4V"""
        x1, y1, x2, y2 = bbox
        width, height = image_size
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Determine UI context based on position
        if center_x > width * 0.8:
            context = "This appears to be an interactive element from a mobile app's action panel."
        elif center_y < height * 0.15:
            context = "This appears to be a navigation or status element from the top of a mobile app."
        elif center_y > height * 0.85:
            context = "This appears to be a navigation element from the bottom of a mobile app."
        else:
            context = "This appears to be a content element from a mobile app interface."
        
        return f"""Analyze this UI element from a mobile app screenshot. {context}

Describe the element focusing on:
1. Element type (button, icon, text, etc.)
2. Visual appearance and colors
3. Any text, numbers, or symbols visible
4. Interactive state (active, inactive, highlighted, etc.)

Be specific about visual details. Keep description concise but informative."""
    
    def _get_fallback_caption(self, bbox: List[int], image_size: Tuple[int, int]) -> str:
        """Generate intelligent fallback caption based on position and size"""
        x1, y1, x2, y2 = bbox
        width, height = image_size
        
        elem_width = x2 - x1
        elem_height = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        area = elem_width * elem_height
        
        # Position-based classification
        if center_x > width * 0.8:  # Right side
            return "action_button" if area < 2000 else "sidebar_panel"
        elif center_y < height * 0.15:  # Top
            return "nav_element"
        elif center_y > height * 0.85:  # Bottom
            return "bottom_nav"
        elif area > width * height * 0.2:  # Large central
            return "main_content"
        elif elem_height < 50 and elem_width > width * 0.5:
            return "text_input"
        else:
            return "ui_element"
    
    def convert_to_ui_elements(self, detections: List[Dict[str, Any]]) -> List[UIElement]:
        """Convert detection dictionaries to UIElement schema objects"""
        ui_elements = []
        
        for detection in detections:
            bbox_coords = detection["bbox"]  # [x1, y1, x2, y2]
            
            # Convert to our BoundingBox format (x, y, width, height)
            bbox = BoundingBox(
                x=bbox_coords[0],
                y=bbox_coords[1],
                width=bbox_coords[2] - bbox_coords[0],
                height=bbox_coords[3] - bbox_coords[1]
            )
            
            ui_element = UIElement(
                id=detection["id"],
                bbox=bbox,
                caption=detection.get("caption", "unknown"),
                confidence=detection["confidence"],
                detection_method=detection.get("detection_method", "yolo"),
                clip_similarity=detection.get("clip_similarity")
            )
            
            ui_elements.append(ui_element)
        
        return ui_elements
    
    async def analyze_screenshot(
        self,
        image: Union[Image.Image, str, Path],
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[UIElement], Dict[str, Any]]:
        """
        Analyze a screenshot and return UI elements with metadata
        
        Args:
            image: PIL Image, file path, or image bytes
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (ui_elements, metadata)
        """
        start_time = time.time()
        
        # Load image if needed
        if not isinstance(image, Image.Image):
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            else:
                # Assume bytes
                image = Image.open(io.BytesIO(image)).convert("RGB")
        
        if progress_callback:
            await progress_callback(ValidationProgress(
                stage="initialization",
                progress_percent=5.0,
                message="Starting screenshot analysis..."
            ))
        
        # Detect UI elements
        detections = await self.detect_ui_elements(image, progress_callback)
        
        # Skip caption generation (not needed - saves GPT-4V API calls and Florence-2 inference)
        # CLIP will handle semantic matching directly from cropped images
        logger.info("Skipping caption generation to save resources - CLIP handles semantic matching")
        
        # Add minimal fallback captions for schema compatibility
        for i, detection in enumerate(detections):
            if "caption" not in detection:
                detection["caption"] = f"ui_element_{i+1}"
            if "detection_method" not in detection:
                detection["detection_method"] = "yolo_only"
        
        # Convert to schema objects
        ui_elements = self.convert_to_ui_elements(detections)
        
        # Prepare metadata
        processing_time = time.time() - start_time
        metadata = {
            "processing_time_seconds": processing_time,
            "total_elements": len(ui_elements),
            "image_size": image.size,
            "device": self.device,
            "models_used": {
                "yolo": self.models_loaded["yolo"],
                "florence2": self.models_loaded["florence2"],
                "gpt4v": self.models_loaded["openai"]
            }
        }
        
        if progress_callback:
            await progress_callback(ValidationProgress(
                stage="complete",
                progress_percent=100.0,
                message=f"Analysis complete! Found {len(ui_elements)} UI elements"
            ))
        
        logger.info(f"Screenshot analysis complete: {len(ui_elements)} elements in {processing_time:.2f}s")
        
        return ui_elements, metadata
    
    async def compare_screenshots(
        self,
        before_image: Union[Image.Image, str, Path],
        after_image: Union[Image.Image, str, Path],
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[UIElement], List[UIElement], List[DetectedChange]]:
        """
        Compare two screenshots and detect changes
        
        Args:
            before_image: Before state image
            after_image: After state image
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (before_elements, after_elements, changes)
        """
        if progress_callback:
            await progress_callback(ValidationProgress(
                stage="comparison",
                progress_percent=5.0,
                message="Starting screenshot comparison..."
            ))
        
        # Analyze both images
        before_elements, _ = await self.analyze_screenshot(before_image)
        
        if progress_callback:
            await progress_callback(ValidationProgress(
                stage="comparison",
                progress_percent=50.0,
                message="Analyzing after state..."
            ))
        
        after_elements, _ = await self.analyze_screenshot(after_image)
        
        # Detect changes
        changes = await self._detect_changes(before_elements, after_elements)
        
        if progress_callback:
            await progress_callback(ValidationProgress(
                stage="comparison",
                progress_percent=100.0,
                message=f"Comparison complete! Found {len(changes)} changes"
            ))
        
        return before_elements, after_elements, changes
    
    async def _detect_changes(
        self,
        before_elements: List[UIElement],
        after_elements: List[UIElement]
    ) -> List[DetectedChange]:
        """Detect changes between two sets of UI elements"""
        changes = []
        
        # Simple change detection based on position and caption similarity
        # In production, you might want more sophisticated matching algorithms
        
        matched_pairs = []
        unmatched_before = before_elements.copy()
        unmatched_after = after_elements.copy()
        
        # Find matching elements based on position overlap
        for before_elem in before_elements:
            best_match = None
            best_overlap = 0
            
            for after_elem in after_elements:
                overlap = self._calculate_element_overlap(before_elem, after_elem)
                if overlap > 0.5 and overlap > best_overlap:
                    best_overlap = overlap
                    best_match = after_elem
            
            if best_match:
                matched_pairs.append((before_elem, best_match))
                if before_elem in unmatched_before:
                    unmatched_before.remove(before_elem)
                if best_match in unmatched_after:
                    unmatched_after.remove(best_match)
        
        # Analyze matched pairs for changes
        for before_elem, after_elem in matched_pairs:
            if before_elem.caption != after_elem.caption:
                changes.append(DetectedChange(
                    element_id=before_elem.id,
                    change_type="state_changed",
                    before_element=before_elem,
                    after_element=after_elem,
                    confidence=0.8,
                    details=f"Element changed from '{before_elem.caption}' to '{after_elem.caption}'"
                ))
        
        # Handle removed elements
        for elem in unmatched_before:
            changes.append(DetectedChange(
                element_id=elem.id,
                change_type="removed",
                before_element=elem,
                after_element=None,
                confidence=0.9,
                details=f"Element '{elem.caption}' was removed"
            ))
        
        # Handle added elements
        for elem in unmatched_after:
            changes.append(DetectedChange(
                element_id=elem.id,
                change_type="added",
                before_element=None,
                after_element=elem,
                confidence=0.9,
                details=f"Element '{elem.caption}' was added"
            ))
        
        return changes
    
    def _calculate_element_overlap(self, elem1: UIElement, elem2: UIElement) -> float:
        """Calculate overlap between two UI elements"""
        bbox1 = elem1.bbox
        bbox2 = elem2.bbox
        
        # Convert to [x1, y1, x2, y2] format
        box1 = [bbox1.x, bbox1.y, bbox1.x + bbox1.width, bbox1.y + bbox1.height]
        box2 = [bbox2.x, bbox2.y, bbox2.x + bbox2.width, bbox2.y + bbox2.height]
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        # Clear cache
        self.caption_cache.clear()
        
        logger.info("OmniParser service cleaned up")


# Global service instance
omniparser_service = OmniParserService()

def get_omniparser_service() -> OmniParserService:
    """Get the global OmniParser service instance"""
    return omniparser_service