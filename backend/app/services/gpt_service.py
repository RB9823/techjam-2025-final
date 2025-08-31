"""
Production-ready GPT service for UI change validation.

This service provides async-compatible GPT-based validation of UI changes
with professional QA-style reasoning using OpenAI's GPT models.
"""

import asyncio
import json
import time
import io
import base64
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, AsyncGenerator, Union, Tuple
import openai
from openai import AsyncOpenAI
import structlog
from PIL import Image

from ..core.config import settings
from ..core.logging import get_logger
from ..schemas.validation import ValidationResponse, DetectedChange, DetectedException


logger = get_logger(__name__)


class GPTServiceError(Exception):
    """Base exception for GPT service errors"""
    pass


class GPTClientError(GPTServiceError):
    """Raised when GPT client fails to initialize"""
    pass


class GPTValidationError(GPTServiceError):
    """Raised when GPT validation fails"""
    pass


class GPTRateLimitError(GPTServiceError):
    """Raised when GPT API rate limit is exceeded"""
    pass


class GPTService:
    """
    Production-ready async GPT service for UI change validation.
    
    Features:
    - Async/await support for non-blocking OpenAI API calls
    - Professional QA-style reasoning and validation
    - Structured logging with contextual information
    - Configuration integration with app settings
    - Comprehensive error handling and recovery
    - Rate limiting and resource management
    - Health status monitoring
    - Integration with Pydantic schemas
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 vision_model: Optional[str] = None,
                 use_openrouter: Optional[bool] = None,
                 max_concurrent_requests: int = 5,
                 request_timeout: float = 60.0):
        """
        Initialize GPT service with OpenAI or OpenRouter support.
        
        Args:
            api_key: API key (defaults to config)
            model: GPT model for reasoning (defaults to config)
            vision_model: GPT vision model (defaults to config)
            use_openrouter: Whether to use OpenRouter instead of OpenAI
            max_concurrent_requests: Maximum concurrent API requests
            request_timeout: Request timeout in seconds
        """
        self.use_openrouter = use_openrouter if use_openrouter is not None else settings.use_openrouter
        
        # Set API key and models based on provider
        if self.use_openrouter:
            self.api_key = api_key or settings.openrouter_api_key
            self.model = model or settings.openrouter_model
            self.vision_model = vision_model or settings.openrouter_model  # Same model for vision
            self.base_url = "https://openrouter.ai/api/v1"
            self.headers = {
                "HTTP-Referer": "https://github.com/techjam-2025",
                "X-Title": "UI Validation API"
            }
            api_provider = "OpenRouter"
        else:
            self.api_key = api_key or settings.openai_api_key
            self.model = model or settings.gpt_model
            self.vision_model = vision_model or settings.gpt_vision_model
            self.base_url = None
            self.headers = {}
            api_provider = "OpenAI"
        
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        
        if not self.api_key:
            raise GPTClientError(
                f"{api_provider} API key not provided. Set {'OPENROUTER_API_KEY' if self.use_openrouter else 'OPENAI_API_KEY'} environment variable."
            )
        
        self._client: Optional[AsyncOpenAI] = None
        self._is_initialized = False
        self._initialization_lock = asyncio.Lock()
        self._request_semaphore: Optional[asyncio.Semaphore] = None
        self._session_counter = 0
        self._successful_requests = 0
        self._failed_requests = 0
        
        logger.info(
            "GPT service initialized",
            provider=api_provider,
            model=self.model,
            vision_model=self.vision_model,
            max_concurrent=max_concurrent_requests,
            timeout=request_timeout
        )
    
    async def initialize(self) -> None:
        """
        Initialize GPT client asynchronously.
        Thread-safe and idempotent.
        """
        if self._is_initialized:
            return
            
        async with self._initialization_lock:
            if self._is_initialized:
                return
                
            logger.info("Initializing GPT service", model=self.model)
            
            try:
                # Initialize client with provider-specific configuration
                client_kwargs = {
                    "api_key": self.api_key,
                    "timeout": self.request_timeout
                }
                
                if self.use_openrouter:
                    client_kwargs["base_url"] = self.base_url
                    client_kwargs["default_headers"] = self.headers
                
                self._client = AsyncOpenAI(**client_kwargs)
                self._request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
                self._is_initialized = True
                
                provider = "OpenRouter" if self.use_openrouter else "OpenAI"
                logger.info("GPT service initialized successfully", 
                           provider=provider, 
                           model=self.model)
                
            except Exception as e:
                error_msg = f"Failed to initialize GPT client: {str(e)}"
                logger.error(error_msg, error=str(e))
                raise GPTClientError(error_msg) from e
    
    async def cleanup(self) -> None:
        """Clean up resources and connections."""
        logger.info("Cleaning up GPT service resources")
        
        if self._client is not None:
            await self._client.close()
            self._client = None
            
        self._request_semaphore = None
        self._is_initialized = False
        
        logger.info(
            "GPT service cleanup completed",
            total_requests=self._successful_requests + self._failed_requests,
            successful_requests=self._successful_requests,
            failed_requests=self._failed_requests
        )
    
    def encode_image_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string for GPT Vision API.
        
        Args:
            image: PIL Image to encode
            
        Returns:
            Base64 encoded image string
        """
        buffer = io.BytesIO()
        # Convert to RGB if needed and save as PNG
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def parse_gpt_json(self, result_text: str, session_id: str, context: str = "") -> dict:
        """
        Parse JSON from GPT response, handling markdown code blocks and other formatting.
        
        Args:
            result_text: Raw GPT response text
            session_id: Session ID for logging
            context: Context for error messages (e.g., "heart detection", "selection")
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            json.JSONDecodeError: If no valid JSON can be extracted
        """
        import re
        
        # Try direct parsing first
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code blocks
        markdown_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
        if markdown_match:
            try:
                result_json = json.loads(markdown_match.group(1))
                logger.debug(f"Session {session_id}: Extracted JSON from markdown for {context}")
                return result_json
            except json.JSONDecodeError:
                pass
        
        # Try extracting any JSON object
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                result_json = json.loads(json_match.group())
                logger.debug(f"Session {session_id}: Extracted JSON object for {context}")
                return result_json
            except json.JSONDecodeError:
                pass
        
        # If we get here, no valid JSON was found
        logger.error(f"Session {session_id}: No valid JSON found in {context}. Response: '{result_text[:200]}...'")
        raise json.JSONDecodeError(f"No valid JSON found in GPT response", result_text, 0)
    
    def _generate_element_detection_prompt(self, element_type: str) -> str:
        """
        Generate element-specific detection prompts based on the UI element type.
        
        Args:
            element_type: The type of UI element to detect (e.g., "heart icon", "arrow", "button")
            
        Returns:
            Formatted detection prompt for GPT Vision
        """
        element_lower = element_type.lower()
        
        # Heart/Like icon detection (most detailed since it was the original)
        if "heart" in element_lower or "like" in element_lower:
            return """You are analyzing a cropped UI element from a social media app to determine if it contains a heart or like icon.

TASK: Determine if this image shows a heart/like icon that users can interact with.

HEART/LIKE ICON EXAMPLES TO LOOK FOR:
- Heart shapes: ♥, ♡, 💖, 🤍, ❤️, 💕 in any style (filled, outline, solid, hollow)
- TikTok like buttons: often white/gray outlines that turn red when liked
- Instagram heart icons: outline or filled hearts
- Social media like buttons: hearts of ANY size within interaction areas
- Heart symbols within buttons or UI groups
- Hearts that may be tiny (even 10-20 pixels)
- Heart-shaped touch targets or clickable areas

VISUAL PATTERNS TO RECOGNIZE:
- Two rounded bumps at the top, pointed bottom (classic heart shape)
- Symmetrical curved shape tapering to a point
- Hearts in ANY color: red, white, black, gray, pink, purple, etc.
- Hearts as outlines, filled shapes, or stylized versions
- Hearts that are part of social interaction elements

CONTEXT CLUES:
- Located in bottom interaction bars (common in TikTok, Instagram)
- Near other social buttons (share, comment, save)
- Part of engagement/reaction UI elements
- May have numbers nearby indicating like counts

BE VERY LIBERAL - LOOK CAREFULLY FOR:
- Subtle heart outlines that might be hard to see
- Small hearts within larger UI elements  
- Hearts that are partially visible or cropped
- Stylized or abstract heart representations
- Hearts that blend into the background

NOT heart icons:
- Pure text without heart symbols
- Numbers or counts alone
- Other geometric shapes (circles, squares, triangles)
- Random UI chrome or decorative elements"""

        # Arrow detection
        elif "arrow" in element_lower:
            return f"""You are analyzing a cropped UI element to determine if it contains an arrow or directional indicator.

TASK: Determine if this image shows an arrow or directional element that users can interact with.

ARROW/DIRECTIONAL EXAMPLES TO LOOK FOR:
- Arrow shapes: →, ←, ↑, ↓, ⇒, ⇐, ⇑, ⇓ in any style
- Triangular directional indicators: ▶, ◀, ▲, ▼, ►, ◄
- Chevron arrows: >, <, ^, v, », «
- Navigation arrows in carousels, menus, dropdowns
- Back/forward arrows in navigation
- Expand/collapse arrows
- Sort arrows (up/down indicators)

VISUAL PATTERNS TO RECOGNIZE:
- Triangular or pointed shapes indicating direction
- Linear shapes with directional heads (→)
- Chevron patterns (> or <)
- Directional symbols within buttons or clickable areas
- Navigation indicators in UI controls

CONTEXT CLUES:
- Located in navigation areas, toolbars, menus
- Part of carousel controls, pagination
- Within dropdown menus or expandable sections
- Near scrollable content areas
- Associated with "next", "previous", "more" functionality

BE VERY LIBERAL - LOOK CAREFULLY FOR:
- Subtle arrow outlines or fills
- Small directional indicators within larger elements
- Stylized or abstract arrow representations
- Arrows that are part of composite UI elements
- Directional hints in interactive areas

NOT arrows:
- Pure text without directional symbols
- Random geometric shapes without clear direction
- Decorative elements that don't indicate direction
- Non-interactive visual elements"""

        # Button detection  
        elif "button" in element_lower:
            return f"""You are analyzing a cropped UI element to determine if it contains a clickable button.

TASK: Determine if this image shows a button or clickable interface element.

BUTTON EXAMPLES TO LOOK FOR:
- Rectangular clickable areas with text labels
- Rounded rectangle buttons with clear boundaries
- Icon buttons (circular or square clickable areas)
- Toggle buttons, switches, checkboxes
- Submit buttons, action buttons (Save, Cancel, OK)
- Navigation buttons, tab buttons
- Floating action buttons (FABs)

VISUAL PATTERNS TO RECOGNIZE:
- Clear boundaries (borders, background colors, shadows)
- Text labels indicating actions (Submit, Save, Click, etc.)
- Consistent button styling with the app's design
- Visual affordances suggesting interactivity
- Hover states, pressed states, or visual feedback

CONTEXT CLUES:
- Located in forms, dialogs, navigation areas
- Contains action-oriented text or icons
- Part of user interface control groups
- Positioned for easy user interaction
- Shows visual states (enabled/disabled)

BE VERY LIBERAL - LOOK CAREFULLY FOR:
- Subtle button borders or backgrounds
- Text-only buttons with minimal styling
- Icon-only buttons without text
- Buttons that blend with the interface design
- Custom-styled buttons that don't look traditional

NOT buttons:
- Pure decorative text or images
- Non-interactive display elements
- Static labels or headings
- Background design elements"""

        # Text field/input detection
        elif "text" in element_lower or "input" in element_lower or "field" in element_lower:
            return f"""You are analyzing a cropped UI element to determine if it contains a text input field or text area.

TASK: Determine if this image shows a text input field, text area, or editable text element.

TEXT INPUT EXAMPLES TO LOOK FOR:
- Single-line text input fields with borders
- Multi-line text areas (textboxes)
- Search input fields with search icons
- Form input fields (email, password, name, etc.)
- Editable text areas with cursors or focus indicators
- Input fields with placeholder text
- Text fields with labels or hints

VISUAL PATTERNS TO RECOGNIZE:
- Rectangular areas with borders or background colors
- Placeholder text in gray or muted colors
- Text cursors or focus indicators
- Clear input boundaries and typing areas
- Labels or hints associated with input areas

CONTEXT CLUES:
- Located in forms, registration pages, search areas
- Contains placeholder text or field labels
- Part of data entry interfaces
- Shows text cursor or active input states
- Associated with keyboards or input methods

BE VERY LIBERAL - LOOK CAREFULLY FOR:
- Minimal-style input fields with subtle borders
- Text areas with typing cursors
- Search boxes with or without search icons
- Input fields that blend with the interface
- Editable text elements in forms

NOT text inputs:
- Static text displays or labels
- Non-editable text content
- Decorative text elements
- Read-only text information"""

        # Icon detection (generic)
        elif "icon" in element_lower:
            return f"""You are analyzing a cropped UI element to determine if it contains a UI icon.

TASK: Determine if this image shows an icon or symbolic interface element.

ICON EXAMPLES TO LOOK FOR:
- Symbolic representations of actions or concepts
- Small graphical elements representing functions
- Interface symbols (home, settings, menu, etc.)
- Social media icons (share, bookmark, profile, etc.)
- System icons (close, minimize, maximize, etc.)
- Status icons (notifications, warnings, success, etc.)
- Navigation icons (back, forward, up, down, etc.)

VISUAL PATTERNS TO RECOGNIZE:
- Small, simplified graphical symbols
- Consistent styling with the app's icon design
- Clear symbolic meaning or representation
- Standard UI iconography patterns
- Scalable symbolic elements

CONTEXT CLUES:
- Located in toolbars, navigation areas, action bars
- Part of interface control systems
- Represents common UI functions or concepts
- Positioned for quick recognition and interaction
- Shows consistent design with other interface icons

BE VERY LIBERAL - LOOK CAREFULLY FOR:
- Simple line art or filled symbolic shapes
- Minimalist icon designs
- Icons within buttons or interactive areas
- Custom-styled icons specific to the app
- Icons that serve functional purposes

NOT icons:
- Complex images or photographs
- Decorative artwork without functional meaning
- Text-based elements
- Random graphical elements without clear purpose"""

        # Generic/fallback detection for any other element type
        else:
            return f"""You are analyzing a cropped UI element to determine if it contains a "{element_type}" element.

TASK: Determine if this image shows a "{element_type}" that users can interact with or that serves the specified function.

WHAT TO LOOK FOR:
- Visual elements that match the description "{element_type}"
- Interactive or functional elements related to "{element_type}"
- UI components that serve the purpose of "{element_type}"
- Design patterns commonly associated with "{element_type}"
- Any visual representation that could be identified as "{element_type}"

ANALYSIS APPROACH:
- Consider the context and typical appearance of "{element_type}" elements
- Look for visual cues, shapes, colors, or text that suggest "{element_type}"
- Consider the functional purpose that "{element_type}" would serve in a UI
- Be flexible with visual variations and styling approaches
- Account for different design systems and visual treatments

BE VERY LIBERAL - LOOK CAREFULLY FOR:
- Any visual element that could reasonably be called "{element_type}"
- Subtle or minimalist representations
- Custom-styled versions that don't look traditional
- Elements that serve the functional purpose of "{element_type}"
- Partial or cropped views of "{element_type}" elements

RESPOND THOUGHTFULLY:
- Consider whether this element serves the purpose of "{element_type}"
- Evaluate visual and functional characteristics
- Be generous in interpretation while remaining accurate"""

    async def detect_element_in_crop(self, crop_image: Image.Image, crop_id: int, element_type: str, session_id: str) -> Dict[str, Any]:
        """
        Detect if a single cropped image contains a specified UI element using GPT Vision.
        
        Args:
            crop_image: Cropped PIL image
            crop_id: Identifier for this crop
            element_type: Type of UI element to detect (e.g., "heart icon", "arrow", "button")
            session_id: Session identifier for logging
            
        Returns:
            Dictionary with detection result: {is_element, confidence, reasoning}
        """
        self._ensure_initialized()
        
        try:
            # Validate crop image
            if crop_image is None:
                raise ValueError("Crop image is None")
            
            crop_size = crop_image.size
            if crop_size[0] <= 0 or crop_size[1] <= 0:
                raise ValueError(f"Invalid crop size: {crop_size}")
            
            # Skip very small crops that are likely noise
            if crop_size[0] < 5 or crop_size[1] < 5:
                logger.debug(f"Session {session_id}: Skipping very small crop {crop_id}: {crop_size}")
                return {
                    "is_element": False,
                    "confidence": 0.0,
                    "reasoning": f"Crop too small: {crop_size}",
                    "element_type": element_type
                }
            
            # Convert image to base64
            image_b64 = self.encode_image_base64(crop_image)
            
            # Generate element-specific detection prompt
            element_prompt = self._generate_element_detection_prompt(element_type)
            
            # Create full prompt with consistent response format
            prompt = f"""{element_prompt}

RESPONSE FORMAT (JSON only):
{{
    "is_element": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Describe exactly what you see - shapes, colors, context, and why it matches or doesn't match '{element_type}'"
}}

Look very carefully and respond in JSON format only."""

            # Make GPT Vision API call
            async with self._request_semaphore:
                start_time = time.time()
                
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=200,
                    temperature=0.1
                )
                
                request_time = time.time() - start_time
                self._successful_requests += 1
                
                # Parse response
                result_text = response.choices[0].message.content
                logger.debug(f"Session {session_id}: GPT response for crop {crop_id}: {result_text}")
                
                result_json = self.parse_gpt_json(result_text, session_id, f"{element_type} detection crop {crop_id}")
                
                is_element = result_json.get("is_element", False)
                confidence = float(result_json.get("confidence", 0.0))
                reasoning = result_json.get("reasoning", "No reasoning provided")
                
                logger.info(f"Session {session_id}: Crop {crop_id} - {element_type}: {is_element} (confidence: {confidence:.3f})")
                
                return {
                    "is_element": is_element,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "element_type": element_type,
                    "request_time": request_time
                }
                
        except Exception as e:
            self._failed_requests += 1
            error_msg = str(e)
            
            if "tile cannot extend outside image" in error_msg:
                logger.warning(f"Session {session_id}: Crop {crop_id} has invalid boundaries - skipping")
            elif "Expecting value: line 1 column 1" in error_msg:
                logger.warning(f"Session {session_id}: GPT returned invalid JSON for crop {crop_id} - likely empty response")
            else:
                logger.error(f"Session {session_id}: Failed to detect {element_type} in crop {crop_id}: {e}")
            
            return {
                "is_element": False,
                "confidence": 0.0,
                "reasoning": f"Detection failed: {str(e)[:100]}...",
                "element_type": element_type,
                "error": str(e)
            }
    
    async def select_best_element(self, element_candidates: List[Dict[str, Any]], element_type: str, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Select the best element when multiple are found using GPT Vision.
        
        Args:
            element_candidates: List of element detection results with images
            element_type: Type of element being selected
            session_id: Session identifier for logging
            
        Returns:
            Dictionary with selected element info or None if no valid selection
        """
        self._ensure_initialized()
        
        if len(element_candidates) == 0:
            return None
        elif len(element_candidates) == 1:
            # Only one element, use it directly
            return {
                **element_candidates[0],
                "selection_reasoning": f"Only {element_type} candidate found",
                "total_candidates": 1
            }
        
        # Multiple elements - use GPT to select the best one
        logger.info(f"Session {session_id}: Selecting best {element_type} from {len(element_candidates)} candidates")
        
        try:
            # Prepare images for GPT (limit to top 5 to avoid token limits)
            limited_candidates = element_candidates[:5]
            element_images = []
            candidate_info = []
            
            for i, candidate in enumerate(limited_candidates):
                image_b64 = self.encode_image_base64(candidate["crop_image"])
                element_images.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                })
                candidate_info.append(f"Image {i}: Crop {candidate['crop_id']} (confidence: {candidate['confidence']:.3f})")
            
            # Create selection prompt
            prompt = f"""You are selecting the best "{element_type}" element from multiple candidates.

CANDIDATES FOUND:
{chr(10).join(candidate_info)}

TASK: Select which "{element_type}" element is the main interactive or most important element that users would interact with.

SELECTION CRITERIA:
- Size: Larger elements are usually more important
- Position: Main elements are often in prominent UI locations
- Visual prominence: More visible and prominent elements
- UI context: Consider typical interface design patterns
- Functionality: Elements that clearly serve the intended "{element_type}" function

RESPONSE FORMAT (JSON only):
{{
    "selected_image": 0-{min(len(limited_candidates)-1, 4)},
    "confidence": 0.0-1.0,
    "reasoning": "Why this {element_type} element is the best choice"
}}

Select the best {element_type} element and respond in JSON format only."""

            # Prepare messages
            content = [{"type": "text", "text": prompt}] + element_images
            
            # Make selection call
            async with self._request_semaphore:
                start_time = time.time()
                
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=300,
                    temperature=0.1
                )
                
                request_time = time.time() - start_time
                self._successful_requests += 1
                
                # Parse response
                result_text = response.choices[0].message.content
                result_json = self.parse_gpt_json(result_text, session_id, f"{element_type} selection")
                
                selected_idx = int(result_json.get("selected_image", 0))
                confidence = float(result_json.get("confidence", 0.5))
                reasoning = result_json.get("reasoning", "No reasoning provided")
                
                # Get the selected element
                if 0 <= selected_idx < len(limited_candidates):
                    selected_element = limited_candidates[selected_idx]
                    
                    logger.info(f"Session {session_id}: Selected crop {selected_element['crop_id']} from {len(element_candidates)} candidates")
                    
                    return {
                        **selected_element,
                        "selection_confidence": confidence,
                        "selection_reasoning": reasoning,
                        "total_candidates": len(element_candidates),
                        "request_time": request_time
                    }
                else:
                    raise ValueError(f"Invalid selection index: {selected_idx}")
                    
        except Exception as e:
            self._failed_requests += 1
            logger.error(f"Session {session_id}: {element_type} selection failed: {e}")
            # Fallback to highest confidence element
            best_element = max(element_candidates, key=lambda x: x.get('confidence', 0))
            return {
                **best_element,
                "selection_confidence": best_element.get('confidence', 0.5),
                "selection_reasoning": f"Selection failed, using highest confidence {element_type}: {str(e)}",
                "total_candidates": len(element_candidates),
                "error": str(e)
            }
    
    async def validate_element_change(self, 
                                     before_element_data: Dict[str, Any],
                                     after_element_data: Dict[str, Any],
                                     expected_change: str,
                                     session_id: str) -> Dict[str, Any]:
        """
        Final comparison to validate if the element change occurred as expected using GPT Vision.
        
        Args:
            before_element_data: Element data from before image (includes crop_image)
            after_element_data: Element data from after image (includes crop_image)
            expected_change: Description of expected change
            session_id: Session identifier for logging
            
        Returns:
            Dictionary with validation result: {is_valid, confidence, reasoning, details}
        """
        self._ensure_initialized()
        
        element_type = before_element_data.get("element_type", "element")
        logger.info(f"Session {session_id}: Validating {element_type} change with GPT Vision")
        
        try:
            # Encode both element images
            before_b64 = self.encode_image_base64(before_element_data["crop_image"])
            after_b64 = self.encode_image_base64(after_element_data["crop_image"])
            
            # Create comparison prompt
            prompt = f"""You are validating a UI change between before/after states of a {element_type}.

EXPECTED CHANGE: {expected_change}

TASK: Compare the two {element_type} elements and determine if the expected change occurred.

ANALYSIS CRITERIA:
1. Visual Similarity: Do these appear to be the same UI element?
2. State Changes: What visual changes occurred (color, fill, style, state, etc.)?
3. Expected vs Actual: Does the change match what was expected?

BEFORE STATE: [First image shows the before state]
AFTER STATE: [Second image shows the after state]

Consider common {element_type} changes:
- Color changes (different colors, states, themes)
- Fill changes (outline → filled, empty → solid)
- Visual state changes (inactive → active, different states)
- Style changes (different visual emphasis or appearance)
- Shape or size changes
- Content changes (text, icons, symbols)

RESPONSE FORMAT (JSON only):
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of what changed and why it matches/doesn't match expectation",
    "visual_changes_detected": ["list", "of", "changes", "observed"],
    "same_ui_element": true/false,
    "change_category": "color_change" | "fill_change" | "style_change" | "no_change" | "different_element"
}}

Analyze both images and respond in JSON format only."""

            # Make comparison call
            async with self._request_semaphore:
                start_time = time.time()
                
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{before_b64}"}
                                },
                                {
                                    "type": "image_url", 
                                    "image_url": {"url": f"data:image/png;base64,{after_b64}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
                
                request_time = time.time() - start_time
                self._successful_requests += 1
                
                # Parse response
                result_text = response.choices[0].message.content
                result_json = self.parse_gpt_json(result_text, session_id, f"{element_type} comparison")
                
                is_valid = result_json.get("is_valid", False)
                confidence = float(result_json.get("confidence", 0.0))
                reasoning = result_json.get("reasoning", "No reasoning provided")
                visual_changes = result_json.get("visual_changes_detected", [])
                same_element = result_json.get("same_ui_element", True)
                change_category = result_json.get("change_category", "unknown")
                
                validation_result = {
                    "is_valid": is_valid,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "visual_changes": visual_changes,
                    "same_ui_element": same_element,
                    "change_category": change_category,
                    "before_confidence": before_element_data.get("confidence", 0.0),
                    "after_confidence": after_element_data.get("confidence", 0.0),
                    "before_crop_id": before_element_data.get("crop_id", -1),
                    "after_crop_id": after_element_data.get("crop_id", -1),
                    "request_time": request_time,
                    "element_type": element_type
                }
                
                logger.info(f"Session {session_id}: Validation complete - Valid: {is_valid} (confidence: {confidence:.3f})")
                return validation_result
                
        except Exception as e:
            self._failed_requests += 1
            logger.error(f"Session {session_id}: {element_type} change validation failed: {e}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "reasoning": f"Validation failed due to error: {str(e)}",
                "error": str(e),
                "element_type": element_type
            }
    
    async def parse_qa_prompt(self, qa_prompt: str) -> Dict[str, Optional[str]]:
        """
        Parse QA prompt to extract region of interest and expected change.
        
        Args:
            qa_prompt: The QA prompt to parse (e.g., "Does heart turn red when liked?")
            
        Returns:
            Dict with 'region_of_interest' and 'expected_change' keys
            
        Example:
            Input: "Does heart turn red when liked?"
            Output: {
                "region_of_interest": "heart icon",
                "expected_change": "turns red when liked"
            }
        """
        self._ensure_initialized()
        
        try:
            prompt = f"""You are a QA prompt parser. Extract the UI element and expected change from QA prompts.

TASK: Parse this QA prompt to identify:
1. REGION OF INTEREST: The specific UI element being tested (e.g., "heart icon", "submit button", "login form")
2. EXPECTED CHANGE: What change is expected to happen (e.g., "turns red", "becomes disabled", "shows error message")

QA PROMPT: "{qa_prompt}"

INSTRUCTIONS:
- Be specific about the UI element type (e.g., "heart icon" not just "heart")  
- Focus on the actual change being tested
- If unclear, make reasonable assumptions based on common UI patterns
- If no specific element can be identified, use null for region_of_interest
- If no change is specified, use null for expected_change

Respond in JSON format only:
{{
    "region_of_interest": "specific UI element type or null",
    "expected_change": "description of expected change or null"
}}

Examples:
- "Does heart turn red when liked?" → {{"region_of_interest": "heart icon", "expected_change": "turns red when liked"}}
- "Is submit button disabled when form is invalid?" → {{"region_of_interest": "submit button", "expected_change": "becomes disabled when form is invalid"}}
- "Does login show error for wrong password?" → {{"region_of_interest": "login form", "expected_change": "shows error message for wrong password"}}

Parse the prompt and respond with JSON only:"""

            # Prepare messages for GPT
            messages = [
                {"role": "system", "content": "You are a QA prompt parsing specialist. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            # Make GPT request
            session_id = f"qa_parsing_{int(time.time())}"
            gpt_response = await self._make_gpt_request(
                messages=messages,
                session_id=session_id,
                temperature=0.1,  # Low temperature for consistent parsing
                max_tokens=200
            )
            
            # Get the already parsed response
            result = gpt_response.get("parsed_response", {})
            
            # Validate and clean results
            parsed_result = {
                "region_of_interest": result.get("region_of_interest"),
                "expected_change": result.get("expected_change")
            }
            
            # Convert null strings to None
            if parsed_result["region_of_interest"] == "null":
                parsed_result["region_of_interest"] = None
            if parsed_result["expected_change"] == "null":
                parsed_result["expected_change"] = None
                
            logger.info("QA prompt parsed successfully",
                       qa_prompt=qa_prompt,
                       region_of_interest=parsed_result["region_of_interest"],
                       expected_change=parsed_result["expected_change"])
            
            return parsed_result
            
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse GPT JSON response for QA prompt", 
                         qa_prompt=qa_prompt, error=str(e))
            # Fallback: return None values
            return {"region_of_interest": None, "expected_change": None}
            
        except Exception as e:
            logger.error("QA prompt parsing failed", qa_prompt=qa_prompt, error=str(e))
            # Fallback: return None values  
            return {"region_of_interest": None, "expected_change": None}
    
    def _ensure_initialized(self) -> None:
        """Ensure service is initialized, raise error if not."""
        if not self._is_initialized:
            raise GPTServiceError("GPT service not initialized. Call initialize() first.")
    
    def _create_validation_prompt(self, analysis_summary: Dict[str, Any]) -> str:
        """
        Create structured prompt for GPT validation.
        
        Args:
            analysis_summary: Complete analysis from UI validation process
            
        Returns:
            Formatted prompt string for GPT
        """
        prompt = f"""You are a professional QA engineer analyzing UI changes between before/after screenshots.

EXPECTED CHANGE: {analysis_summary['expected_change']}

ANALYSIS DATA:
=============

Image Comparison:
- Before: {analysis_summary.get('image_comparison', {}).get('before', {}).get('elements_detected', 0)} UI elements detected
- After: {analysis_summary.get('image_comparison', {}).get('after', {}).get('elements_detected', 0)} UI elements detected

Before Screenshot Elements:
{json.dumps(analysis_summary.get('image_comparison', {}).get('before', {}).get('elements', []), indent=2)}

After Screenshot Elements:
{json.dumps(analysis_summary.get('image_comparison', {}).get('after', {}).get('elements', []), indent=2)}

Detected Changes:
{json.dumps(analysis_summary.get('detected_changes', []), indent=2)}

Detected Exceptions:
{json.dumps(analysis_summary.get('detected_exceptions', []), indent=2)}

Summary Statistics:
- Total changes detected: {analysis_summary.get('summary_stats', {}).get('total_changes', 0)}
- Expected exceptions (normal): {analysis_summary.get('summary_stats', {}).get('expected_exceptions', 0)}
- Unexpected exceptions (issues): {analysis_summary.get('summary_stats', {}).get('unexpected_exceptions', 0)}

VALIDATION TASK:
===============
Analyze whether the EXPECTED CHANGE occurred based on the UI analysis data above.

Provide your response in the following JSON format:
{{
    "validation": "YES" or "NO",
    "reasoning": "Detailed professional QA reasoning explaining your decision",
    "confidence": float between 0.0 and 1.0,
    "recommendation": "Optional recommendation for next steps",
    "analysis_details": {{
        "change_detection_summary": "Brief summary of what changes were detected",
        "exception_analysis": "Analysis of any exceptions found",
        "root_cause": "Likely root cause if validation failed",
        "confidence_factors": "What factors influenced your confidence level"
    }}
}}

PROFESSIONAL QA REASONING GUIDELINES:
- Be specific about what elements changed or didn't change
- Reference exact element positions and captions when relevant
- Classify exceptions as expected (normal business prompts) vs unexpected (system issues)
- Provide confidence levels based on evidence quality
- Include actionable recommendations for failed validations
- Consider timing issues, rendering problems, and interaction failures
- Use precise technical language appropriate for QA reporting

Your response must be valid JSON only, no additional text."""

        return prompt
    
    async def _make_gpt_request(self, 
                               messages: List[Dict[str, str]], 
                               session_id: str,
                               temperature: float = 0.1,
                               max_tokens: int = 1500) -> Dict[str, Any]:
        """
        Make async GPT API request with rate limiting and error handling.
        
        Args:
            messages: Messages for chat completion
            session_id: Session identifier for logging
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Parsed JSON response from GPT
            
        Raises:
            GPTRateLimitError: If rate limit exceeded
            GPTValidationError: If API request fails
        """
        self._ensure_initialized()
        
        async with self._request_semaphore:
            try:
                logger.info(
                    "Making GPT API request",
                    session_id=session_id,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                start_time = time.time()
                
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}
                )
                
                request_time = time.time() - start_time
                self._successful_requests += 1
                
                result_text = response.choices[0].message.content
                result_json = json.loads(result_text)
                
                logger.info(
                    "GPT API request completed",
                    session_id=session_id,
                    request_time=f"{request_time:.2f}s",
                    response_tokens=len(result_text.split()),
                    total_requests=self._successful_requests
                )
                
                return {
                    "parsed_response": result_json,
                    "raw_response": result_text,
                    "request_time": request_time,
                    "model_used": self.model
                }
                
            except openai.RateLimitError as e:
                self._failed_requests += 1
                error_msg = f"GPT API rate limit exceeded: {str(e)}"
                logger.error(
                    error_msg,
                    session_id=session_id,
                    error=str(e),
                    failed_requests=self._failed_requests
                )
                raise GPTRateLimitError(error_msg) from e
                
            except json.JSONDecodeError as e:
                self._failed_requests += 1
                error_msg = f"GPT response parsing failed: {str(e)}"
                raw_response = response.choices[0].message.content if 'response' in locals() else "No response"
                
                logger.error(
                    error_msg,
                    session_id=session_id,
                    error=str(e),
                    raw_response=raw_response[:500],
                    failed_requests=self._failed_requests
                )
                
                # Return structured error response
                return {
                    "parsed_response": {
                        "validation": "NO",
                        "reasoning": f"GPT response parsing failed: {str(e)}. This may indicate a formatting issue with the AI response.",
                        "confidence": 0.0,
                        "recommendation": "Retry the validation request or check the prompt formatting.",
                        "analysis_details": {
                            "error": "JSON parsing failed",
                            "raw_response_preview": raw_response[:200]
                        }
                    },
                    "raw_response": raw_response,
                    "request_time": time.time() - start_time if 'start_time' in locals() else 0,
                    "model_used": self.model,
                    "error": "json_parsing_failed"
                }
                
            except Exception as e:
                self._failed_requests += 1
                error_msg = f"GPT API request failed: {str(e)}"
                logger.error(
                    error_msg,
                    session_id=session_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    failed_requests=self._failed_requests
                )
                
                # Return structured error response
                return {
                    "parsed_response": {
                        "validation": "NO",
                        "reasoning": f"GPT validation failed with error: {str(e)}. This may be due to network issues or API problems.",
                        "confidence": 0.0,
                        "recommendation": "Check API key, network connection, and try again. If problem persists, contact support.",
                        "analysis_details": {
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    },
                    "raw_response": None,
                    "request_time": 0,
                    "model_used": self.model,
                    "error": "api_request_failed"
                }
    
    async def validate_ui_change_async(self, 
                                      analysis_summary: Dict[str, Any],
                                      custom_instructions: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate UI change with GPT reasoning asynchronously.
        
        Args:
            analysis_summary: Analysis summary from UI validation process
            custom_instructions: Optional custom validation instructions
            
        Returns:
            Dictionary with detailed validation result and reasoning
        """
        self._ensure_initialized()
        
        # Generate session ID for tracking
        self._session_counter += 1
        session_id = f"gpt_validation_{int(time.time() * 1000) % 100000}_{self._session_counter}"
        
        expected_change = analysis_summary.get('expected_change', 'No expected change specified')
        
        logger.info(
            "Starting GPT validation",
            session_id=session_id,
            expected_change=expected_change[:100] + "..." if len(expected_change) > 100 else expected_change,
            total_changes=analysis_summary.get('summary_stats', {}).get('total_changes', 0),
            total_exceptions=len(analysis_summary.get('detected_exceptions', []))
        )
        
        try:
            # Create validation prompt
            prompt = self._create_validation_prompt(analysis_summary)
            
            # Add custom instructions if provided
            system_content = """You are a professional QA engineer with expertise in UI testing and automated validation. 
Provide detailed, technical analysis in valid JSON format."""
            
            if custom_instructions:
                system_content += f"\n\nAdditional Instructions: {custom_instructions}"
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            
            # Make GPT API request
            gpt_response = await self._make_gpt_request(messages, session_id)
            result_json = gpt_response["parsed_response"]
            
            # Structure the validation result
            validation_result = {
                "is_valid": result_json.get("validation", "NO").upper() == "YES",
                "reasoning": result_json.get("reasoning", "No reasoning provided"),
                "confidence": float(result_json.get("confidence", 0.5)),
                "recommendation": result_json.get("recommendation"),
                "analysis_details": result_json.get("analysis_details", {}),
                "raw_response": gpt_response["raw_response"],
                "model_used": gpt_response["model_used"],
                "request_time": gpt_response["request_time"],
                "session_id": session_id,
                "has_error": "error" in gpt_response
            }
            
            logger.info(
                "GPT validation completed",
                session_id=session_id,
                is_valid=validation_result["is_valid"],
                confidence=validation_result["confidence"],
                request_time=f"{validation_result['request_time']:.2f}s",
                has_error=validation_result["has_error"]
            )
            
            return validation_result
            
        except Exception as e:
            error_msg = f"GPT validation failed for session {session_id}: {str(e)}"
            logger.error(
                error_msg,
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__,
                expected_change=expected_change
            )
            
            # Return error response in expected format
            return {
                "is_valid": False,
                "reasoning": f"GPT validation encountered an error: {str(e)}",
                "confidence": 0.0,
                "recommendation": "Retry the validation or check service status",
                "analysis_details": {"error": str(e), "error_type": type(e).__name__},
                "raw_response": None,
                "model_used": self.model,
                "request_time": 0.0,
                "session_id": session_id,
                "has_error": True
            }
    
    async def create_detailed_report_async(self, 
                                         validation_result: Dict[str, Any], 
                                         analysis_summary: Dict[str, Any]) -> str:
        """
        Create a detailed professional QA report asynchronously.
        
        Args:
            validation_result: Result from validate_ui_change_async
            analysis_summary: Original analysis summary
            
        Returns:
            Formatted QA report string
        """
        logger.debug(
            "Creating detailed validation report",
            session_id=validation_result.get("session_id", "unknown"),
            is_valid=validation_result["is_valid"]
        )
        
        # Build report asynchronously to avoid blocking
        await asyncio.sleep(0)  # Yield control
        
        report_parts = []
        
        # Header
        report_parts.append("UI CHANGE VALIDATION REPORT")
        report_parts.append("=" * 50)
        report_parts.append("")
        
        # Validation result
        status = "✅ PASS" if validation_result["is_valid"] else "❌ FAIL"
        report_parts.append(f"VALIDATION RESULT: {status}")
        report_parts.append(f"CONFIDENCE LEVEL: {validation_result['confidence']:.1%}")
        report_parts.append(f"MODEL USED: {validation_result.get('model_used', 'unknown')}")
        report_parts.append(f"REQUEST TIME: {validation_result.get('request_time', 0):.2f}s")
        report_parts.append("")
        
        # Expected change
        report_parts.append("EXPECTED CHANGE:")
        report_parts.append(analysis_summary.get('expected_change', 'Not specified'))
        report_parts.append("")
        
        # Professional QA analysis
        report_parts.append("PROFESSIONAL QA ANALYSIS:")
        report_parts.append(validation_result["reasoning"])
        report_parts.append("")
        
        # Technical details
        report_parts.append("TECHNICAL DETAILS:")
        report_parts.append("=" * 30)
        report_parts.append("")
        
        image_comparison = analysis_summary.get('image_comparison', {})
        before_elements = image_comparison.get('before', {}).get('elements_detected', 0)
        after_elements = image_comparison.get('after', {}).get('elements_detected', 0)
        
        report_parts.append(f"Image Analysis:")
        report_parts.append(f"- Before: {before_elements} elements detected")
        report_parts.append(f"- After: {after_elements} elements detected")
        report_parts.append("")
        
        # Change detection
        detected_changes = analysis_summary.get('detected_changes', [])
        summary_stats = analysis_summary.get('summary_stats', {})
        
        report_parts.append(f"Change Detection:")
        report_parts.append(f"- Total changes: {summary_stats.get('total_changes', 0)}")
        
        if detected_changes:
            change_types = list(set(c.get('type', 'unknown') for c in detected_changes))
            report_parts.append(f"- Change types: {', '.join(change_types)}")
        report_parts.append("")
        
        # Exception analysis
        detected_exceptions = analysis_summary.get('detected_exceptions', [])
        report_parts.append(f"Exception Analysis:")
        report_parts.append(f"- Expected exceptions: {summary_stats.get('expected_exceptions', 0)}")
        report_parts.append(f"- Unexpected exceptions: {summary_stats.get('unexpected_exceptions', 0)}")
        
        # List unexpected exceptions
        unexpected_exceptions = [
            exc for exc in detected_exceptions 
            if exc.get('type') == 'unexpected'
        ]
        
        if unexpected_exceptions:
            report_parts.append("")
            report_parts.append("UNEXPECTED EXCEPTIONS DETECTED:")
            for exc in unexpected_exceptions:
                category = exc.get('category', 'unknown').upper()
                description = exc.get('description', 'No description')
                report_parts.append(f"- {category}: {description}")
        
        # Recommendation
        if validation_result.get('recommendation'):
            report_parts.append("")
            report_parts.append("RECOMMENDATION:")
            report_parts.append(validation_result['recommendation'])
        
        # Detailed analysis
        analysis_details = validation_result.get('analysis_details', {})
        if analysis_details and not validation_result.get('has_error', False):
            report_parts.append("")
            report_parts.append("DETAILED ANALYSIS:")
            for key, value in analysis_details.items():
                if value and key != 'error':
                    formatted_key = key.replace('_', ' ').title()
                    report_parts.append(f"- {formatted_key}: {value}")
        
        # Error information if present
        if validation_result.get('has_error', False):
            report_parts.append("")
            report_parts.append("ERROR INFORMATION:")
            if 'error' in analysis_details:
                report_parts.append(f"- Error: {analysis_details['error']}")
            if 'error_type' in analysis_details:
                report_parts.append(f"- Error Type: {analysis_details['error_type']}")
        
        report_parts.append("")
        report_parts.append("=" * 50)
        
        return "\n".join(report_parts)
    
    async def batch_validate_async(self, 
                                  batch_summaries: List[Dict[str, Any]],
                                  max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Validate multiple UI changes concurrently.
        
        Args:
            batch_summaries: List of analysis summaries to validate
            max_concurrent: Maximum concurrent validations
            
        Returns:
            List of validation results
        """
        self._ensure_initialized()
        
        if not batch_summaries:
            logger.warning("Empty batch provided for validation")
            return []
        
        batch_id = f"batch_{int(time.time() * 1000) % 100000}"
        
        logger.info(
            "Starting batch GPT validation",
            batch_id=batch_id,
            batch_size=len(batch_summaries),
            max_concurrent=max_concurrent
        )
        
        # Create semaphore for batch concurrency control
        batch_semaphore = asyncio.Semaphore(max_concurrent)
        
        async def validate_single(summary: Dict[str, Any], index: int) -> Dict[str, Any]:
            async with batch_semaphore:
                try:
                    result = await self.validate_ui_change_async(summary)
                    result["batch_index"] = index
                    return result
                except Exception as e:
                    logger.error(
                        "Batch validation item failed",
                        batch_id=batch_id,
                        index=index,
                        error=str(e)
                    )
                    return {
                        "is_valid": False,
                        "reasoning": f"Validation failed: {str(e)}",
                        "confidence": 0.0,
                        "has_error": True,
                        "batch_index": index
                    }
        
        # Execute batch validation
        start_time = time.time()
        
        validation_tasks = [
            validate_single(summary, i) 
            for i, summary in enumerate(batch_summaries)
        ]
        
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        batch_time = time.time() - start_time
        
        # Process results and handle exceptions
        processed_results = []
        successful_count = 0
        failed_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Batch validation exception",
                    batch_id=batch_id,
                    index=i,
                    error=str(result)
                )
                processed_results.append({
                    "is_valid": False,
                    "reasoning": f"Batch validation exception: {str(result)}",
                    "confidence": 0.0,
                    "has_error": True,
                    "batch_index": i
                })
                failed_count += 1
            else:
                processed_results.append(result)
                if result.get("has_error", False):
                    failed_count += 1
                else:
                    successful_count += 1
        
        logger.info(
            "Batch GPT validation completed",
            batch_id=batch_id,
            total_items=len(batch_summaries),
            successful_count=successful_count,
            failed_count=failed_count,
            batch_time=f"{batch_time:.2f}s",
            avg_time_per_item=f"{batch_time / len(batch_summaries):.2f}s"
        )
        
        return processed_results
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get service health status."""
        return {
            "service": "gpt_service",
            "initialized": self._is_initialized,
            "model": self.model,
            "vision_model": self.vision_model,
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_timeout": self.request_timeout,
            "total_requests": self._successful_requests + self._failed_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (
                self._successful_requests / (self._successful_requests + self._failed_requests)
                if (self._successful_requests + self._failed_requests) > 0
                else 1.0
            ),
            "sessions_processed": self._session_counter,
            "api_key_configured": bool(self.api_key)
        }


# Global service instance
_gpt_service: Optional[GPTService] = None


async def get_gpt_service() -> GPTService:
    """
    Get global GPT service instance for dependency injection.
    Initializes service if not already done.
    
    Returns:
        GPTService: Initialized GPT service instance
    """
    global _gpt_service
    
    if _gpt_service is None:
        _gpt_service = GPTService()
        await _gpt_service.initialize()
    
    return _gpt_service


async def cleanup_gpt_service() -> None:
    """Clean up global GPT service instance."""
    global _gpt_service
    
    if _gpt_service is not None:
        await _gpt_service.cleanup()
        _gpt_service = None


# Context manager for service lifecycle
@asynccontextmanager
async def gpt_service_context() -> AsyncGenerator[GPTService, None]:
    """
    Context manager for GPT service lifecycle management.
    
    Usage:
        async with gpt_service_context() as gpt_service:
            result = await gpt_service.validate_ui_change_async(analysis_summary)
    """
    service = GPTService()
    try:
        await service.initialize()
        yield service
    finally:
        await service.cleanup()