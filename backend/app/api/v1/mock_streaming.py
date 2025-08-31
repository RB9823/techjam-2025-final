"""
Mock streaming endpoint for UI validation pipeline demonstration
Provides realistic progression through AI pipeline stages with comprehensive mock data
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List
from enum import Enum

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import structlog

from ...schemas.validation import (
    ValidationProgress,
    ValidationResponse,
    ValidationJobStatus,
    StreamMessage,
    UIElement,
    BoundingBox,
    DetectedChange,
    DetectedException,
    ChangeType,
    ExceptionType,
    ExceptionCategory
)

router = APIRouter(tags=["mock-streaming"])
logger = structlog.get_logger(__name__)


class MockScenario(str, Enum):
    """Pre-defined validation scenarios for demonstration"""
    HEART_LIKE = "heart_like"
    BUTTON_STATE = "button_state"  
    MODAL_APPEARANCE = "modal_appearance"


def get_mock_ui_elements_before(scenario: MockScenario) -> List[UIElement]:
    """Generate realistic before-state UI elements for different scenarios"""
    
    if scenario == MockScenario.HEART_LIKE:
        return [
            UIElement(
                id="header_nav",
                bbox=BoundingBox(x=0, y=0, width=1920, height=80),
                caption="Navigation header with logo and menu items",
                confidence=0.95,
                detection_method="omniparser",
                clip_similarity=0.3
            ),
            UIElement(
                id="post_content",
                bbox=BoundingBox(x=300, y=120, width=800, height=400),
                caption="Social media post with image and text content",
                confidence=0.92,
                detection_method="omniparser", 
                clip_similarity=0.7
            ),
            UIElement(
                id="heart_icon_unfilled",
                bbox=BoundingBox(x=320, y=540, width=32, height=32),
                caption="Heart icon in unfilled/unliked state (gray color)",
                confidence=0.98,
                detection_method="gpt4v",
                clip_similarity=0.95
            ),
            UIElement(
                id="like_count",
                bbox=BoundingBox(x=360, y=545, width=60, height=22),
                caption="Like count showing '247 likes'",
                confidence=0.89,
                detection_method="florence2",
                clip_similarity=0.8
            ),
            UIElement(
                id="comment_button",
                bbox=BoundingBox(x=450, y=540, width=32, height=32),
                caption="Comment button icon",
                confidence=0.91,
                detection_method="yolo",
                clip_similarity=0.4
            ),
            UIElement(
                id="share_button",
                bbox=BoundingBox(x=490, y=540, width=32, height=32),
                caption="Share button icon",
                confidence=0.88,
                detection_method="yolo",
                clip_similarity=0.3
            )
        ]
    
    elif scenario == MockScenario.BUTTON_STATE:
        return [
            UIElement(
                id="form_container",
                bbox=BoundingBox(x=400, y=200, width=600, height=500),
                caption="Contact form container with input fields",
                confidence=0.96,
                detection_method="omniparser",
                clip_similarity=0.6
            ),
            UIElement(
                id="name_input",
                bbox=BoundingBox(x=450, y=250, width=500, height=40),
                caption="Name input field with placeholder text",
                confidence=0.94,
                detection_method="florence2",
                clip_similarity=0.7
            ),
            UIElement(
                id="email_input",
                bbox=BoundingBox(x=450, y=310, width=500, height=40),
                caption="Email input field with validation",
                confidence=0.93,
                detection_method="florence2",
                clip_similarity=0.7
            ),
            UIElement(
                id="message_textarea",
                bbox=BoundingBox(x=450, y=370, width=500, height=120),
                caption="Message textarea for user input",
                confidence=0.91,
                detection_method="florence2",
                clip_similarity=0.6
            ),
            UIElement(
                id="submit_button_enabled",
                bbox=BoundingBox(x=450, y=520, width=150, height=45),
                caption="Submit button in enabled state (blue background, white text)",
                confidence=0.97,
                detection_method="gpt4v",
                clip_similarity=0.92
            ),
            UIElement(
                id="cancel_button",
                bbox=BoundingBox(x=620, y=520, width=100, height=45),
                caption="Cancel button (gray background)",
                confidence=0.89,
                detection_method="yolo",
                clip_similarity=0.4
            )
        ]
    
    elif scenario == MockScenario.MODAL_APPEARANCE:
        return [
            UIElement(
                id="data_table",
                bbox=BoundingBox(x=100, y=150, width=1200, height=600),
                caption="Data table with user records and actions",
                confidence=0.94,
                detection_method="omniparser",
                clip_similarity=0.8
            ),
            UIElement(
                id="table_header",
                bbox=BoundingBox(x=100, y=150, width=1200, height=50),
                caption="Table header with column names",
                confidence=0.92,
                detection_method="florence2",
                clip_similarity=0.5
            ),
            UIElement(
                id="user_row_1",
                bbox=BoundingBox(x=100, y=200, width=1200, height=60),
                caption="User row: John Doe, john@example.com, Active",
                confidence=0.89,
                detection_method="florence2",
                clip_similarity=0.7
            ),
            UIElement(
                id="delete_button_row1",
                bbox=BoundingBox(x=1150, y=215, width=80, height=30),
                caption="Delete button for first user row (red background)",
                confidence=0.96,
                detection_method="gpt4v",
                clip_similarity=0.93
            ),
            UIElement(
                id="edit_button_row1",
                bbox=BoundingBox(x=1050, y=215, width=80, height=30),
                caption="Edit button for first user row",
                confidence=0.91,
                detection_method="yolo",
                clip_similarity=0.4
            )
        ]
    
    return []


def get_mock_ui_elements_after(scenario: MockScenario) -> List[UIElement]:
    """Generate realistic after-state UI elements showing expected changes"""
    
    if scenario == MockScenario.HEART_LIKE:
        elements = get_mock_ui_elements_before(scenario)
        # Modify the heart icon to show filled/liked state
        for element in elements:
            if element.id == "heart_icon_unfilled":
                element.id = "heart_icon_filled"
                element.caption = "Heart icon in filled/liked state (red color with animation effect)"
                element.confidence = 0.97
                element.clip_similarity = 0.98
            elif element.id == "like_count":
                element.caption = "Like count showing '248 likes' (increased by 1)"
                element.confidence = 0.91
        return elements
    
    elif scenario == MockScenario.BUTTON_STATE:
        elements = get_mock_ui_elements_before(scenario)
        # Show form submission loading state
        for element in elements:
            if element.id == "submit_button_enabled":
                element.id = "submit_button_loading"
                element.caption = "Submit button in loading state (disabled, spinner icon, gray background)"
                element.confidence = 0.95
                element.clip_similarity = 0.96
        
        # Add loading spinner element
        elements.append(UIElement(
            id="loading_spinner",
            bbox=BoundingBox(x=470, y=530, width=24, height=24),
            caption="Loading spinner icon inside submit button",
            confidence=0.93,
            detection_method="gpt4v",
            clip_similarity=0.89
        ))
        return elements
    
    elif scenario == MockScenario.MODAL_APPEARANCE:
        elements = get_mock_ui_elements_before(scenario)
        
        # Add modal backdrop
        elements.append(UIElement(
            id="modal_backdrop",
            bbox=BoundingBox(x=0, y=0, width=1920, height=1080),
            caption="Dark modal backdrop overlay covering entire screen",
            confidence=0.98,
            detection_method="omniparser",
            clip_similarity=0.95
        ))
        
        # Add confirmation modal
        elements.append(UIElement(
            id="confirmation_modal",
            bbox=BoundingBox(x=660, y=340, width=600, height=400),
            caption="Confirmation dialog modal with delete warning",
            confidence=0.97,
            detection_method="gpt4v",
            clip_similarity=0.98
        ))
        
        elements.append(UIElement(
            id="modal_title",
            bbox=BoundingBox(x=690, y=370, width=300, height=30),
            caption="Modal title: 'Confirm Deletion'",
            confidence=0.94,
            detection_method="florence2",
            clip_similarity=0.8
        ))
        
        elements.append(UIElement(
            id="modal_message",
            bbox=BoundingBox(x=690, y=420, width=540, height=80),
            caption="Warning message: 'Are you sure you want to delete John Doe? This action cannot be undone.'",
            confidence=0.91,
            detection_method="florence2",
            clip_similarity=0.9
        ))
        
        elements.append(UIElement(
            id="modal_cancel_button",
            bbox=BoundingBox(x=1050, y=660, width=100, height=40),
            caption="Cancel button in modal",
            confidence=0.89,
            detection_method="yolo",
            clip_similarity=0.5
        ))
        
        elements.append(UIElement(
            id="modal_delete_button",
            bbox=BoundingBox(x=1160, y=660, width=100, height=40),
            caption="Confirm delete button (red background)",
            confidence=0.95,
            detection_method="gpt4v",
            clip_similarity=0.94
        ))
        
        return elements
    
    return []


def get_mock_detected_changes(scenario: MockScenario) -> List[DetectedChange]:
    """Generate realistic detected changes for each scenario"""
    
    changes = []
    
    if scenario == MockScenario.HEART_LIKE:
        changes.extend([
            DetectedChange(
                element_id="heart_icon_unfilled",
                change_type=ChangeType.STATE_CHANGED,
                before_element=UIElement(
                    id="heart_icon_unfilled",
                    bbox=BoundingBox(x=320, y=540, width=32, height=32),
                    caption="Heart icon in unfilled/unliked state (gray color)",
                    confidence=0.98,
                    detection_method="gpt4v"
                ),
                after_element=UIElement(
                    id="heart_icon_filled",
                    bbox=BoundingBox(x=320, y=540, width=32, height=32),
                    caption="Heart icon in filled/liked state (red color)",
                    confidence=0.97,
                    detection_method="gpt4v"
                ),
                confidence=0.96,
                details="Heart icon changed from gray unfilled state to red filled state, indicating successful like action"
            ),
            DetectedChange(
                element_id="like_count",
                change_type=ChangeType.MODIFIED,
                confidence=0.88,
                details="Like count incremented from 247 to 248, confirming the like was registered"
            )
        ])
    
    elif scenario == MockScenario.BUTTON_STATE:
        changes.append(DetectedChange(
            element_id="submit_button_enabled",
            change_type=ChangeType.STATE_CHANGED,
            confidence=0.94,
            details="Submit button transitioned from enabled state to loading/disabled state with spinner, preventing double submission"
        ))
    
    elif scenario == MockScenario.MODAL_APPEARANCE:
        changes.extend([
            DetectedChange(
                element_id="modal_backdrop",
                change_type=ChangeType.ADDED,
                confidence=0.97,
                details="Dark modal backdrop appeared, covering the background content"
            ),
            DetectedChange(
                element_id="confirmation_modal",
                change_type=ChangeType.ADDED,
                confidence=0.98,
                details="Confirmation dialog modal appeared with delete warning and action buttons"
            )
        ])
    
    return changes


def get_mock_validation_reasoning(scenario: MockScenario) -> Dict[str, Any]:
    """Generate professional QA validation reasoning for each scenario"""
    
    if scenario == MockScenario.HEART_LIKE:
        return {
            "is_valid": True,
            "confidence": 0.94,
            "reasoning": """✅ VALIDATION PASSED: Heart Like Functionality

**Analysis Summary:**
The UI interaction successfully demonstrates the expected heart like behavior. The heart icon correctly transitioned from an unfilled gray state to a filled red state upon user interaction.

**Key Evidence:**
• Heart icon visual state change detected (unfilled → filled, gray → red)
• Like count incremented correctly (247 → 248)
• No unexpected UI exceptions or errors detected
• Interaction feedback was immediate and visually clear

**Professional Assessment:**
This represents a standard and well-implemented like functionality that meets user experience expectations. The visual feedback is appropriate and the state persistence appears correct."""
        }
    
    elif scenario == MockScenario.BUTTON_STATE:
        return {
            "is_valid": True,
            "confidence": 0.91,
            "reasoning": """✅ VALIDATION PASSED: Submit Button State Management

**Analysis Summary:**
The form submission properly implements loading state management. The submit button correctly transitioned to a disabled state with loading indicator, preventing potential double submissions.

**Key Evidence:**
• Submit button state change detected (enabled → loading/disabled)
• Loading spinner appeared within button
• Button background color changed to indicate disabled state
• Form maintains proper UX during async operations

**Professional Assessment:**
This demonstrates proper form submission handling with appropriate user feedback. The loading state prevents user confusion and system errors from duplicate submissions."""
        }
    
    elif scenario == MockScenario.MODAL_APPEARANCE:
        return {
            "is_valid": True,
            "confidence": 0.97,
            "reasoning": """✅ VALIDATION PASSED: Modal Dialog Appearance

**Analysis Summary:**
The delete action successfully triggered the expected confirmation modal dialog. The modal implementation follows proper UX patterns with appropriate warning messaging and clear action choices.

**Key Evidence:**
• Modal backdrop overlay detected covering background
• Confirmation dialog appeared with correct positioning
• Warning message clearly explains the destructive action
• Cancel and confirm buttons provide clear user choice options

**Professional Assessment:**
This represents an excellent implementation of destructive action confirmation. The modal prevents accidental deletions and provides clear communication about irreversible consequences."""
        }
    
    return {"is_valid": False, "confidence": 0.0, "reasoning": "Unknown scenario"}


def get_progress_messages(stage: str, scenario: MockScenario) -> Dict[str, Any]:
    """Get detailed progress messages for each stage"""
    
    base_messages = {
        "initialization": {
            "message": "Initializing AI validation pipeline",
            "details": "Loading OmniParser, CLIP, and GPT models for comprehensive UI analysis"
        },
        "validation": {
            "message": "Validating input screenshots",
            "details": "Checking image quality, resolution, and format compatibility"
        },
        "parsing": {
            "message": "Detecting UI elements with OmniParser",
            "details": "Using advanced computer vision to identify all interactive elements"
        },
        "filtering": {
            "message": "Applying CLIP semantic filtering",
            "details": "Filtering elements relevant to the validation query using multimodal AI"
        },
        "analysis": {
            "message": "Analyzing UI state changes",
            "details": "Comparing before/after states to detect visual and functional changes"
        },
        "exceptions": {
            "message": "Detecting UI exceptions and anomalies",
            "details": "Scanning for unexpected behaviors, errors, or layout issues"
        },
        "validation": {
            "message": "Generating professional QA validation",
            "details": "Using GPT to provide detailed reasoning and validation assessment"
        },
        "compilation": {
            "message": "Compiling comprehensive validation report",
            "details": "Aggregating all analysis results into final validation response"
        },
        "completed": {
            "message": "Validation completed successfully",
            "details": "All pipeline stages completed with detailed results available"
        }
    }
    
    # Add scenario-specific context
    scenario_context = {
        MockScenario.HEART_LIKE: " for heart like interaction",
        MockScenario.BUTTON_STATE: " for form button state change", 
        MockScenario.MODAL_APPEARANCE: " for modal dialog appearance"
    }
    
    message_data = base_messages.get(stage, {"message": f"Processing {stage}", "details": ""})
    message_data["message"] += scenario_context.get(scenario, "")
    
    return message_data


async def simulate_realistic_processing_delay():
    """Add realistic delay between processing stages"""
    # Random delay between 1.5-3.5 seconds to simulate real AI processing
    import random
    delay = random.uniform(1.5, 3.5)
    await asyncio.sleep(delay)


@router.websocket("/mock-stream")
async def mock_validation_stream(websocket: WebSocket):
    """
    Mock WebSocket endpoint for demonstrating UI validation streaming.
    
    Client should send JSON messages with scenario selection:
    ```json
    {
        "type": "validate",
        "data": {
            "scenario": "heart_like|button_state|modal_appearance",
            "qa_prompt": "Does heart turn red when liked?",
            "before_image_base64": "data:image/png;base64,...",
            "after_image_base64": "data:image/png;base64,..."
        }
    }
    ```
    
    Returns realistic streaming progress and final validation results.
    """
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    session_logger = logger.bind(connection_id=connection_id, endpoint="mock_stream")
    
    session_logger.info("Mock WebSocket connected")
    
    try:
        while True:
            # Wait for client message
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                message_data = message.get("data", {})
                
                session_logger.info("Received mock validation request", type=message_type)
                
                if message_type == "validate":
                    await handle_mock_validation(
                        websocket,
                        connection_id,
                        message_data,
                        session_logger
                    )
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"message": f"Unknown message type: {message_type}"}
                    }))
                    
            except json.JSONDecodeError as e:
                session_logger.error("Invalid JSON received", error=str(e))
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": {"message": "Invalid JSON format"}
                }))
                
    except WebSocketDisconnect:
        session_logger.info("Mock WebSocket disconnected normally")
    except Exception as e:
        session_logger.error("Mock WebSocket error", error=str(e))
    

def get_mock_debug_assets(scenario: MockScenario, job_id: str) -> Dict[str, Any]:
    """Generate mock debug assets and crop metadata"""
    
    session_id = job_id[:8]
    base_debug_url = f"/api/v1/mock-stream/debug/{session_id}"
    
    debug_data = {
        "session_id": session_id,
        "debug_enabled": True,
        "crop_count": len(get_mock_ui_elements_before(scenario)),
        "crops": [],
        "analysis_metadata": {
            "scenario": scenario,
            "detection_methods": ["omniparser", "gpt4v", "florence2", "yolo"],
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    # Generate mock crop data for each detected element
    elements = get_mock_ui_elements_before(scenario)
    for i, element in enumerate(elements):
        crop_filename = f"crop_{i:03d}_{element.id}.png"
        debug_data["crops"].append({
            "element_id": element.id,
            "crop_filename": crop_filename,
            "crop_url": f"{base_debug_url}/crop/{crop_filename}",
            "bbox": element.bbox.dict(),
            "confidence": element.confidence,
            "detection_method": element.detection_method,
            "clip_similarity": element.clip_similarity,
            "analysis_notes": f"Detected {element.caption.lower()} with {element.confidence:.1%} confidence"
        })
    
    return debug_data


def get_mock_image_references(scenario: MockScenario, job_id: str) -> Dict[str, str]:
    """Generate mock image URLs for before/after screenshots and crops"""
    
    base_url = f"/api/v1/mock-stream/images/{job_id[:8]}"
    
    return {
        "before_image_url": f"{base_url}/before_screenshot.png",
        "after_image_url": f"{base_url}/after_screenshot.png", 
        "before_thumbnail": f"{base_url}/before_thumb.png",
        "after_thumbnail": f"{base_url}/after_thumb.png",
        "comparison_overlay": f"{base_url}/comparison_overlay.png",
        "heatmap_visualization": f"{base_url}/attention_heatmap.png"
    }


async def handle_mock_validation(
    websocket: WebSocket,
    connection_id: str,
    data: Dict[str, Any],
    session_logger
):
    """Handle mock validation request with realistic progression"""
    job_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    # Extract scenario
    scenario_str = data.get("scenario", "heart_like")
    try:
        scenario = MockScenario(scenario_str)
    except ValueError:
        scenario = MockScenario.HEART_LIKE
    
    qa_prompt = data.get("qa_prompt", f"Mock validation for {scenario}")
    
    session_logger = session_logger.bind(job_id=job_id, scenario=scenario)
    session_logger.info("Starting mock validation")
    
    # Generate mock assets
    debug_assets = get_mock_debug_assets(scenario, job_id)
    image_refs = get_mock_image_references(scenario, job_id)
    
    # Define processing stages with realistic progression
    stages = [
        ("initialization", 0),
        ("validation", 5),
        ("parsing", 10),
        ("filtering", 40),
        ("analysis", 60),
        ("exceptions", 70),
        ("validation", 80),
        ("compilation", 95),
        ("completed", 100)
    ]
    
    try:
        # Progress through all stages
        for stage_name, progress_percent in stages:
            await simulate_realistic_processing_delay()
            
            # Get detailed progress message
            progress_info = get_progress_messages(stage_name, scenario)
            
            # Create enhanced progress data
            progress_data = ValidationProgress(
                stage=stage_name,
                progress_percent=progress_percent,
                message=progress_info["message"]
            )
            
            # Add extra details for certain stages
            extra_data = {"details": progress_info["details"]}
            
            if stage_name == "parsing":
                before_elements = get_mock_ui_elements_before(scenario)
                extra_data.update({
                    "elements_detected": len(before_elements),
                    "detection_methods": ["omniparser", "gpt4v", "florence2", "yolo"],
                    "processing_time": "2.3s",
                    "before_elements": [elem.dict() for elem in before_elements],
                    "after_elements": [elem.dict() for elem in before_elements],
                    "debug_assets": debug_assets,
                    "image_references": image_refs
                })
            elif stage_name == "filtering":
                filtered_elements = get_mock_ui_elements_before(scenario)[:3]  # Simulate filtering
                extra_data.update({
                    "clip_similarity_scores": [0.95, 0.92, 0.89, 0.85, 0.78],
                    "elements_filtered": len(filtered_elements),
                    "relevance_threshold": 0.75,
                    "filtered_elements": [elem.dict() for elem in filtered_elements],
                    "filtered_after_elements": [elem.dict() for elem in filtered_elements],
                    "clip_analysis": {
                        "query_embedding_computed": True,
                        "semantic_matching_complete": True,
                        "top_matches": [
                            {"element_id": filtered_elements[0].id, "similarity": 0.95},
                            {"element_id": filtered_elements[1].id, "similarity": 0.92},
                            {"element_id": filtered_elements[2].id, "similarity": 0.89}
                        ]
                    }
                })
            elif stage_name == "analysis":
                detected_changes = get_mock_detected_changes(scenario)
                extra_data.update({
                    "changes_detected": len(detected_changes),
                    "change_types": [change.change_type for change in detected_changes],
                    "confidence_range": "0.88-0.98",
                    "detected_changes": [change.dict() for change in detected_changes],
                    "before_after_comparison": {
                        "elements_before": len(get_mock_ui_elements_before(scenario)),
                        "elements_after": len(get_mock_ui_elements_after(scenario)),
                        "new_elements": 1 if scenario == MockScenario.MODAL_APPEARANCE else 0,
                        "modified_elements": len(detected_changes)
                    }
                })
            elif stage_name == "validation":
                validation_result = get_mock_validation_reasoning(scenario)
                extra_data.update({
                    "preliminary_result": validation_result["is_valid"],
                    "confidence": validation_result["confidence"],
                    "reasoning_length": len(validation_result["reasoning"]),
                    "gpt_analysis": {
                        "model_used": "gpt-4-vision-preview",
                        "tokens_consumed": 1247,
                        "processing_time": "3.8s",
                        "reasoning_quality": "comprehensive"
                    }
                })
            
            # Send progress message - bypass ValidationProgress model to preserve all data
            progress_message = StreamMessage(
                job_id=job_id,
                type="progress",
                data={
                    "stage": stage_name,
                    "progress_percent": progress_percent,
                    "message": progress_info["message"],
                    "timestamp": datetime.utcnow(),
                    **extra_data
                }
            )
            
            await websocket.send_text(progress_message.json())
            session_logger.info(f"Sent {stage_name} progress", progress=progress_percent)
        
        # Generate comprehensive final result
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        validation_result = get_mock_validation_reasoning(scenario)
        
        final_response = ValidationResponse(
            job_id=job_id,
            status=ValidationJobStatus.COMPLETED,
            is_valid=validation_result["is_valid"],
            reasoning=validation_result["reasoning"],
            confidence=validation_result["confidence"],
            detected_changes=get_mock_detected_changes(scenario),
            detected_exceptions=[],  # No exceptions for successful scenarios
            before_elements=get_mock_ui_elements_before(scenario),
            after_elements=get_mock_ui_elements_after(scenario),
            processing_time_seconds=processing_time,
            completed_at=datetime.utcnow(),
            stats={
                "scenario": scenario,
                "before_elements_count": len(get_mock_ui_elements_before(scenario)),
                "after_elements_count": len(get_mock_ui_elements_after(scenario)),
                "changes_detected": len(get_mock_detected_changes(scenario)),
                "exceptions_detected": 0,
                "pipeline_stages": len(stages),
                "total_processing_time": f"{processing_time:.2f}s",
                "image_references": image_refs,
                "debug_assets": debug_assets,
                "ai_pipeline_stats": {
                    "omniparser_elements": len(get_mock_ui_elements_before(scenario)),
                    "clip_filtered_elements": len(get_mock_ui_elements_before(scenario)[:3]),
                    "gpt_validation_confidence": validation_result["confidence"],
                    "total_ai_processing_time": f"{processing_time * 0.8:.1f}s",
                    "model_performance": {
                        "omniparser_accuracy": 0.94,
                        "clip_relevance_score": 0.87,
                        "gpt_reasoning_quality": 0.92
                    }
                }
            }
        )
        
        # Send final result
        result_message = StreamMessage(
            job_id=job_id,
            type="result",
            data=final_response
        )
        
        await websocket.send_text(result_message.json())
        session_logger.info("Mock validation completed successfully",
                           processing_time=processing_time,
                           is_valid=final_response.is_valid)
        
    except Exception as e:
        session_logger.error("Mock validation failed", error=str(e))
        
        error_message = StreamMessage(
            job_id=job_id,
            type="error",
            data={
                "message": f"Mock validation failed: {str(e)}",
                "job_id": job_id,
                "scenario": scenario
            }
        )
        
        await websocket.send_text(error_message.json())