"""
WebSocket endpoints for real-time streaming
"""
import json
import uuid
from typing import Dict, Any, Set
import asyncio
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import structlog

from ...schemas.validation import (
    ValidationProgress,
    ValidationResponse,
    StreamMessage,
    ValidationRequest
)
from ...services.validation_service import ValidationService, get_validation_service
from ...utils.file_handler import FileHandler, get_file_handler
from ...core.config import settings

router = APIRouter(tags=["websocket"])
logger = structlog.get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for streaming validation progress"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.job_connections: Dict[str, Set[str]] = {}  # job_id -> set of connection_ids
    
    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection and return connection ID"""
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        logger.info("WebSocket connected", connection_id=connection_id)
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        # Remove from job mappings
        for job_id, conn_ids in self.job_connections.items():
            conn_ids.discard(connection_id)
        
        logger.info("WebSocket disconnected", connection_id=connection_id)
    
    def subscribe_to_job(self, connection_id: str, job_id: str):
        """Subscribe a connection to job updates"""
        if job_id not in self.job_connections:
            self.job_connections[job_id] = set()
        self.job_connections[job_id].add(connection_id)
    
    async def send_to_connection(self, connection_id: str, message: StreamMessage):
        """Send message to a specific connection"""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(message.json())
            except Exception as e:
                logger.warning("Failed to send message", 
                              connection_id=connection_id, 
                              error=str(e))
                self.disconnect(connection_id)
    
    async def broadcast_to_job(self, job_id: str, message: StreamMessage):
        """Send message to all connections subscribed to a job"""
        if job_id in self.job_connections:
            connections = list(self.job_connections[job_id])
            for connection_id in connections:
                await self.send_to_connection(connection_id, message)


# Global connection manager
connection_manager = ConnectionManager()


@router.websocket("/stream")
async def websocket_validation_stream(
    websocket: WebSocket,
    validation_service: ValidationService = Depends(get_validation_service),
    file_handler: FileHandler = Depends(get_file_handler)
):
    """
    WebSocket endpoint for streaming validation progress.
    
    Client should send JSON messages with validation requests and receive
    real-time progress updates and final results.
    
    Message format:
    ```json
    {
        "type": "validate",
        "data": {
            "qa_prompt": "Does heart turn red when liked?",
            "before_image_base64": "data:image/png;base64,...",
            "after_image_base64": "data:image/png;base64,...",
            "options": {
                "max_gpt_calls": 10,
                "detection_confidence": settings.detection_confidence
            }
        }
    }
    ```
    
    Response messages:
    - `{"type": "progress", "data": {...}}` - Progress updates
    - `{"type": "result", "data": {...}}` - Final validation result
    - `{"type": "error", "data": {...}}` - Error information
    """
    connection_id = await connection_manager.connect(websocket)
    session_logger = logger.bind(connection_id=connection_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                message_data = message.get("data", {})
                
                session_logger.info("Received WebSocket message", type=message_type)
                
                if message_type == "validate":
                    await handle_validation_request(
                        websocket,
                        connection_id,
                        message_data,
                        validation_service,
                        file_handler,
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
        session_logger.info("WebSocket disconnected normally")
    except Exception as e:
        session_logger.error("WebSocket error", error=str(e), exc_info=True)
    finally:
        connection_manager.disconnect(connection_id)


async def handle_validation_request(
    websocket: WebSocket,
    connection_id: str,
    data: Dict[str, Any],
    validation_service: ValidationService,
    file_handler: FileHandler,
    session_logger
):
    """Handle a validation request over WebSocket"""
    job_id = str(uuid.uuid4())
    
    try:
        # Extract request data
        qa_prompt = data.get("qa_prompt")
        before_image_data = data.get("before_image_base64")
        after_image_data = data.get("after_image_base64")
        options = data.get("options", {})
        
        if not all([qa_prompt, before_image_data, after_image_data]):
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"message": "Missing required fields: qa_prompt, before_image_base64, after_image_base64"}
            }))
            return
        
        # Subscribe connection to job updates
        connection_manager.subscribe_to_job(connection_id, job_id)
        
        # Progress callback for streaming updates
        async def progress_callback(progress: ValidationProgress):
            message = StreamMessage(
                job_id=job_id,
                type="progress",
                data=progress
            )
            await connection_manager.send_to_connection(connection_id, message)
        
        # Save base64 images to temporary files
        before_path = await file_handler.save_base64_image(before_image_data, f"ws_before_{job_id}")
        after_path = await file_handler.save_base64_image(after_image_data, f"ws_after_{job_id}")
        
        # Create validation request
        request = ValidationRequest(
            qa_prompt=qa_prompt,
            enable_streaming=True,
            max_gpt_calls=options.get("max_gpt_calls"),
            detection_confidence=options.get("detection_confidence")
        )
        
        # Process validation with streaming
        result = await validation_service.validate_ui_change(
            request=request,
            before_image_path=before_path,
            after_image_path=after_path,
            progress_callback=progress_callback
        )
        
        # Send final result
        final_message = StreamMessage(
            job_id=job_id,
            type="result",
            data=result
        )
        await connection_manager.send_to_connection(connection_id, final_message)
        
        session_logger.info("WebSocket validation completed", 
                           job_id=job_id,
                           is_valid=result.is_valid)
        
    except Exception as e:
        session_logger.error("WebSocket validation failed", 
                           job_id=job_id,
                           error=str(e))
        
        error_message = StreamMessage(
            job_id=job_id,
            type="error",
            data={"message": str(e), "job_id": job_id}
        )
        await connection_manager.send_to_connection(connection_id, error_message)
    
    finally:
        # Cleanup temporary files
        try:
            import os
            if 'before_path' in locals() and os.path.exists(before_path):
                os.unlink(before_path)
            if 'after_path' in locals() and os.path.exists(after_path):
                os.unlink(after_path)
        except Exception as e:
            session_logger.warning("Failed to cleanup temporary files", error=str(e))