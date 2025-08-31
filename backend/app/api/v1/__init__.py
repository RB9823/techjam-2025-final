from .validation import router as validation_router
from .health import router as health_router
from .websocket import router as websocket_router
from .element_detection import router as element_detection_router

__all__ = ["validation_router", "health_router", "websocket_router", "element_detection_router"]