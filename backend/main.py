"""
UI Validation API - Production FastAPI Backend
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
import structlog

from app.core.config import settings
from app.core.logging import configure_logging
from app.middleware.error_handler import setup_error_handlers
from app.middleware.cors import setup_cors
from app.api.v1.validation import router as validation_router
from app.api.v1.health import router as health_router
from app.api.v1.websocket import router as websocket_router
from app.api.v1.element_detection import router as element_detection_router
from app.api.v1.mock_streaming import router as mock_streaming_router
from app.services import get_validation_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown"""
    # Configure logging
    configure_logging()
    logger = structlog.get_logger(__name__)
    
    # Startup
    logger.info("Starting UI Validation API", version=settings.app_version)
    
    try:
        # Initialize services
        logger.info("Initializing AI services...")
        validation_service = await get_validation_service()
        logger.info("AI services initialized successfully")
        
        # Yield control to the application
        yield
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down UI Validation API")
        try:
            # Cleanup services
            if 'validation_service' in locals():
                await validation_service.cleanup()
            logger.info("Services cleaned up successfully")
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title="UI Validation API",
        description="""
        Production-ready FastAPI backend for AI-powered UI change validation.
        
        ## Features
        
        * **AI-Powered Analysis**: Uses OmniParser (YOLO + Florence-2) for UI element detection
        * **Semantic Filtering**: CLIP-based relevance filtering for efficient processing  
        * **Professional Validation**: GPT-4 powered reasoning for QA validation
        * **Real-time Streaming**: WebSocket support for progress updates
        * **Batch Processing**: Efficient concurrent validation of multiple test cases
        * **Production Ready**: Comprehensive error handling, logging, and monitoring
        
        ## Workflow
        
        1. **Upload Images**: Before and after screenshots with QA prompt
        2. **AI Analysis**: OmniParser detects UI elements with bounding boxes
        3. **Smart Filtering**: CLIP identifies elements relevant to the QA prompt
        4. **Validation**: GPT provides professional QA reasoning and verdict
        5. **Results**: Detailed validation report with detected changes
        
        ## Use Cases
        
        * **UI Testing**: Validate that UI changes work as expected
        * **Visual Regression**: Detect unintended UI changes
        * **QA Automation**: Professional-grade validation reasoning
        * **A/B Testing**: Compare different UI states
        """,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan
    )
    
    # Set up middleware
    setup_cors(app)
    setup_error_handlers(app)
    
    # Include routers
    app.include_router(validation_router, prefix=settings.api_v1_prefix)
    app.include_router(health_router, prefix=settings.api_v1_prefix)
    app.include_router(websocket_router, prefix=settings.api_v1_prefix)
    app.include_router(element_detection_router, prefix=settings.api_v1_prefix)
    app.include_router(mock_streaming_router, prefix=settings.api_v1_prefix)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "UI Validation API",
            "version": settings.app_version,
            "docs": "/docs",
            "health": f"{settings.api_v1_prefix}/health"
        }
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )
