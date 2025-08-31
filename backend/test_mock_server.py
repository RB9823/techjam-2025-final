#!/usr/bin/env python3
"""
Standalone test server for mock streaming endpoint
Runs without AI model dependencies for quick testing
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

# Import only the mock streaming router
from app.api.v1.mock_streaming import router as mock_streaming_router

def create_test_app() -> FastAPI:
    """Create minimal FastAPI app for testing mock streaming"""
    
    app = FastAPI(
        title="Mock Streaming Test Server",
        description="Lightweight server for testing mock streaming without AI dependencies",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include only the mock streaming router
    app.include_router(mock_streaming_router, prefix="/api/v1")
    
    @app.get("/")
    async def root():
        return {
            "message": "Mock Streaming Test Server",
            "mock_endpoint": "/api/v1/mock-stream",
            "scenarios": ["heart_like", "button_state", "modal_appearance"],
            "test_ui": "/test"
        }
    
    @app.get("/test")
    async def test_ui():
        """Serve the HTML test interface"""
        html_path = Path(__file__).parent / "test_streaming.html"
        return FileResponse(html_path, media_type="text/html")
    
    return app

app = create_test_app()

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Mock Streaming Test Server (No AI models required)")
    print("=" * 60)
    print("📡 WebSocket endpoint: ws://localhost:8000/api/v1/mock-stream")
    print("🌐 Web test interface: http://localhost:8000/test")
    print("📚 Available scenarios: heart_like, button_state, modal_appearance")
    print("=" * 60)
    print("🧪 To test with Python client: python test_mock_streaming.py")
    print("🌐 To test in browser: Open http://localhost:8000/test")
    print()
    
    uvicorn.run(
        "test_mock_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )