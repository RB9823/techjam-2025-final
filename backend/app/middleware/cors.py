"""
CORS middleware configuration
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import settings


def setup_cors(app: FastAPI) -> None:
    """Set up CORS middleware for the FastAPI app"""
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"]
    )