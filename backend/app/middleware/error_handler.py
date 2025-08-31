"""
Global error handling middleware
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import structlog

from ..core.exceptions import UIValidationException

logger = structlog.get_logger(__name__)


def setup_error_handlers(app: FastAPI) -> None:
    """Set up global error handlers for the FastAPI app"""
    
    @app.exception_handler(UIValidationException)
    async def validation_exception_handler(request: Request, exc: UIValidationException):
        """Handle custom validation exceptions"""
        logger.error("Validation exception", 
                    error=exc.message,
                    status_code=exc.status_code,
                    details=exc.details,
                    path=request.url.path)
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "details": exc.details,
                "type": type(exc).__name__
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors"""
        logger.error("Request validation error",
                    errors=exc.errors(),
                    path=request.url.path)
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "Request validation failed",
                "details": exc.errors(),
                "type": "RequestValidationError"
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger.error("HTTP exception",
                    status_code=exc.status_code,
                    detail=exc.detail,
                    path=request.url.path)
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "type": "HTTPException"
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""
        logger.error("Unexpected exception", 
                    error=str(exc),
                    exc_info=True,
                    path=request.url.path)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "type": type(exc).__name__
            }
        )