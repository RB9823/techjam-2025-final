#!/usr/bin/env python3
"""
Production startup script for UI Validation API
"""
import asyncio
import uvicorn
from app.core.config import settings


def main():
    """Run the FastAPI application"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
        access_log=True,
        workers=1 if settings.debug else 4
    )


if __name__ == "__main__":
    main()