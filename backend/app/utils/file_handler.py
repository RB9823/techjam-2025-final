"""
File handling utilities for image uploads and management
"""
import os
import base64
import uuid
import io
import aiofiles
from pathlib import Path
from typing import Optional
from datetime import datetime
from PIL import Image
import structlog

from fastapi import UploadFile
from ..core.config import settings
from ..core.exceptions import FileUploadError

logger = structlog.get_logger(__name__)


class FileHandler:
    """Handles file uploads, validation, and temporary file management"""
    
    def __init__(self, upload_dir: Optional[str] = None):
        self.upload_dir = Path(upload_dir or settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(service="file_handler")
    
    async def save_upload(self, upload_file: UploadFile, prefix: str = "") -> str:
        """
        Save an uploaded file to the upload directory.
        
        Args:
            upload_file: FastAPI UploadFile object
            prefix: Optional prefix for the filename
            
        Returns:
            Path to the saved file
            
        Raises:
            FileUploadError: If file validation or saving fails
        """
        try:
            # Validate file
            await self._validate_upload_file(upload_file)
            
            # Generate unique filename
            file_extension = Path(upload_file.filename).suffix
            unique_filename = f"{prefix}_{uuid.uuid4()}{file_extension}"
            file_path = self.upload_dir / unique_filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await upload_file.read()
                await f.write(content)
            
            # Validate saved image
            await self._validate_image_file(str(file_path))
            
            self.logger.info("File uploaded successfully", 
                           filename=upload_file.filename,
                           saved_path=str(file_path),
                           size=len(content))
            
            return str(file_path)
            
        except FileUploadError:
            raise
        except Exception as e:
            self.logger.error("File upload failed", 
                            filename=upload_file.filename,
                            error=str(e))
            raise FileUploadError(f"Failed to save file: {e}")
    
    async def save_base64_image(self, base64_data: str, prefix: str = "") -> str:
        """
        Save a base64-encoded image to the upload directory.
        
        Args:
            base64_data: Base64-encoded image data (with or without data URL prefix)
            prefix: Optional prefix for the filename
            
        Returns:
            Path to the saved file
        """
        try:
            # Parse base64 data
            if base64_data.startswith('data:'):
                # Extract base64 part from data URL
                header, base64_content = base64_data.split(',', 1)
                # Extract format from header (e.g., "data:image/png;base64")
                format_part = header.split(';')[0].split('/')[1]
                file_extension = f".{format_part}"
            else:
                base64_content = base64_data
                file_extension = ".png"  # Default to PNG
            
            # Decode base64
            image_data = base64.b64decode(base64_content)
            
            # Validate image data
            try:
                image = Image.open(io.BytesIO(image_data))
                image.verify()
            except Exception as e:
                raise FileUploadError(f"Invalid image data: {e}")
            
            # Save to file
            unique_filename = f"{prefix}_{uuid.uuid4()}{file_extension}"
            file_path = self.upload_dir / unique_filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(image_data)
            
            self.logger.info("Base64 image saved successfully",
                           saved_path=str(file_path),
                           size=len(image_data))
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error("Base64 image save failed", error=str(e))
            raise FileUploadError(f"Failed to save base64 image: {e}")
    
    async def _validate_upload_file(self, upload_file: UploadFile) -> None:
        """Validate uploaded file meets requirements"""
        # Check content type
        if upload_file.content_type not in settings.allowed_file_types:
            raise FileUploadError(
                f"File type not allowed: {upload_file.content_type}. "
                f"Allowed types: {', '.join(settings.allowed_file_types)}"
            )
        
        # Check file size
        if upload_file.size > settings.max_file_size:
            raise FileUploadError(
                f"File too large: {upload_file.size} bytes. "
                f"Maximum size: {settings.max_file_size} bytes"
            )
        
        # Check filename
        if not upload_file.filename:
            raise FileUploadError("Filename is required")
    
    async def _validate_image_file(self, file_path: str) -> None:
        """Validate that the saved file is a valid image"""
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            # Clean up invalid file
            try:
                os.unlink(file_path)
            except:
                pass
            raise FileUploadError(f"Invalid image file: {e}")
    
    async def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Remove old uploaded files"""
        cutoff_timestamp = datetime.now().timestamp() - (max_age_hours * 3600)
        removed_count = 0
        
        try:
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_timestamp:
                    file_path.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                self.logger.info("Cleaned up old files", removed_count=removed_count)
            
        except Exception as e:
            self.logger.error("File cleanup failed", error=str(e))
        
        return removed_count


# Global file handler instance
_file_handler: Optional[FileHandler] = None


async def get_file_handler() -> FileHandler:
    """Get or create the global file handler instance"""
    global _file_handler
    
    if _file_handler is None:
        _file_handler = FileHandler()
    
    return _file_handler