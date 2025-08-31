"""
Debug service for development and troubleshooting
Provides crop visualization and debugging capabilities
"""
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import structlog

from ..core.config import settings
from ..schemas.validation import UIElement, BoundingBox

logger = structlog.get_logger(__name__)


class DebugService:
    """
    Debug service for saving and visualizing UI element crops.
    Helps developers understand what CLIP is detecting and how elements are processed.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize debug service.
        
        Args:
            output_dir: Directory to save debug outputs (defaults to config)
        """
        self.output_dir = Path(output_dir or settings.debug_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(service="debug")
        self.enabled = settings.enable_debug_crops
    
    async def save_clip_filtered_crops(self,
                                     image_path: str,
                                     elements: List[UIElement],
                                     qa_prompt: str,
                                     session_id: Optional[str] = None) -> Optional[str]:
        """
        Save CLIP-filtered crops for visual inspection.
        
        Args:
            image_path: Path to original image
            elements: List of UI elements with CLIP scores
            qa_prompt: The QA prompt used for CLIP filtering
            session_id: Session identifier for file naming
            
        Returns:
            Path to debug directory if crops were saved, None if disabled
        """
        if not self.enabled:
            return None
        
        session_id = session_id or str(uuid.uuid4())[:8]
        session_logger = self.logger.bind(session_id=session_id)
        
        try:
            # Create session-specific directory
            session_dir = self.output_dir / f"session_{session_id}"
            session_dir.mkdir(exist_ok=True)
            
            # Load full image
            full_image = Image.open(image_path).convert('RGB')
            image_name = Path(image_path).stem
            
            session_logger.info("Saving CLIP crops for debugging",
                              image_path=image_path,
                              elements_count=len(elements),
                              qa_prompt=qa_prompt)
            
            # Save metadata
            metadata = {
                "session_id": session_id,
                "image_path": image_path,
                "qa_prompt": qa_prompt,
                "total_elements": len(elements),
                "crops": []
            }
            
            # Save each crop with CLIP score
            for i, element in enumerate(elements):
                try:
                    bbox = element.bbox
                    clip_score = getattr(element, 'clip_similarity', 0.0)
                    
                    # Crop the element
                    left, top = bbox.x, bbox.y
                    right, bottom = left + bbox.width, top + bbox.height
                    cropped = full_image.crop((left, top, right, bottom))
                    
                    # Create filename with CLIP score and element info
                    filename = f"{image_name}_crop_{i:02d}_score_{clip_score:.3f}_{element.detection_method}.png"
                    filepath = session_dir / filename
                    
                    # Save crop
                    cropped.save(filepath)
                    
                    # Add to metadata
                    metadata["crops"].append({
                        "crop_id": i,
                        "filename": filename,
                        "clip_score": clip_score,
                        "confidence": element.confidence,
                        "caption": element.caption,
                        "detection_method": element.detection_method,
                        "bbox": {
                            "x": bbox.x,
                            "y": bbox.y, 
                            "width": bbox.width,
                            "height": bbox.height
                        }
                    })
                    
                except Exception as e:
                    session_logger.warning("Failed to save crop", 
                                         crop_id=i,
                                         error=str(e))
            
            # Save metadata file
            metadata_path = session_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            session_logger.info("Debug crops saved successfully",
                              crops_saved=len(metadata["crops"]),
                              output_dir=str(session_dir))
            
            return str(session_dir)
            
        except Exception as e:
            session_logger.error("Failed to save debug crops", error=str(e))
            return None
    
    async def save_comparison_crops(self,
                                  before_element: Image.Image,
                                  after_element: Image.Image,
                                  element_type: str,
                                  validation_result: Dict[str, Any],
                                  session_id: Optional[str] = None) -> Optional[str]:
        """
        Save before/after element crops for comparison debugging.
        
        Args:
            before_element: Cropped before state
            after_element: Cropped after state
            element_type: Type of element being compared
            validation_result: GPT validation result
            session_id: Session identifier
            
        Returns:
            Path to saved comparison files
        """
        if not self.enabled:
            return None
        
        session_id = session_id or str(uuid.uuid4())[:8]
        session_logger = self.logger.bind(session_id=session_id)
        
        try:
            # Create comparison directory
            comparison_dir = self.output_dir / f"comparison_{session_id}"
            comparison_dir.mkdir(exist_ok=True)
            
            # Save before/after crops
            before_path = comparison_dir / f"before_{element_type.replace(' ', '_')}.png"
            after_path = comparison_dir / f"after_{element_type.replace(' ', '_')}.png"
            
            before_element.save(before_path)
            after_element.save(after_path)
            
            # Save validation result
            result_path = comparison_dir / "validation_result.json"
            with open(result_path, 'w') as f:
                import json
                json.dump({
                    "element_type": element_type,
                    "validation_result": validation_result,
                    "session_id": session_id,
                    "before_image": str(before_path),
                    "after_image": str(after_path)
                }, f, indent=2)
            
            session_logger.info("Comparison crops saved",
                              element_type=element_type,
                              output_dir=str(comparison_dir))
            
            return str(comparison_dir)
            
        except Exception as e:
            session_logger.error("Failed to save comparison crops", error=str(e))
            return None
    
    async def create_debug_summary(self,
                                 session_id: str,
                                 validation_summary: Dict[str, Any]) -> Optional[str]:
        """
        Create a comprehensive debug summary for a validation session.
        
        Args:
            session_id: Session identifier
            validation_summary: Complete validation results and metadata
            
        Returns:
            Path to summary file if created
        """
        if not self.enabled:
            return None
        
        try:
            summary_path = self.output_dir / f"session_{session_id}_summary.json"
            
            with open(summary_path, 'w') as f:
                import json
                json.dump(validation_summary, f, indent=2, default=str)
            
            self.logger.info("Debug summary created",
                           session_id=session_id,
                           summary_path=str(summary_path))
            
            return str(summary_path)
            
        except Exception as e:
            self.logger.error("Failed to create debug summary",
                            session_id=session_id,
                            error=str(e))
            return None
    
    async def cleanup_old_debug_files(self, max_age_hours: int = 48) -> int:
        """
        Clean up old debug files.
        
        Args:
            max_age_hours: Remove files older than this many hours
            
        Returns:
            Number of files removed
        """
        if not self.output_dir.exists():
            return 0
        
        from datetime import datetime
        cutoff_timestamp = datetime.now().timestamp() - (max_age_hours * 3600)
        removed_count = 0
        
        try:
            for item in self.output_dir.iterdir():
                if item.is_file() and item.stat().st_mtime < cutoff_timestamp:
                    item.unlink()
                    removed_count += 1
                elif item.is_dir():
                    # Remove empty directories or old session directories
                    try:
                        if not any(item.iterdir()) or item.stat().st_mtime < cutoff_timestamp:
                            import shutil
                            shutil.rmtree(item)
                            removed_count += 1
                    except:
                        pass  # Skip if directory not empty or has permission issues
            
            if removed_count > 0:
                self.logger.info("Cleaned up old debug files", removed_count=removed_count)
            
        except Exception as e:
            self.logger.error("Debug cleanup failed", error=str(e))
        
        return removed_count
    
    def get_debug_status(self) -> Dict[str, Any]:
        """Get debug service status and statistics"""
        try:
            # Count files in debug directory
            file_count = 0
            directory_count = 0
            
            if self.output_dir.exists():
                for item in self.output_dir.iterdir():
                    if item.is_file():
                        file_count += 1
                    elif item.is_dir():
                        directory_count += 1
            
            return {
                "enabled": self.enabled,
                "output_dir": str(self.output_dir),
                "debug_files": file_count,
                "debug_directories": directory_count,
                "status": "healthy"
            }
            
        except Exception as e:
            return {
                "enabled": self.enabled,
                "status": "error",
                "error": str(e)
            }


# Global debug service instance
_debug_service: Optional[DebugService] = None


def get_debug_service() -> DebugService:
    """Get or create the global debug service instance"""
    global _debug_service
    
    if _debug_service is None:
        _debug_service = DebugService()
    
    return _debug_service