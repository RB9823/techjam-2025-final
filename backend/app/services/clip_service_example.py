"""
Example usage of the production CLIP service.

This demonstrates how to use the CLIPService for semantic filtering of UI elements.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any

from .clip_service import CLIPService, get_clip_service, clip_service_context


async def example_usage():
    """Example of using the CLIP service."""
    
    # Example UI elements (similar to what OmniParser would produce)
    example_elements = [
        {
            'id': 'heart_icon',
            'bbox': {'x': 100, 'y': 200, 'width': 50, 'height': 50},
            'description': 'heart icon button',
            'text': 'heart'
        },
        {
            'id': 'share_button', 
            'bbox': {'x': 200, 'y': 200, 'width': 50, 'height': 50},
            'description': 'share button',
            'text': 'share'
        },
        {
            'id': 'comment_icon',
            'bbox': {'x': 300, 'y': 200, 'width': 50, 'height': 50}, 
            'description': 'comment icon',
            'text': 'comment'
        }
    ]
    
    qa_prompt = "Does heart turn red when liked?"
    
    print("🚀 CLIP Service Example")
    print(f"Processing {len(example_elements)} elements with prompt: '{qa_prompt}'")
    
    # Method 1: Using global service instance (recommended for FastAPI)
    print("\n📋 Method 1: Using global service instance")
    try:
        clip_service = await get_clip_service()
        
        # Note: You would need an actual image file for this to work
        image_path = "example_screenshot.png"  # Replace with actual path
        
        # This would fail without actual image, but shows the API
        # filtered_elements = await clip_service.filter_elements_async(
        #     image_path=image_path,
        #     elements=example_elements,
        #     qa_prompt=qa_prompt,
        #     max_elements=5,
        #     save_debug_crops=True
        # )
        
        # Get service health status instead
        health = await clip_service.get_health_status()
        print(f"Service health: {health}")
        
    except Exception as e:
        print(f"Error with global service: {e}")
    
    # Method 2: Using context manager (good for isolated operations)  
    print("\n🔒 Method 2: Using context manager")
    try:
        async with clip_service_context() as clip_service:
            health = await clip_service.get_health_status()
            print(f"Context manager service health: {health}")
            
    except Exception as e:
        print(f"Error with context manager: {e}")
    
    # Method 3: Manual service creation (for custom configurations)
    print("\n⚙️ Method 3: Manual service creation")
    try:
        # Create service with custom settings
        custom_service = CLIPService(
            similarity_threshold=0.2,
            max_elements=3
        )
        
        await custom_service.initialize()
        
        health = await custom_service.get_health_status()
        print(f"Custom service health: {health}")
        
        # Clean up
        await custom_service.cleanup()
        
    except Exception as e:
        print(f"Error with custom service: {e}")
    
    print("\n✅ Example completed!")


async def integration_example():
    """Example of integrating with FastAPI dependency injection."""
    
    # This shows how you would use it in a FastAPI endpoint
    from fastapi import Depends
    
    # In your FastAPI app, you would define:
    # async def process_validation(clip_service: CLIPService = Depends(get_clip_service)):
    #     filtered_elements = await clip_service.filter_elements_async(...)
    #     return filtered_elements
    
    print("🔗 FastAPI Integration Example")
    print("In your FastAPI endpoint:")
    print("""
async def validate_ui(
    request: ValidationRequest,
    clip_service: CLIPService = Depends(get_clip_service)
):
    # Filter elements using CLIP
    filtered_elements = await clip_service.filter_elements_async(
        image_path=request.image_path,
        elements=parsed_elements,
        qa_prompt=request.qa_prompt,
        max_elements=request.clip_max_elements
    )
    
    # Continue with GPT processing...
    return ValidationResponse(...)
    """)


async def error_handling_example():
    """Example of proper error handling."""
    
    print("⚠️ Error Handling Example")
    
    try:
        # Using service without initialization
        service = CLIPService()
        # This will raise CLIPServiceError
        health = await service.get_health_status()
        
    except Exception as e:
        print(f"Expected error (service not initialized): {type(e).__name__}: {e}")
    
    try:
        # Using invalid model name
        service = CLIPService(model_name="invalid/model-name")
        await service.initialize()
        
    except Exception as e:
        print(f"Expected error (invalid model): {type(e).__name__}: {e}")


if __name__ == "__main__":
    """Run examples if script is executed directly."""
    asyncio.run(example_usage())
    asyncio.run(integration_example())
    asyncio.run(error_handling_example())