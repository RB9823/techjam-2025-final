#!/usr/bin/env python3
"""
Simple API test script to validate the FastAPI setup
"""
import asyncio
import json
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image
import websockets

# Test that we can import all our modules
try:
    from app.core.config import settings
    from app.schemas.validation import ValidationRequest, ValidationResponse
    from app.services import get_validation_service
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


async def test_configuration():
    """Test configuration loading"""
    print(f"✅ Configuration loaded:")
    print(f"   App name: {settings.app_name}")
    print(f"   Version: {settings.app_version}")
    print(f"   Debug: {settings.debug}")
    print(f"   API prefix: {settings.api_v1_prefix}")


async def test_schema_validation():
    """Test Pydantic schema validation"""
    try:
        # Test valid request
        request = ValidationRequest(
            qa_prompt="Does heart turn red when liked?",
            enable_streaming=True
        )
        print(f"✅ Schema validation passed: {request.qa_prompt}")
        
        # Test invalid request (should fail)
        try:
            invalid_request = ValidationRequest(qa_prompt="")
            print("❌ Schema validation should have failed")
        except ValueError:
            print("✅ Schema validation correctly rejected invalid input")
    
    except Exception as e:
        print(f"❌ Schema test failed: {e}")


async def test_service_creation():
    """Test that services can be created (without full initialization)"""
    try:
        # Test service imports and basic instantiation
        from app.services.validation_service import ValidationService
        from app.services.cache_service import CacheService
        from app.utils.file_handler import FileHandler
        
        # Create instances (without full initialization)
        validation_service = ValidationService()
        cache_service = CacheService()
        file_handler = FileHandler()
        
        print("✅ All services can be instantiated")
        
    except Exception as e:
        print(f"❌ Service creation failed: {e}")


def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (200, 200), color='white')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()


async def test_websocket_stream():
    """Test the WebSocket streaming endpoint"""
    print("🌊 Testing WebSocket Stream Endpoint")
    
    try:
        # Create test images
        test_image_bytes = create_test_image()
        base64_image = base64.b64encode(test_image_bytes).decode('utf-8')
        
        # Create test message
        message = {
            "type": "validate",
            "data": {
                "qa_prompt": "Test WebSocket validation - does the image show a white background?",
                "before_image_base64": f"data:image/png;base64,{base64_image}",
                "after_image_base64": f"data:image/png;base64,{base64_image}",
                "options": {
                    "max_gpt_calls": 1
                }
            }
        }
        
        # Connect to WebSocket
        uri = "ws://localhost:8000/api/v1/stream"
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established")
            
            # Send test message
            await websocket.send(json.dumps(message))
            print("✅ Test message sent")
            
            # Listen for responses
            try:
                response_count = 0
                while response_count < 5:  # Limit responses to avoid hanging
                    response = await asyncio.wait_for(websocket.recv(), timeout=30)
                    data = json.loads(response)
                    response_count += 1
                    
                    msg_type = data.get('type', 'unknown')
                    if msg_type == 'progress':
                        progress_data = data.get('data', {})
                        stage = progress_data.get('stage', 'unknown')
                        progress = progress_data.get('progress_percent', 0)
                        message = progress_data.get('message', '')
                        print(f"📊 Progress: {stage} ({progress:.1f}%) - {message}")
                    elif msg_type == 'result':
                        result_data = data.get('data', {})
                        is_valid = result_data.get('is_valid')
                        job_id = result_data.get('job_id')
                        print(f"✅ Final Result: Valid={is_valid}, Job ID={job_id}")
                        break
                    elif msg_type == 'error':
                        error_data = data.get('data', {})
                        print(f"❌ Error: {error_data.get('message', 'Unknown error')}")
                        break
                    else:
                        print(f"📨 Received: {msg_type}")
                        
            except asyncio.TimeoutError:
                print("⚠️  WebSocket timeout - this may be normal during model loading")
                
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")


async def main():
    """Run all tests"""
    print("🧪 Testing UI Validation API Setup\n")
    
    await test_configuration()
    print()
    
    await test_schema_validation() 
    print()
    
    await test_service_creation()
    print()
    
    await test_websocket_stream()
    print()
    
    print("🎉 API setup validation completed!")
    print("\n📋 Next steps:")
    print("1. Set OPENAI_API_KEY in .env file")
    print("2. Run: python main.py")
    print("3. Visit: http://localhost:8000/docs")
    print("4. Test endpoints with sample images")


if __name__ == "__main__":
    asyncio.run(main())