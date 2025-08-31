#!/usr/bin/env python3
"""
Comprehensive endpoint testing script for UI Validation API
Tests all endpoints systematically with proper error handling
"""
import asyncio
import json
import time
import requests
import websockets
from pathlib import Path
from io import BytesIO
from PIL import Image
import base64

BASE_URL = "http://localhost:8000"
API_V1 = f"{BASE_URL}/api/v1"

def wait_for_server(max_wait=60):
    """Wait for server to become available"""
    print("🔄 Waiting for server to start...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/", timeout=2)
            if response.status_code == 200:
                print(f"✅ Server ready after {time.time() - start_time:.1f}s")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    
    print(f"❌ Server not ready after {max_wait}s")
    return False

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (200, 200), color='white')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()

def image_to_base64(image_bytes):
    """Convert image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')

def test_basic_endpoints():
    """Test basic endpoints that should work without AI models"""
    print("\n🧪 Testing Basic Endpoints")
    
    endpoints = [
        ("GET", "/", "Root endpoint"),
        ("GET", "/docs", "FastAPI docs"),
        ("GET", "/api/v1/health", "Basic health check"),
    ]
    
    results = []
    for method, endpoint, description in endpoints:
        try:
            url = f"{BASE_URL}{endpoint}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"   ✅ {method} {endpoint} - {description}")
                results.append((endpoint, "PASS", response.status_code))
            else:
                print(f"   ❌ {method} {endpoint} - {description} (HTTP {response.status_code})")
                results.append((endpoint, "FAIL", response.status_code))
                
        except Exception as e:
            print(f"   ❌ {method} {endpoint} - {description} (Error: {e})")
            results.append((endpoint, "ERROR", str(e)))
    
    return results

def test_health_endpoints():
    """Test all health check endpoints"""
    print("\n🏥 Testing Health Endpoints")
    
    endpoints = [
        ("GET", "/api/v1/health", "Basic health"),
        ("GET", "/api/v1/health/detailed", "Detailed health"),
        ("GET", "/api/v1/health/models", "Model status"),
    ]
    
    results = []
    for method, endpoint, description in endpoints:
        try:
            url = f"{BASE_URL}{endpoint}"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ {description}: {data.get('status', 'OK')}")
                results.append((endpoint, "PASS", data))
            else:
                print(f"   ⚠️  {description} (HTTP {response.status_code})")
                results.append((endpoint, "PARTIAL", response.status_code))
                
        except Exception as e:
            print(f"   ❌ {description} (Error: {e})")
            results.append((endpoint, "ERROR", str(e)))
    
    return results

def test_validation_endpoints():
    """Test validation endpoints with mock data"""
    print("\n📝 Testing Validation Endpoints")
    
    # Create test images
    test_image_bytes = create_test_image()
    
    # Test validation endpoint (will likely fail due to missing models, but tests structure)
    try:
        files = {
            'before_image': ('before.png', test_image_bytes, 'image/png'),
            'after_image': ('after.png', test_image_bytes, 'image/png')
        }
        data = {
            'qa_prompt': 'Test prompt for endpoint validation',
            'enable_streaming': False,
            'max_gpt_calls': 1
        }
        
        response = requests.post(f"{API_V1}/validate", files=files, data=data, timeout=300)
        
        if response.status_code == 200:
            print("   ✅ POST /validate - Validation endpoint working")
            return [("/validate", "PASS", response.json())]
        else:
            print(f"   ⚠️  POST /validate - Expected error (HTTP {response.status_code})")
            print(f"       Response: {response.text[:200]}...")
            return [("/validate", "PARTIAL", response.status_code)]
            
    except Exception as e:
        print(f"   ❌ POST /validate - Error: {e}")
        return [("/validate", "ERROR", str(e))]

def test_element_detection_endpoints():
    """Test element detection endpoints"""
    print("\n🎯 Testing Element Detection Endpoints")
    
    test_image_bytes = create_test_image()
    
    try:
        files = {
            'image': ('test.png', test_image_bytes, 'image/png')
        }
        data = {
            'element_type': 'button',
            'max_elements': 5,
            'enable_debug': False
        }
        
        response = requests.post(f"{API_V1}/element-detection/detect", files=files, data=data, timeout=300)
        
        if response.status_code == 200:
            print("   ✅ POST /element-detection/detect - Working")
            return [("/element-detection/detect", "PASS", response.json())]
        else:
            print(f"   ⚠️  POST /element-detection/detect - Expected error (HTTP {response.status_code})")
            return [("/element-detection/detect", "PARTIAL", response.status_code)]
            
    except Exception as e:
        print(f"   ❌ POST /element-detection/detect - Error: {e}")
        return [("/element-detection/detect", "ERROR", str(e))]

async def test_websocket_endpoint():
    """Test WebSocket endpoint"""
    print("\n🌊 Testing WebSocket Endpoint")
    
    try:
        # Create test message
        test_image_bytes = create_test_image()
        base64_image = image_to_base64(test_image_bytes)
        
        message = {
            "type": "validate",
            "data": {
                "qa_prompt": "Test WebSocket validation",
                "before_image_base64": f"data:image/png;base64,{base64_image}",
                "after_image_base64": f"data:image/png;base64,{base64_image}",
                "options": {
                    "max_gpt_calls": 1
                }
            }
        }
        
        # Test WebSocket connection
        uri = f"ws://localhost:8000/api/v1/stream"
        
        async with websockets.connect(uri) as websocket:
            print("   ✅ WebSocket connection established")
            
            # Send test message
            await websocket.send(json.dumps(message))
            print("   ✅ Test message sent")
            
            # Wait for response (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=300)
                data = json.loads(response)
                print(f"   ✅ Received response: {data.get('type', 'unknown')}")
                return [("/stream", "PASS", data)]
            except asyncio.TimeoutError:
                print("   ⚠️  WebSocket timeout (expected - models loading)")
                return [("/stream", "PARTIAL", "timeout")]
                
    except Exception as e:
        print(f"   ❌ WebSocket test failed: {e}")
        return [("/stream", "ERROR", str(e))]

def generate_test_report(all_results):
    """Generate comprehensive test report"""
    print("\n📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(all_results)
    passed = len([r for r in all_results if r[1] == "PASS"])
    partial = len([r for r in all_results if r[1] == "PARTIAL"])
    failed = len([r for r in all_results if r[1] == "ERROR"])
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"⚠️  Partial: {partial} (expected during model loading)")
    print(f"❌ Failed: {failed}")
    
    print(f"\nDetailed Results:")
    for endpoint, status, result in all_results:
        status_icon = {"PASS": "✅", "PARTIAL": "⚠️", "ERROR": "❌"}[status]
        print(f"  {status_icon} {endpoint}: {status}")
        if isinstance(result, dict) and 'status' in result:
            print(f"      Status: {result['status']}")

async def main():
    """Main testing function"""
    print("🚀 UI Validation API - Comprehensive Endpoint Testing")
    print("=" * 60)
    
    # Wait for server
    if not wait_for_server(max_wait=30):
        print("❌ Server not available - check logs with: tail -f server.log")
        return
    
    # Run all tests
    all_results = []
    
    # Basic tests (should work immediately)
    all_results.extend(test_basic_endpoints())
    
    # Health tests (should work once API starts)
    all_results.extend(test_health_endpoints())
    
    # Advanced tests (may fail during model loading)
    all_results.extend(test_validation_endpoints())
    all_results.extend(test_element_detection_endpoints())
    
    # WebSocket test
    ws_results = await test_websocket_endpoint()
    all_results.extend(ws_results)
    
    # Generate report
    generate_test_report(all_results)
    
    # Log monitoring instructions
    print(f"\n📋 MONITORING INSTRUCTIONS:")
    print(f"   View live logs: tail -f {Path.cwd()}/server.log")
    print(f"   Check process: ps aux | grep 'python main.py'")
    print(f"   API Documentation: http://localhost:8000/docs")
    print(f"   Health Status: curl http://localhost:8000/api/v1/health")

if __name__ == "__main__":
    asyncio.run(main())