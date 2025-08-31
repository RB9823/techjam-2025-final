#!/usr/bin/env python3
"""
Test script for mock streaming WebSocket endpoint
Validates that the streaming endpoint returns expected progression and data
"""
import asyncio
import json
import websockets
from datetime import datetime


async def test_mock_streaming():
    """Test the mock streaming WebSocket endpoint"""
    uri = "ws://localhost:8000/api/v1/mock-stream"
    
    print("🚀 Testing Mock Streaming WebSocket Endpoint")
    print("=" * 50)
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to {uri}")
            
            # Test each scenario
            scenarios = [
                ("heart_like", "Does the heart turn red when clicked?"),
                ("button_state", "Does the submit button show loading state after clicking?"),
                ("modal_appearance", "Does a confirmation modal appear when delete is clicked?")
            ]
            
            for scenario, qa_prompt in scenarios:
                print(f"\n🧪 Testing scenario: {scenario}")
                print(f"📝 QA Prompt: {qa_prompt}")
                print("-" * 40)
                
                # Send validation request
                request = {
                    "type": "validate",
                    "data": {
                        "scenario": scenario,
                        "qa_prompt": qa_prompt,
                        "before_image_base64": "data:image/png;base64,mock_before_data",
                        "after_image_base64": "data:image/png;base64,mock_after_data"
                    }
                }
                
                await websocket.send(json.dumps(request))
                
                # Collect all streaming messages
                messages = []
                final_result = None
                
                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        message = json.loads(response)
                        messages.append(message)
                        
                        if message["type"] == "progress":
                            progress_data = message["data"]
                            print(f"📊 {progress_data['stage'].title()}: {progress_data['progress_percent']:.0f}% - {progress_data['message']}")
                            
                            # Show detailed data for key stages
                            if progress_data['stage'] == 'parsing' and 'before_elements' in progress_data:
                                print(f"   🔍 Detected {len(progress_data['before_elements'])} UI elements")
                            elif progress_data['stage'] == 'filtering' and 'filtered_elements' in progress_data:
                                print(f"   🎯 Filtered to {len(progress_data['filtered_elements'])} relevant elements")
                            elif progress_data['stage'] == 'analysis' and 'detected_changes' in progress_data:
                                print(f"   🔄 Found {len(progress_data['detected_changes'])} UI changes")
                        
                        elif message["type"] == "result":
                            final_result = message["data"]
                            print(f"\n✅ Final Result: {'PASSED' if final_result['is_valid'] else 'FAILED'}")
                            print(f"🎯 Confidence: {final_result['confidence']:.1%}")
                            print(f"⏱️  Processing Time: {final_result['processing_time_seconds']:.1f}s")
                            break
                            
                        elif message["type"] == "error":
                            print(f"❌ Error: {message['data']['message']}")
                            break
                            
                    except asyncio.TimeoutError:
                        print("⏰ Timeout waiting for response")
                        break
                
                # Validate we received expected progress stages
                progress_stages = [msg["data"]["stage"] for msg in messages if msg["type"] == "progress"]
                expected_stages = ["initialization", "validation", "parsing", "filtering", "analysis", "exceptions", "validation", "compilation", "completed"]
                
                print(f"\n📈 Progress Stages: {len(progress_stages)}/{len(expected_stages)}")
                for stage in expected_stages:
                    status = "✅" if stage in progress_stages else "❌"
                    print(f"   {status} {stage}")
                
                if final_result:
                    print(f"\n📊 Final Stats:")
                    stats = final_result.get("stats", {})
                    print(f"   • Elements Before: {stats.get('before_elements_count', 0)}")
                    print(f"   • Elements After: {stats.get('after_elements_count', 0)}")
                    print(f"   • Changes Detected: {stats.get('changes_detected', 0)}")
                    print(f"   • Debug Assets: {'✅' if 'debug_assets' in stats else '❌'}")
                    print(f"   • Image References: {'✅' if 'image_references' in stats else '❌'}")
                
                print("=" * 50)
                
                # Small delay between scenarios
                await asyncio.sleep(1)
            
            print("\n🎉 All scenarios tested successfully!")
                
    except ConnectionRefusedError:
        print("❌ Connection refused. Make sure the server is running on port 8000")
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    print("🧪 Mock Streaming WebSocket Test")
    print("Start the test server first: python test_mock_server.py")
    print("OR use the lightweight server: python -m uvicorn test_mock_server:app --reload")
    print()
    
    asyncio.run(test_mock_streaming())