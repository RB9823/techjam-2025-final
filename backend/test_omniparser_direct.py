#!/usr/bin/env python3
"""
Direct test script for OmniParser service without FastAPI
This isolates the service to test YOLO functionality directly
"""

import asyncio
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.omniparser_service import OmniParserService


def draw_bounding_boxes(image, detections, output_path):
    """Draw bounding boxes on image and save for debugging"""
    
    # Create a copy of the image to draw on
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Try to use a default font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Generate colors for bounding boxes
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", 
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
    ]
    
    print(f"Drawing {len(detections)} bounding boxes...")
    
    for i, detection in enumerate(detections):
        bbox = detection.get('bbox', [])
        confidence = detection.get('confidence', 0)
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            color = colors[i % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label with confidence
            label = f"{i+1}: {confidence:.3f}"
            
            # Calculate text size and position
            try:
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except:
                # Fallback for older PIL versions
                text_width, text_height = draw.textsize(label, font=font)
            
            # Position label above the box, or inside if no space
            label_y = max(y1 - text_height - 5, y1 + 5)
            label_x = x1
            
            # Draw background rectangle for text
            draw.rectangle(
                [label_x, label_y, label_x + text_width + 4, label_y + text_height + 4],
                fill=color,
                outline=color
            )
            
            # Draw text
            draw.text((label_x + 2, label_y + 2), label, fill="white", font=font)
    
    # Save the image with bounding boxes
    img_with_boxes.save(output_path)
    print(f"✅ Debug image saved: {output_path}")
    
    return img_with_boxes


async def test_omniparser_direct():
    """Test OmniParser service directly"""
    
    print("🧪 Testing OmniParser service directly (no FastAPI)")
    print("=" * 50)
    
    # Initialize service
    print("1. Initializing OmniParser service...")
    service = OmniParserService()
    
    # Test with a simple test image
    test_image_path = "test_simple.png"  # Use existing test image
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image {test_image_path} not found!")
        print("Please ensure test_simple.png exists in the current directory")
        return
    
    try:
        # Load image
        print(f"2. Loading test image: {test_image_path}")
        image = Image.open(test_image_path)
        print(f"   Image size: {image.size}")
        print(f"   Image mode: {image.mode}")
        
        # Test UI element detection
        print("3. Testing UI element detection...")
        detections = await service.detect_ui_elements(image)
        
        print(f"✅ Detection completed!")
        print(f"   Found {len(detections)} UI elements")
        
        # Save debug image with bounding boxes
        if detections:
            debug_output_path = f"debug_detections_{test_image_path.replace('.png', '_annotated.png')}"
            draw_bounding_boxes(image, detections, debug_output_path)
            
            print("\n📋 Detection Results (Top 10):")
            # Show top 10 detections sorted by confidence
            sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)[:10]
            for i, detection in enumerate(sorted_detections):
                print(f"   Element {i+1}:")
                print(f"     - ID: {detection.get('id', 'N/A')}")
                print(f"     - BBox: {detection.get('bbox', 'N/A')}")
                print(f"     - Confidence: {detection.get('confidence', 'N/A')}")
                print(f"     - Area: {detection.get('area', 'N/A')}")
            
            if len(detections) > 10:
                print(f"   ... and {len(detections) - 10} more elements")
        else:
            print("   No UI elements detected")
        
        # Test full analysis
        print("\n4. Testing full screenshot analysis...")
        result = await service.analyze_screenshot(image)
        
        print(f"✅ Full analysis completed!")
        print(f"   Elements: {len(result.get('elements', []))}")
        print(f"   Has structure: {'structure' in result}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 Direct test completed successfully!")
    print("This confirms the OmniParser service works outside of FastAPI")


async def test_yolo_specifically():
    """Test just the YOLO detection part"""
    
    print("\n🎯 Testing YOLO detection specifically")
    print("=" * 40)
    
    service = OmniParserService()
    
    # Load YOLO model explicitly
    try:
        print("1. Loading YOLO model...")
        await service._load_yolo_model()
        print("✅ YOLO model loaded successfully!")
        
        # Test image
        test_image_path = "test_simple.png"
        if not Path(test_image_path).exists():
            print(f"❌ Test image {test_image_path} not found!")
            return
        
        image = Image.open(test_image_path)
        print(f"2. Running YOLO detection on {image.size} image...")
        
        # Run detection directly
        results = await asyncio.get_event_loop().run_in_executor(
            service.thread_pool,
            service._run_yolo_detection,
            image
        )
        
        print("✅ YOLO detection completed!")
        print(f"   Raw results type: {type(results)}")
        
        # Extract detections
        detections = service._extract_detections_from_results(results)
        print(f"   Extracted {len(detections)} detections")
        
        # Save debug image for YOLO-specific test
        if detections:
            debug_output_path = f"debug_yolo_only_{test_image_path.replace('.png', '_annotated.png')}"
            draw_bounding_boxes(image, detections, debug_output_path)
            
            # Show top 5 detections
            sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)[:5]
            for i, det in enumerate(sorted_detections):
                print(f"   Detection {i+1}: {det}")
            
            if len(detections) > 5:
                print(f"   ... and {len(detections) - 5} more detections")
        else:
            print("   No detections found")
        
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Starting direct OmniParser tests...")
    
    # Run the tests
    asyncio.run(test_omniparser_direct())
    asyncio.run(test_yolo_specifically())
    
    print("\n✨ All tests completed!")