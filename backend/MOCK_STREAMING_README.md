# Mock Streaming API Documentation

🚀 **Comprehensive mock streaming implementation for UI validation pipeline demonstration**

## Overview

This mock streaming implementation provides realistic AI pipeline progression with detailed mock data, perfect for frontend integration testing and demos **without requiring heavy AI model downloads**.

## Features

✨ **3 Realistic Scenarios**
- ❤️ **Heart Like Animation**: Social media like button interaction
- 🔄 **Button State Change**: Form submission loading states  
- 📝 **Modal Dialog Appearance**: Confirmation modal workflows

🎯 **Comprehensive Mock Data**
- Detailed UI element detection with bounding boxes
- Professional QA validation reasoning
- Realistic AI confidence scores and processing times
- Debug assets and image reference metadata
- Complete pipeline stage progression (9 stages)

📊 **Rich Progress Streaming**
- Real-time progress updates (0-100%)
- Stage-specific metadata and analysis
- Visual asset references for each detected element
- Performance metrics and model statistics

## Quick Start

### 1. Start the Test Server (No AI models required!)

```bash
cd backend
python test_mock_server.py
```

This starts a lightweight FastAPI server on port 8000 without any AI dependencies.

### 2. Test Options

**Option A: Web Interface (Recommended)**
```bash
# Open browser to: http://localhost:8000/test
```
- Beautiful interactive UI
- Real-time progress visualization
- Click scenarios to test streaming

**Option B: Python Client**
```bash
python test_mock_streaming.py
```
- Command-line WebSocket client
- Automated testing of all scenarios
- Detailed progress logging

**Option C: Direct WebSocket Connection**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/mock-stream');
```

## WebSocket Message Format

### Request Message
```json
{
  "type": "validate",
  "data": {
    "scenario": "heart_like|button_state|modal_appearance",
    "qa_prompt": "Does heart turn red when liked?",
    "before_image_base64": "data:image/png;base64,...",
    "after_image_base64": "data:image/png;base64,..."
  }
}
```

### Progress Messages
```json
{
  "job_id": "uuid",
  "type": "progress", 
  "data": {
    "stage": "parsing",
    "progress_percent": 10.0,
    "message": "Detecting UI elements with OmniParser",
    "details": "Using advanced computer vision...",
    "elements_detected": 6,
    "before_elements": [...],
    "debug_assets": {...},
    "image_references": {...}
  }
}
```

### Final Result Message
```json
{
  "job_id": "uuid",
  "type": "result",
  "data": {
    "is_valid": true,
    "confidence": 0.94,
    "reasoning": "✅ VALIDATION PASSED: Heart Like Functionality...",
    "detected_changes": [...],
    "before_elements": [...],
    "after_elements": [...],
    "stats": {
      "scenario": "heart_like",
      "processing_time": "12.3s",
      "debug_assets": {...},
      "ai_pipeline_stats": {...}
    }
  }
}
```

## Pipeline Stages

The mock streaming progresses through 9 realistic AI processing stages:

1. **Initialization** (0%) - Loading AI models and setup
2. **Validation** (5%) - Input image validation  
3. **Parsing** (10%) - OmniParser UI element detection
4. **Filtering** (40%) - CLIP semantic relevance filtering
5. **Analysis** (60%) - UI state change detection
6. **Exceptions** (70%) - Anomaly and error detection
7. **Validation** (80%) - GPT professional QA reasoning  
8. **Compilation** (95%) - Final report generation
9. **Completed** (100%) - Results ready

Each stage includes realistic processing delays (1.5-3.5s) and detailed metadata.

## Scenario Details

### ❤️ Heart Like Animation
- **Query**: "Does heart turn red when clicked?"
- **Elements**: 6 UI elements (nav, post, heart, counter, etc.)
- **Changes**: Heart icon state change (gray → red), like count increment
- **Result**: ✅ PASSED with 94% confidence

### 🔄 Button State Change  
- **Query**: "Does submit button show loading state?"
- **Elements**: 6 UI elements (form, inputs, buttons)
- **Changes**: Button state (enabled → loading), spinner appearance
- **Result**: ✅ PASSED with 91% confidence

### 📝 Modal Dialog Appearance
- **Query**: "Does confirmation modal appear?"  
- **Elements**: 5 base elements + 6 new modal elements
- **Changes**: Modal backdrop and dialog appearance
- **Result**: ✅ PASSED with 97% confidence

## Integration with Frontend

### React/NextJS Integration

```typescript
interface StreamMessage {
  job_id: string;
  type: 'progress' | 'result' | 'error';
  data: any;
}

const useValidationStream = () => {
  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState('');
  const [result, setResult] = useState(null);
  
  const startValidation = (scenario: string) => {
    const ws = new WebSocket('ws://localhost:8000/api/v1/mock-stream');
    
    ws.onmessage = (event) => {
      const message: StreamMessage = JSON.parse(event.data);
      
      if (message.type === 'progress') {
        setProgress(message.data.progress_percent);
        setCurrentStage(message.data.stage);
      } else if (message.type === 'result') {
        setResult(message.data);
      }
    };
    
    ws.send(JSON.stringify({
      type: 'validate',
      data: { scenario, qa_prompt: 'Test query' }
    }));
  };
  
  return { progress, currentStage, result, startValidation };
};
```

### Vue Integration

```vue
<template>
  <div>
    <div class="progress-bar">
      <div :style="{width: progress + '%'}" class="progress-fill">
        {{ Math.round(progress) }}%
      </div>
    </div>
    <div>Stage: {{ currentStage }}</div>
    <button @click="startValidation('heart_like')">Test Heart Like</button>
  </div>
</template>

<script>
export default {
  data() {
    return { progress: 0, currentStage: '', ws: null };
  },
  methods: {
    startValidation(scenario) {
      this.ws = new WebSocket('ws://localhost:8000/api/v1/mock-stream');
      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'progress') {
          this.progress = message.data.progress_percent;
          this.currentStage = message.data.stage;
        }
      };
      this.ws.send(JSON.stringify({
        type: 'validate',
        data: { scenario }
      }));
    }
  }
};
</script>
```

## Mock Data Assets

### Debug Assets Structure
```json
{
  "session_id": "a1b2c3d4",
  "debug_enabled": true,
  "crop_count": 6,
  "crops": [
    {
      "element_id": "heart_icon_unfilled",
      "crop_filename": "crop_000_heart_icon_unfilled.png",
      "crop_url": "/api/v1/mock-stream/debug/a1b2c3d4/crop/crop_000_heart_icon_unfilled.png",
      "bbox": {"x": 320, "y": 540, "width": 32, "height": 32},
      "confidence": 0.98,
      "detection_method": "gpt4v"
    }
  ]
}
```

### Image References  
```json
{
  "before_image_url": "/api/v1/mock-stream/images/a1b2c3d4/before_screenshot.png",
  "after_image_url": "/api/v1/mock-stream/images/a1b2c3d4/after_screenshot.png",
  "comparison_overlay": "/api/v1/mock-stream/images/a1b2c3d4/comparison_overlay.png",
  "heatmap_visualization": "/api/v1/mock-stream/images/a1b2c3d4/attention_heatmap.png"
}
```

## Production Integration

To integrate this with your production backend:

1. **Import the mock router**:
   ```python
   from app.api.v1.mock_streaming import router as mock_streaming_router
   app.include_router(mock_streaming_router, prefix="/api/v1")
   ```

2. **Use the same message format** for your real WebSocket endpoint

3. **Replace mock functions** with real AI service calls:
   - `get_mock_ui_elements_*()` → Real OmniParser detection
   - `get_mock_validation_reasoning()` → Real GPT validation  
   - `get_mock_detected_changes()` → Real change detection

## Performance Notes

- ⚡ **Lightweight**: No AI model downloads required (runs instantly)
- 🔄 **Realistic Timing**: 1.5-3.5s delays between stages (configurable)
- 📊 **Rich Data**: Comprehensive mock data for thorough frontend testing
- 🎯 **Production Ready**: Same message format as real implementation

## Troubleshooting

**Connection Issues**:
```bash
# Check if server is running
curl http://localhost:8000/

# Test WebSocket connection
python test_mock_streaming.py
```

**CORS Issues**:
- The test server includes CORS middleware for all origins
- For production, configure CORS appropriately

**Missing Dependencies**:
```bash
pip install fastapi uvicorn websockets
```

---

🎉 **Ready to integrate with your frontend!** The mock streaming provides everything needed for beautiful progress visualization and realistic AI pipeline demonstration.