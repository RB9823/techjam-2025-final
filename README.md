# AI-Driven UI Consistency Testing

**Automated detection and validation of UI inconsistencies using multimodal AI pipeline**

[![Demo Video](https://img.shields.io/badge/📹_Live_Demo-Watch_Pipeline-blue?style=for-the-badge)](./demo_v2.mp4)

*Click above to watch: Real-time AI pipeline detecting UI transition states across dynamic interfaces*

Modern applications change constantly, and even small visual differences—like a button color or icon inconsistency—can break user trust or regress functionality. Traditional test scripts often miss these subtle inconsistencies, or require heavy manual labeling.

## Problem Statement

Consider validating that a heart icon properly transitions from empty to filled when a user likes a TikTok video. Current VLMs face a critical challenge: the heart overlay sits atop constantly changing video content with dynamic backgrounds, motion blur, and varying lighting conditions. These models struggle to isolate the static UI transition state from temporal video frames, often misidentifying background elements as the target interface component or failing to detect the subtle state change entirely. Traditional automated tests lack the visual understanding to validate these transition states. The challenge is to reliably detect UI state transitions across dynamic interfaces where visual context constantly shifts, requiring robust filtering to separate interface changes from content changes.

## Solution Architecture

Our system processes two key inputs:
- Reference UI flow or test case
- Screenshots of current app state

These inputs pass through a multi-stage AI pipeline designed to detect both obvious and subtle inconsistencies while minimizing computational cost.

## Technical Pipeline

### Stage 1: Element Detection
We extract candidate UI elements using OmniParser and related detection models, identifying potential points of change across the interface.

### Stage 2: Filtering and Suppression  
Non-maximum suppression removes redundant bounding boxes, reducing hundreds of raw detections to approximately 70 key elements for further analysis.

### Stage 3: Semantic Matching
Elements are encoded using CLIP for semantic similarity analysis. We retain the top 20 most relevant matches to the test case, dramatically reducing computational requirements.

### Stage 4: Consensus Validation
We compare "before" and "after" states across multiple models, running a consensus vote to determine whether the UI maintains consistency or represents a regression.

## Performance Benefits

This approach provides several key advantages:

**No Manual Labeling Required**: Unlike traditional pipelines that require extensive training data and manual annotation, our system adapts to new UI patterns automatically.

**Cost and Speed Optimization**: Early-stage filtering dramatically reduces computational requirements. Processing time decreases from 5+ minutes to 15-30 seconds, while API costs drop by approximately 80%.

**Robustness Across Variations**: The consensus voting mechanism adapts to regional design differences and UI variations while maintaining sensitivity to genuine regressions.

## Implementation Details

### Model Integration
The system integrates four specialized models:
- **YOLO**: Object detection for UI element identification
- **Florence-2**: Visual captioning for semantic understanding
- **CLIP**: Semantic similarity encoding for relevance filtering  
- **GPT-4V**: Visual reasoning for final validation decisions

### Architecture
- **FastAPI backend** with async/await for concurrent model execution
- **WebSocket streaming** for real-time progress updates
- **Redis caching** for repeated query optimization
- **Docker deployment** for consistent environment management

## Demonstration

https://github-production-user-asset-6210df.s3.amazonaws.com/63988139/483917485-92380a17-3b13-460a-a9a1-096ce730de6a.mp4

The system processes any UI screenshot through the complete pipeline:

1. **Element Detection**: Upload before/after screenshots to see YOLO identify UI components
2. **Filtering Visualization**: Watch non-maximum suppression reduce detection candidates  
3. **Semantic Analysis**: Observe CLIP encoding filter elements by relevance
4. **Consensus Evaluation**: See multiple models vote on consistency validation

### Quick Start
```bash
git clone https://github.com/shayaansultan/techjam-2025-final.git
cd techjam-2025-final

# Start backend
cd backend && docker-compose up -d

# Start frontend  
cd frontend && pnpm install && pnpm dev

# Demo at http://localhost:3000
```

## Applications

**UI Regression Testing**: Automated detection of visual inconsistencies across app versions without manual test case creation.

**Cross-Platform Validation**: Verify UI consistency across different devices, regions, or user contexts using the same underlying test logic.

**Design System Compliance**: Validate adherence to design systems and style guides at scale, catching deviations that manual review might miss.

## Setup Instructions

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ with pnpm  
- OpenAI API key

### Configuration
```bash
# Clone repository
git clone https://github.com/shayaansultan/techjam-2025-final.git
cd techjam-2025-final/backend

# Configure environment
cp .env.example .env
# Add OPENAI_API_KEY to .env file

# Start services
docker-compose up -d

# Start frontend
cd ../frontend && pnpm install && pnpm dev
```

### Demo Workflow
1. Navigate to http://localhost:3000
2. Upload reference UI screenshot
3. Add transition step with description (e.g., "heart icon changes color when selected")
4. Upload comparison screenshot
5. Observe pipeline stages with real-time bounding box visualization
6. Review consensus validation results

## Performance Metrics

| Metric | Traditional Approach | Our Pipeline |
|--------|---------------------|--------------|
| Processing Time | 5+ minutes | 15-30 seconds |
| API Cost per Validation | $2.40 | $0.48 |
| Manual Labeling Required | Yes | No |
| Concurrent Processing | 1-2 | 100+ |
| Element Type Coverage | Pre-defined | Universal |

## System Components

```
Frontend (Next.js)     Backend (FastAPI)      AI Pipeline
├── Canvas rendering   ├── Async routing      ├── OmniParser
├── WebSocket client   ├── Model management   ├── CLIP encoding  
├── Progress tracking  ├── Cache layer        ├── Consensus voting
└── Result display     └── Health monitoring  └── GPT-4V analysis
```

## Technical Documentation

- **API Documentation**: http://localhost:8000/docs
- **Frontend Demo**: http://localhost:3000  
- **Backend Architecture**: See `backend/README.md`
- **Development Setup**: See individual component READMEs

Built for TechJam 2025 - automated UI consistency testing through multimodal AI.
