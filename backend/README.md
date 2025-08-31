# UI Validation API - Production Backend

A production-ready FastAPI backend for AI-powered UI change validation using a sophisticated pipeline of computer vision and large language models.

## Table of Contents

- [UI Validation API - Production Backend](#ui-validation-api---production-backend)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Architecture and Workflow](#architecture-and-workflow)
    - [Architecture Diagram](#architecture-diagram)
    - [Workflow Stages](#workflow-stages)
  - [Design Reasoning](#design-reasoning)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Local Development Setup](#local-development-setup)
  - [Running the Application](#running-the-application)
    - [Local Development](#local-development)
    - [Docker Deployment](#docker-deployment)
  - [API Endpoints](#api-endpoints)
    - [Core Endpoints](#core-endpoints)
    - [Monitoring](#monitoring)
  - [API Usage](#api-usage)
    - [WebSocket Streaming](#websocket-streaming)
  - [Testing](#testing)
  - [Code Quality](#code-quality)
  - [Configuration](#configuration)
  - [Production Deployment](#production-deployment)
    - [Hardware Requirements](#hardware-requirements)
    - [Scaling Considerations](#scaling-considerations)
  - [Support](#support)

## Features

- **Advanced AI Pipeline**: Orchestrates multiple AI models for highly nuanced UI understanding.
- **Multi-Modal Element Detection**: Uses `YOLO` for broad object detection and `Florence-2` for fine-grained visual captioning.
- **Targeted Analysis with GPT-4 Vision**: Intelligently parses user prompts to identify a "region of interest" and uses GPT-4V to precisely locate and validate specific elements (e.g., "the heart icon").
- **Semantic Filtering**: Employs `CLIP` to semantically filter detected UI elements based on the user's query, focusing the analysis on what matters.
- **Professional QA Validation**: Leverages `GPT-4` to provide detailed, professional-grade reasoning and a final verdict on whether the UI change is valid.
- **Real-time Streaming**: Provides progress updates over `WebSockets` for a responsive user experience.
- **Production Ready**: Built with robust error handling, structured `JSON` logging, performance monitoring, and caching.

## Architecture and Workflow

### Architecture Diagram

```
+------------------------+      +---------------------------+      +---------------------------------------------+
|      FastAPI App       |      |    Validation Service     |      |                 AI Pipeline                 |
|------------------------|      |---------------------------|      |---------------------------------------------|
| - WebSocket Endpoint   |<---->| - Pipeline Orchestration  |<---->| 1. GPT-4 (Prompt Parsing)                   |
| - API Routers          |      | - Async Task Management   |      | 2. OmniParser (YOLO + Florence-2)           |
| - Pydantic Schemas     |      | - Progress Tracking       |      | 3. CLIP (Semantic Filtering)                |
+------------------------+      +---------------------------+      | 4. GPT-4V (Targeted Element Validation)     |
        |                                                            +---------------------------------------------+
        |
+------------------------+
|      Redis Cache       |
|------------------------|
| - Caching AI Results   |
| - GPT Caption Cache    |
+------------------------+
```

### Workflow Stages

The validation process is a multi-stage pipeline orchestrated by the `ValidationService`:

1.  **Prompt Parsing**: The user's `qa_prompt` is first analyzed by GPT-4 to extract the specific **region of interest** (e.g., "heart icon") and the **expected change** (e.g., "turns red").
2.  **UI Element Detection**: The `OmniParserService` analyzes the *before* and *after* images. It uses a `YOLO` model to detect all potential UI elements and their bounding boxes.
3.  **Semantic Filtering**: If the number of detected elements is large, the `CLIPService` filters them based on semantic similarity to the user's prompt or the extracted region of interest, drastically narrowing down the search space.
4.  **Targeted GPT-4V Validation**: If a specific "region of interest" was identified, the service enters a powerful sub-workflow:
    *   It crops the filtered elements from both images.
    *   It sends these cropped images to `GPT-4V` in parallel to ask: "Is this a {region\_of\_interest}?".
    *   If multiple candidates are found, it uses GPT-4V again to select the *best* match from each image.
    *   Finally, it performs a visual comparison of the definitive *before* and *after* elements to validate the change, providing highly accurate and context-aware results.
5.  **GPT-4 QA Reasoning**: The final set of relevant elements and detected changes are passed to the `GPTService`. It generates a professional, text-based QA validation, including a final `is_valid` verdict and detailed `reasoning`.
6.  **Response**: The complete `ValidationResponse`, including the verdict, reasoning, and detected elements, is returned to the user.

## Design Reasoning

Our design philosophy centered on building a robust, accurate, and extensible UI validation pipeline. The current architecture is the result of iterative development and deliberate choices based on empirical results.

-   **Initial Approach (VL-CLIP)**: We began with a modified Vision-Language CLIP model, aiming for a single, elegant solution to interpret the user's prompt and directly identify changes in the images. While promising, this approach struggled with the nuance required for precise UI element localization and state change validation (ie "did the icon turn red?"). It provided general understanding but lacked the specificity needed for QA-grade assertions.

-   **Pivoting to a Multi-Stage Pipeline**: To achieve higher accuracy, we moved to a multi-stage pipeline, separating element detection from semantic understanding and validation.
    -   **Element Detection (GroundingDINO vs. OmniParser)**: Our first choice for element detection was GroundingDINO, a powerful open-set object detector. However, we found that crafting effective text prompts to reliably detect a wide variety of UI elements was challenging and inconsistent. To solve this, we transitioned to the **OmniParser** model, which combines a traditional object detector (YOLO) with a vision-language model for captioning. This initially produced a very large number of overlapping bounding boxes (often over 300 per image). To manage this, we implemented **Non-Maximum Suppression (NMS)**, which intelligently merges overlapping boxes and filters redundant detections. This step was critical, drastically reducing the number of candidate boxes (e.g., from ~320 to ~76) and allowing subsequent pipeline stages to perform more efficiently and accurately.
    -   **LLM-based Validation (Florence-v2 vs. GPT-4o/Claude 3.5)**: We initially used the Florence-v2 model for its visual question-answering capabilities. However, we encountered limitations in its ability to perform the detailed, comparative reasoning required for our final validation step. We therefore switched to more powerful, general-purpose vision models like **GPT-4o** and **Anthropic's Claude 3.5 Sonnet** (as configured in `app/core/config.py`). These models provide the high-level reasoning necessary to compare two elements and deliver a definitive QA verdict.

-   **Exploring and Discarding Alternative Paths**: We rigorously tested several other computer vision techniques that ultimately proved unsuitable for this specific problem:
    -   **Image Subtraction & Perceptual Hashing (pHash)**: While effective for detecting global image differences, these methods were too brittle. They are highly sensitive to minor, irrelevant shifts in layout, rendering, or anti-aliasing, leading to a high rate of false positives. They cannot distinguish between a significant UI change and a trivial pixel shift.
    -   **Depth and Noise-Based Segmentation**: We explored using depth or image noise patterns to segment UI elements. These approaches were inconsistent across different UI styles and failed to provide semantically meaningful boundaries for components.
    -   **Merkle Trees**: We considered using Merkle trees to create a structural hash of the UI layout. This idea was discarded as it would require a level of access to the application's internal component tree (like the DOM), which is not available when working directly from images.

-   **Final Architectural Choices**:
    -   **FastAPI & Asynchronous Operations**: Chosen for its high performance and native `asyncio` support, which is essential for orchestrating concurrent calls to multiple AI services and ensuring the API remains responsive.
    -   **Microservices-Oriented Structure**: Separating logic into distinct services (`ValidationService`, `OmniParserService`, etc.) makes the system modular, testable, and easier to maintain and scale.

## Installation

### Prerequisites

-   Python 3.12
-   [uv](https://github.com/astral-sh/uv) (for fast package management)
-   Docker and Docker Compose

### Local Development Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd backend
    ```

2.  **Set up the environment:**
    Copy the example environment file and update it with your API keys.
    ```bash
    cp .env.example .env
    ```
    You **must** add your `OPENAI_API_KEY` to the `.env` file.

3.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv --python 3.12
    source .venv/bin/activate
    uv sync
    ```

## Running the Application

### Local Development

To run the application with hot-reloading enabled:

```bash
uv run python main.py
```

The API will be available at `http://localhost:8000`.

### Docker Deployment

For a production-like environment, use Docker Compose. This will start the FastAPI backend and the Redis cache.

```bash
docker-compose up -d --build
```

To view the application logs:

```bash
docker-compose logs -f backend
```

*Note: The `worker` and `flower` services in `docker-compose.yml` are defined but not currently used by the application, as it relies on `asyncio` for concurrency.*

## API Endpoints

### Core Endpoints

-   `WebSocket /api/v1/stream`: The primary endpoint for real-time streaming of validation progress and results.
-   `POST /api/v1/validate`: (Not Recommended for UI) A standard HTTP endpoint for validation. Less responsive than the WebSocket.
-   `GET /docs`: Access the interactive API documentation (Swagger UI).

### Monitoring

-   `GET /`: Get basic information about the API.
-   `GET /api/v1/health`: Perform a basic health check.
-   `GET /api/v1/health/detailed`: Get a detailed health report of the API and its dependent AI services.
-   `GET /api/v1/health/models`: Check the loading status of the AI models.

## API Usage

### WebSocket Streaming

The recommended way to interact with the API is via WebSockets for a real-time experience.

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/stream');

ws.onopen = () => {
  console.log('WebSocket connection established.');
  ws.send(JSON.stringify({
    type: 'validate',
    data: {
      qa_prompt: 'Does the heart icon turn red when an item is liked?',
      before_image_base64: 'data:image/png;base64,...', // base64 encoded image
      after_image_base64: 'data:image/png;base64,...'   // base64 encoded image
    }
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === 'progress') {
    console.log(`[${message.data.progress_percent}%] ${message.data.stage}: ${message.data.message}`);
  } else if (message.type === 'result') {
    console.log('Validation Complete:', message.data);
    ws.close();
  } else if (message.type === 'error') {
    console.error('Validation Failed:', message.data);
    ws.close();
  }
};

ws.onclose = () => {
  console.log('WebSocket connection closed.');
};
```

## Testing

1.  **Install development dependencies:**
    ```bash
    uv sync --group dev
    ```

2.  **Run the test suite:**
    ```bash
    uv run pytest
    ```

## Code Quality

This project uses several tools to maintain code quality:

-   **Formatting**: `black` and `isort`
-   **Type Checking**: `mypy`
-   **Linting**: `ruff`

To run the checks:

```bash
uv run black app/
uv run isort app/
uv run mypy app/
uv run ruff check app/
```

## Configuration

Key environment variables are defined in `.env`:

-   `OPENAI_API_KEY`: **Required**. Your OpenAI API key.
-   `USE_GPU`: Set to `true` to enable GPU acceleration for local AI models.
-   `DETECTION_CONFIDENCE`: Confidence threshold for YOLO object detection (e.g., `0.1`).
-   `CLIP_MAX_ELEMENTS`: The max number of elements to pass to the final validation stage after CLIP filtering.
-   `MAX_GPT_CALLS`: Rate limiting for GPT-4V calls during the element captioning phase.
-   `CACHE_TTL`: Time-to-live for cached results in Redis (in seconds).
-   `ENABLE_DEBUG_CROPS`: Set to `true` to save debug image crops to the `debug_output` directory.
-   `REDIS_URL`: The URL for the Redis instance.
-   `ALLOWED_ORIGINS`: A comma-separated list of allowed CORS origins for the API.

## Production Deployment

### Hardware Requirements

-   **CPU**: 4+ cores recommended.
-   **RAM**: 16GB+ is strongly recommended, as the AI models are memory-intensive.
-   **GPU**: An NVIDIA GPU with at least 8GB of VRAM is highly recommended for acceptable inference performance.
-   **Storage**: 20GB+ for the application, dependencies, and AI model cache.

## Scaling Considerations

-   **Workers**: Increase the number of Uvicorn workers to handle more concurrent requests.
-   **Redis**: For high availability, consider a managed Redis service or a Redis cluster.
-   **Model Serving**: For very high throughput, consider deploying the AI models as separate, dedicated microservices using a framework like NVIDIA Triton Inference Server. This would allow the core validation service to scale independently of the GPU-intensive model inference.

## Support

If you encounter issues:

1.  Refer to the interactive API documentation at `/docs`.
2.  Review the application logs for detailed error information.
3.  Use the `/api/v1/health/models` endpoint to verify that all AI models have been initialized correctly.
4.  Ensure your `OPENAI_API_KEY` is valid and has sufficient credits.