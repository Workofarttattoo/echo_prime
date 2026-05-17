"""
REST API Server for Byte-Level Inference

Fast, production-ready API for byte-level models

Endpoints:
- POST /generate - Generate text
- POST /generate/stream - Stream generation
- GET /health - Health check
- GET /models - List available models
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncIterator
import uvicorn
import sys
from pathlib import Path
import json

# Add inference to path
sys.path.append(str(Path(__file__).parent.parent / "inference"))
sys.path.append(str(Path(__file__).parent.parent / "models"))

from engine import ByteInferenceEngine, GenerationConfig


# API Models
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    bytes_generated: int
    time_seconds: float
    bytes_per_second: float


class HealthResponse(BaseModel):
    status: str
    model_type: str
    model_size: str
    device: str


# Initialize FastAPI
app = FastAPI(
    title="Byte-Level Inference API",
    description="Non-tokenized AI inference platform",
    version="1.0.0"
)

# Global inference engine (loaded on startup)
engine: Optional[ByteInferenceEngine] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global engine

    print("🚀 Loading inference engine...")

    engine = ByteInferenceEngine(
        model_type="transformer",  # or "mamba"
        model_size="small",
        device="cuda"
    )

    print("✅ Inference engine ready!")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy",
        model_type=engine.model_type,
        model_size=engine.model_size,
        device=str(engine.device)
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from prompt

    Example:
    ```
    curl -X POST http://localhost:8000/generate \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "The future of AI is", "max_new_tokens": 100}'
    ```
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    config = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=False
    )

    try:
        result = engine.generate(request.prompt, config)

        return GenerateResponse(
            text=result['text'],
            prompt=result['prompt'],
            bytes_generated=result['bytes_generated'],
            time_seconds=result['time_seconds'],
            bytes_per_second=result['bytes_per_second']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    Stream generation token by token

    Example:
    ```
    curl -X POST http://localhost:8000/generate/stream \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "Once upon a time", "stream": true}'
    ```
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    config = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=True
    )

    async def event_stream() -> AsyncIterator[str]:
        """Stream events as server-sent events"""
        prompt_bytes = engine.encode(request.prompt)

        for chunk in engine._generate_stream(prompt_bytes, config):
            # Format as SSE
            data = json.dumps(chunk)
            yield f"data: {data}\n\n"

        # Send done signal
        yield "data: {\"done\": true}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )


@app.get("/models")
async def list_models():
    """List available model configurations"""
    return {
        "available_models": [
            {"type": "transformer", "size": "tiny"},
            {"type": "transformer", "size": "small"},
            {"type": "transformer", "size": "medium"},
            {"type": "transformer", "size": "large"},
            {"type": "mamba", "size": "tiny"},
            {"type": "mamba", "size": "small"},
            {"type": "mamba", "size": "medium"},
            {"type": "mamba", "size": "large"},
        ],
        "current_model": {
            "type": engine.model_type if engine else None,
            "size": engine.model_size if engine else None
        }
    }


def main():
    """Run the API server"""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║  Byte-Level Inference API Server                     ║
    ║                                                       ║
    ║  Endpoints:                                          ║
    ║    POST /generate        - Generate text             ║
    ║    POST /generate/stream - Stream generation         ║
    ║    GET  /health          - Health check              ║
    ║    GET  /models          - List models               ║
    ║                                                       ║
    ║  Docs: http://localhost:8000/docs                    ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
