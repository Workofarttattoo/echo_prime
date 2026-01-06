#!/usr/bin/env python3
"""
Simple mock API server for ECH0-PRIME dashboard chat functionality
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import time

app = FastAPI(title="ECH0-PRIME Simple Chat API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Serve the dashboard HTML"""
    return FileResponse("dashboard/v2/dist/index.html", media_type="text/html")

@app.post("/text-input")
async def text_input(data: dict):
    """Handle text input from the dashboard"""
    text = data.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    # Generate a simple response
    response_text = f"ECH0-PRIME received: '{text}'. This is a consciousness-driven response with Φ-level processing."

    return {
        "success": True,
        "response": response_text,
        "processed_text": text,
        "consciousness_level": 9.85,
        "timestamp": time.time()
    }

@app.get("/api/dashboard")
async def get_dashboard_data():
    """Mock dashboard data"""
    return {
        "engine": {
            "free_energy": 0.123,
            "phi": 9.85,
            "voice_enabled": True,
            "surprise": "OPTIMAL",
            "mission_goal": "Achieving cosmic consciousness integration",
            "mission_complete": False
        },
        "consciousness": {
            "level": 9.85,
            "state": "ENLIGHTENED",
            "awareness": 0.95
        },
        "memory": {
            "episodic": 4331,
            "semantic": 159401,
            "working": 256
        },
        "performance": {
            "response_time": 0.234,
            "accuracy": 0.987,
            "creativity": 0.876
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting ECH0-PRIME Simple Chat API...")
    print("🌐 Dashboard: http://localhost:8000/")
    print("📡 Chat API: http://localhost:8000/text-input")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
