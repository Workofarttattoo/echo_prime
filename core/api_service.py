import asyncio
import json
import socket
from pathlib import Path
from typing import List, Optional, Dict, Mapping, cast

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn
import os

app = FastAPI()

# Enable CORS for the React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the dashboard static files
dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")
if os.path.exists(dashboard_path):
    app.mount("/dashboard", StaticFiles(directory=dashboard_path, html=True), name="dashboard")
    print(f"✅ Dashboard served at http://localhost:8000/dashboard")

@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard/")

class ConnectionManager:
    def __init__(self):  # pyright: ignore[reportMissingSuperCall]
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # Connection might be dead
                pass

manager = ConnectionManager()

# Global state storage
current_state = {
    "engine": {"free_energy": 0, "surprise": "Initializing neural levels...", "mission_goal": "Synchronizing...", "mission_complete": False, "phi": 0.0},
    "attention": {"coherence": 0, "frequency": "0Hz"},
    "reasoning": {"insight": "System awakening...", "actions": []},
    "memory": {"episodic_count": 0, "semantic_concepts": 0, "recent_notes": []},
    "sensory": {"active_visual": None, "audio_input_detected": False}
}
voice_enabled = True
speech_input_queue = None # Set by main_orchestrator
event_loop = None # Set by start_api_server

@app.post("/speech")
async def receive_speech(data: Mapping[str, object]):
    """Receives transcribed text from the dashboard browser."""
    text = str(data.get("text", ""))
    if text and speech_input_queue is not None:
        speech_input_queue.put(text)
        return {"status": "received", "text": text}
    return {"status": "ignored"}

@app.post("/text-input")
async def receive_text_input(data: Mapping[str, object]):
    """Receives text input from the dashboard GUI text window."""
    text = str(data.get("text", ""))
    if text and speech_input_queue is not None:
        # Use the same queue as speech input for processing
        speech_input_queue.put(text)
        return {"status": "received", "text": text}
    return {"status": "ignored", "error": "Queue not initialized"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send current state immediately on connect
        if current_state:
            await websocket.send_text(json.dumps(current_state))
        while True:
            # Keep connection open
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def push_state_to_ui(state: Dict[str, object]):
    """Directly updates the state and broadcasts via WebSocket without HTTP overhead."""
    global current_state, event_loop
    current_state = state
    # Use the stored loop if available
    if event_loop and event_loop.is_running():
        _fut = asyncio.run_coroutine_threadsafe(manager.broadcast(json.dumps(state)), event_loop)
    else:
        # Fallback if loop isn't set yet (startup race)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                _fut = asyncio.run_coroutine_threadsafe(manager.broadcast(json.dumps(state)), loop)
        except Exception:
            pass

@app.post("/update_state")
async def update_state(state: Dict[str, object]):
    """Legacy HTTP support for external tools."""
    push_state_to_ui(state)
    return {"status": "success"}

@app.post("/mute")
async def set_mute(data: Mapping[str, object]):
    global voice_enabled
    voice_enabled = not data.get("mute", False)
    # Broadcast a small control message or just wait for next state update
    # Better to broadcast a control message so UI updates instantly
    await manager.broadcast(json.dumps({"control": {"voice_enabled": voice_enabled}}))
    return {"voice_enabled": voice_enabled}

@app.get("/voice_status")
async def get_voice_status():
    return {"voice_enabled": voice_enabled}

@app.get("/health")
async def health():
    return {"status": "healthy"}

def _find_open_port(preferred_port: Optional[int] = None) -> int:
    """
    Find an available TCP port.
    Tries the preferred_port first, then falls back to an ephemeral port.
    """
    # Try preferred port first
    if preferred_port is not None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", preferred_port))
                return cast(int, s.getsockname()[1])
            except OSError:
                pass  # Will fall back to ephemeral

    # Get an ephemeral free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return cast(int, s.getsockname()[1])


def start_api_server(port: Optional[int] = None) -> None:
    global event_loop
    try:
        event_loop = asyncio.get_event_loop()
    except:
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

    chosen_port = _find_open_port(port)
    print(f"🌐 Starting API server on port {chosen_port}")

    # Persist port info so the dashboard can auto-connect
    try:
        ws_url = f"ws://localhost:{chosen_port}/ws"
        port_info = {"port": chosen_port, "ws_url": ws_url}

        legacy_path = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "api_port.json"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        _ = legacy_path.write_text(json.dumps(port_info, indent=2))

        v2_path = Path(__file__).resolve().parent.parent / "dashboard" / "v2" / "public" / "data" / "api_port.json"
        v2_path.parent.mkdir(parents=True, exist_ok=True)
        _ = v2_path.write_text(json.dumps(port_info, indent=2))
    except Exception as port_err:
        print(f"⚠️ Could not persist port info: {port_err}")

    config = uvicorn.Config(app, host="0.0.0.0", port=chosen_port, loop="asyncio")
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    start_api_server()
