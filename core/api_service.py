import asyncio
import json
import socket
from pathlib import Path
from typing import List, Optional, Dict, Mapping, cast, Any
from queue import Queue

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn
import os
import time
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global event_loop
    event_loop = asyncio.get_running_loop()
    yield

app = FastAPI(lifespan=lifespan)

# Enable CORS for the React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mounting static files for vision feed
sensory_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sensory_input")
if not os.path.exists(sensory_path):
    os.makedirs(sensory_path)
app.mount("/sensory", StaticFiles(directory=sensory_path), name="sensory")

# Serve the modern v2 dashboard as the primary interface
dashboard_v2_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "v2", "dist")
legacy_dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")

if os.path.exists(dashboard_v2_path):
    app.mount("/dashboard", StaticFiles(directory=dashboard_v2_path, html=True), name="dashboard")
    
    # Mount legacy dashboard at /v1
    if os.path.exists(legacy_dashboard_path):
        app.mount("/v1", StaticFiles(directory=legacy_dashboard_path, html=True), name="v1")
elif os.path.exists(legacy_dashboard_path):
    app.mount("/dashboard", StaticFiles(directory=legacy_dashboard_path, html=True), name="dashboard")

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
# Main orchestrator will replace this with its transcription queue; we keep a fallback to avoid 500s
speech_input_queue: Optional[Queue] = None
pending_level: Optional[int] = None
agi_instance = None # Set by main_orchestrator
event_loop = None # Set by start_api_server

@app.post("/set_level")
async def set_level(data: Mapping[str, object]):
    """Sets the operational level of the AGI."""
    global agi_instance
    level = data.get("level")
    if level is not None and agi_instance:
        try:
            level_int = int(level)
            agi_instance.set_operational_level(level_int)
            return {"status": "success", "level": level_int}
        except (ValueError, TypeError):
            return {"status": "error", "message": "Invalid level format"}
    return {"status": "error", "message": "AGI instance not connected or level missing"}

@app.post("/ingest_wisdom")
async def ingest_wisdom():
    """Triggers the wisdom ingestion process from the external drive."""
    import subprocess
    try:
        # Run the ingestion script as a separate process
        process = subprocess.Popen([sys.executable, "simple_wisdom_ingest.py"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        return {"status": "started", "message": "Ingestion process initiated in background."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/process_wisdom")
async def process_wisdom():
    """Triggers the cognitive integration of ingested wisdom."""
    import subprocess
    try:
        process = subprocess.Popen([sys.executable, "wisdom_processor.py"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        return {"status": "started", "message": "Cognitive integration initiated in background."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/speech")
async def receive_speech(data: Mapping[str, object]):
    """Receives transcribed text from the dashboard browser."""
    text = str(data.get("text", ""))
    if text and speech_input_queue is not None:
        speech_input_queue.put(text)
        return {"status": "received", "text": text}
    return {"status": "ignored", "error": "queue_not_ready"}

async def broadcast_state(state: Dict[str, Any]):
    """Broadcasts the state to all connected WebSocket clients."""
    await manager.broadcast(json.dumps(state))

def push_state_to_ui(state: Dict[str, object]):
    """Directly updates the state and broadcasts via WebSocket without HTTP overhead."""
    global current_state, event_loop
    current_state = state
    
    if event_loop:
        try:
            # Check if we are already in the event loop
            try:
                running_loop = asyncio.get_running_loop()
                if running_loop == event_loop:
                    event_loop.create_task(broadcast_state(dict(state))) # type: ignore
                    return
            except RuntimeError:
                pass
            
            # Not in the loop, use threadsafe
            asyncio.run_coroutine_threadsafe(broadcast_state(dict(state)), event_loop) # type: ignore
        except Exception:
            pass

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

@app.post("/state")
@app.post("/update_state")
async def update_state(data: Mapping[str, Any]):
    """Update the global state and broadcast it."""
    global current_state
    current_state = dict(data)
    await broadcast_state(current_state)
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

@app.post("/set_level")
async def set_level(data: Mapping[str, object]):
    """Set operational level requested from UI."""
    global pending_level
    try:
        level_val = int(data.get("level", 10))
        pending_level = level_val
        await manager.broadcast(json.dumps({"control": {"pending_level": level_val}}))
        return {"status": "success", "level": level_val}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/ingest_wisdom")
async def ingest_wisdom():
    """Stub endpoint for Mission Control (ingest)."""
    # Placeholder: real ingestion should be wired to orchestrator task queue.
    return {"status": "started"}

@app.post("/process_wisdom")
async def process_wisdom():
    """Stub endpoint for Mission Control (process)."""
    return {"status": "started"}

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
    
    dashboard_v2_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "v2", "dist")
    legacy_dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")
    
    if os.path.exists(dashboard_v2_path):
        print(f"🚀 Modern Dashboard (v2) served at http://localhost:{chosen_port}/dashboard")
        if os.path.exists(legacy_dashboard_path):
            print(f"📜 Legacy Dashboard (v1) available at http://localhost:{chosen_port}/v1")
    elif os.path.exists(legacy_dashboard_path):
        print(f"✅ Dashboard (v1) served at http://localhost:{chosen_port}/dashboard")
    try:
        ws_url = f"ws://127.0.0.1:{chosen_port}/ws"
        port_info = {"port": chosen_port, "ws_url": ws_url}
        print(f"📄 Persisting port info to data files... (port {chosen_port})")

        root_dir = Path(__file__).resolve().parent.parent
        print(f"📂 Project root identified as: {root_dir}")

        legacy_path = root_dir / "dashboard" / "data" / "api_port.json"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text(json.dumps(port_info, indent=2))
        print(f"✅ Wrote legacy path: {legacy_path}")

        v2_path = root_dir / "dashboard" / "v2" / "public" / "data" / "api_port.json"
        v2_path.parent.mkdir(parents=True, exist_ok=True)
        v2_path.write_text(json.dumps(port_info, indent=2))
        print(f"✅ Wrote v2 path: {v2_path}")
        
        # Also write to dist/data for the production build if it exists
        v2_dist_path = root_dir / "dashboard" / "v2" / "dist" / "data" / "api_port.json"
        if v2_dist_path.parent.parent.exists(): # If dist exists
            v2_dist_path.parent.mkdir(parents=True, exist_ok=True)
            v2_dist_path.write_text(json.dumps(port_info, indent=2))
            print(f"✅ Wrote v2 dist path: {v2_dist_path}")
    except Exception as port_err:
        print(f"⚠️ Could not persist port info: {port_err}")

    config = uvicorn.Config(app, host="0.0.0.0", port=chosen_port, loop="asyncio")
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    start_api_server(8000)
