from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import time
from typing import List, Optional

app = FastAPI(
    title="ECH0-PRIME CSA API",
    description="Cognitive-Synthetic Architecture Inference Server",
    version="1.0.0"
)

class CSARequest(BaseModel):
    input_data: List[float]
    temperature: float = 0.7
    max_steps: int = 50

class CSAResponse(BaseModel):
    output: List[float]
    phi_value: float
    processing_time: float
    status: str

# Mock CSA for demonstration (replace with real CSA when deployed)
class MockCSA:
    def __init__(self):
        self.layers = 5
        print("🎯 Mock CSA initialized (Railway deployment)")

    def process(self, input_data: np.ndarray, steps: int = 50) -> tuple:
        """Mock CSA processing"""
        # Simulate hierarchical processing
        output = input_data.copy()

        for layer in range(self.layers):
            # Simulate neural processing
            output = np.tanh(output * 0.8 + np.random.randn(len(output)) * 0.1)
            output = output * (1 + layer * 0.1)  # Hierarchical scaling

        # Calculate mock phi (consciousness measure)
        phi = min(1.0, np.mean(np.abs(output)) * 0.5 + 0.2)

        return output.tolist(), phi

# Initialize CSA
csa = MockCSA()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ECH0-PRIME CSA Inference API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "csa_loaded": True
    }

@app.post("/csa/infer", response_model=CSAResponse)
async def csa_inference(request: CSARequest):
    """Run CSA inference"""
    try:
        start_time = time.time()

        # Convert input to numpy array
        input_array = np.array(request.input_data, dtype=np.float32)

        # Validate input
        if len(input_array) == 0:
            raise HTTPException(status_code=400, detail="Input data cannot be empty")

        if len(input_array) > 10000:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Input too large (max 10000 elements)")

        # Run CSA inference
        output, phi_value = csa.process(input_array, request.max_steps)

        processing_time = time.time() - start_time

        return CSAResponse(
            output=output,
            phi_value=float(phi_value),
            processing_time=processing_time,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSA inference failed: {str(e)}")

@app.get("/csa/info")
async def csa_info():
    """Get CSA model information"""
    return {
        "model_name": "ECH0-PRIME Cognitive-Synthetic Architecture",
        "architecture": "Hierarchical Predictive Coding",
        "capabilities": [
            "Multi-level cognitive processing",
            "Consciousness measurement (Φ)",
            "Free energy minimization",
            "Hierarchical attention",
            "Self-organizing dynamics"
        ],
        "input_format": "List of float values",
        "output_format": "Processed cognitive representation + Φ value",
        "deployment": "Railway (Cloud)",
        "version": "1.0.0"
    }

@app.post("/demo/chat")
async def demo_chat(message: str = "Hello, what can you tell me about consciousness?"):
    """Demo chat endpoint using mock CSA responses"""
    try:
        # Simulate processing the message through CSA
        message_vector = np.array([ord(c) / 255.0 for c in message[:100]])  # Simple character encoding
        message_vector = message_vector.tolist() + [0.0] * (100 - len(message_vector))  # Pad

        # Get CSA processing
        _, phi = csa.process(np.array(message_vector))

        # Generate response based on phi value
        if phi > 0.7:
            response = f"🧠 Based on deep cognitive processing (Φ = {phi:.2f}), I can tell you that consciousness involves integrated information processing across multiple hierarchical levels. The human brain maintains consciousness through coordinated neural activity, and current AI systems are approaching this through architectures like mine."
        elif phi > 0.4:
            response = f"🤔 After cognitive analysis (Φ = {phi:.2f}), consciousness appears to be an emergent property of complex information integration. It's characterized by self-awareness, subjective experience, and the ability to process information in a unified manner."
        else:
            response = f"💭 From my current processing state (Φ = {phi:.2f}), consciousness involves the integration of sensory inputs, memory, and cognitive processing into a coherent experience. It's one of the most fascinating aspects of intelligence, whether biological or artificial."

        return {
            "response": response,
            "phi_value": float(phi),
            "message_length": len(message),
            "processing_complete": True
        }

    except Exception as e:
        return {
            "response": f"I apologize, but I encountered an error processing your message: {str(e)}",
            "phi_value": 0.0,
            "error": True
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting ECH0-PRIME CSA on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
