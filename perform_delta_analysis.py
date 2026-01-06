import asyncio
import os
import sys
import json
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from main_orchestrator import EchoPrimeAGI

async def perform_delta_analysis():
    print("🚀 ECH0-PRIME: Performing Delta Encoding Analysis (Phase 1 -> Phase 2)...")
    print("=" * 60)

    agi = EchoPrimeAGI(lightweight=True)
    
    # Define reference state (Phase 1)
    reference_state = {
        "phase": 1,
        "status": "Architecture Validated",
        "capabilities": [
            "Hierarchical Generative Model (Basic)",
            "Free Energy Engine (Prototype)",
            "Ollama Bridge (Standard)",
            "Memory (Local JSON)"
        ],
        "benchmarks": {
            "turing": "9%",
            "gsm8k": "Baseline"
        }
    }
    
    # Define current state (Phase 2)
    current_state = {
        "phase": 2,
        "status": "Capability Development Active",
        "capabilities": [
            "Hierarchical Generative Model (Neural Accelerated)",
            "Free Energy Engine (Optimized)",
            "Prompt Masterworks (20 Advanced Protocols)",
            "Deep Memory (Pinecone + Compressed KB)",
            "Swarm/Hive Mind (Initializing)",
            "Modern AGI Cockpit (v2)"
        ],
        "benchmarks": {
            "turing": "64%",
            "gsm8k": "Verified",
            "knowledge": "Massive Ingestion (GooAQ, Field Manual)"
        }
    }
    
    # Perform Delta Encoding
    print("Generating delta encoding...")
    delta_report = agi.delta_encoding(reference_state, current_state)
    
    # Save the report
    os.makedirs("knowledge_artifacts", exist_ok=True)
    file_path = "knowledge_artifacts/phase2_delta_report.json"
    
    report_data = {
        "reference": reference_state,
        "current": current_state,
        "delta": delta_report,
        "timestamp": str(asyncio.get_event_loop().time())
    }
    
    with open(file_path, "w") as f:
        json.dump(report_data, f, indent=2)
        
    print("\n" + "=" * 60)
    print(f"✅ Delta Encoding Analysis saved to {file_path}")
    print("=" * 60)
    
    # Output a snippet
    print("\n--- DELTA REPORT PREVIEW ---")
    print(delta_report[:500] + "...")

if __name__ == "__main__":
    asyncio.run(perform_delta_analysis())
