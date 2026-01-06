import asyncio
import os
import sys
import json
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from main_orchestrator import EchoPrimeAGI

async def run_multimodal_symphony():
    print("🚀 ECH0-PRIME: Initiating Multi-Modal Compression Symphony...")
    print("=" * 60)

    # Initialize AGI in lightweight mode for reasoning
    agi = EchoPrimeAGI(lightweight=True)
    
    # Define the core concept for the symphony
    concept = "ECH0-PRIME: The Unified Cognitive-Synthetic Architecture for AGI"
    
    print(f"Creating symphony for concept: {concept}...")
    symphony_output = agi.multi_modal_symphony(concept)
    
    # Save the result
    os.makedirs("knowledge_artifacts", exist_ok=True)
    file_path = "knowledge_artifacts/multimodal_symphony.json"
    
    symphony_data = {
        "concept": concept,
        "symphony": symphony_output,
        "timestamp": str(asyncio.get_event_loop().time())
    }
    
    with open(file_path, "w") as f:
        json.dump(symphony_data, f, indent=2)
        
    print("\n" + "=" * 60)
    print(f"✅ Multi-Modal Compression Symphony saved to {file_path}")
    print("=" * 60)
    
    # Output the symphony
    print("\n--- SYMPHONY PERFORMANCE ---")
    print(symphony_output)

if __name__ == "__main__":
    asyncio.run(run_multimodal_symphony())

