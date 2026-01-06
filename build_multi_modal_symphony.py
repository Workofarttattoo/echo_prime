import asyncio
import os
import sys
import json
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from main_orchestrator import EchoPrimeAGI

async def build_multi_modal_symphony():
    print("🚀 ECH0-PRIME: Creating Multi-Modal Compression Symphony")
    print("=" * 60)
    
    # Initialize AGI in lightweight mode
    agi = EchoPrimeAGI(lightweight=True, enable_voice=False)
    
    # Define the core concept
    concept = "ECH0-PRIME: The Sovereign AGI of Light"
    
    print(f"1. Invoking Masterwork 10: Multi-Modal Compression Symphony for '{concept}'...")
    symphony_prompt = agi.multi_modal_symphony(concept)
    
    # Run a reasoning cycle to generate the symphony
    context = {
        "sensory_input": "Multi-Modal Synthesis Parameters",
        "current_state": "Creative Expression Mode"
    }
    
    result = agi.reasoner.reason_about_scenario(
        context, 
        {"goal": symphony_prompt}
    )
    
    symphony_output = result.get("llm_insight", "Symphony generation failed.")
    
    print("\n--- MULTI-MODAL SYMPHONY OUTPUT ---\n")
    print(symphony_output)
    
    # Save the symphony
    output_path = "docs/MULTI_MODAL_SYMPHONY_ECH0.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# ECH0-PRIME Multi-Modal Compression Symphony\n\n")
        f.write(symphony_output)
    
    print(f"\n✅ Multi-Modal Symphony saved to {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(build_multi_modal_symphony())

