import asyncio
import os
import sys
import json
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from main_orchestrator import EchoPrimeAGI

async def build_project_lattice():
    print("🚀 ECH0-PRIME: Building Semantic Lattice for Project Architecture")
    print("=" * 60)
    
    # Initialize AGI in lightweight mode for reasoning
    agi = EchoPrimeAGI(lightweight=True, enable_voice=False)
    
    # Define the domain and key concepts
    domain = "ECH0-PRIME AGI Architecture"
    concepts = [
        "Hierarchical Generative Model (HGM)",
        "Free Energy Engine (Variational Inference)",
        "Quantum Attention (QuLab)",
        "Global Workspace Theory (GWT)",
        "Compressed Knowledge Base (10^15 scale)",
        "Prompt Masterworks (20 levels)",
        "Swarm & Hive Mind Intelligence",
        "Predictive Coding Hierarchy (5 levels)",
        "Recursive Mirror Metacognition"
    ]
    
    print(f"1. Invoking Masterwork 8: Semantic Lattice for {domain}...")
    lattice_prompt = agi.semantic_lattice(domain, concepts)
    
    # Run a reasoning cycle to generate the lattice
    # We use Recursive Mirror to observe the process
    print("2. Observing reasoning process via Masterwork 7: Recursive Mirror...")
    
    # Use the reasoner directly to get the output
    context = {
        "sensory_input": "Internal Architecture Metadata",
        "current_state": "Self-Analysis Mode"
    }
    
    # Create the combined prompt for the LLM
    combined_prompt = (
        f"{lattice_prompt}\n\n"
        f"PROTOCOL: While building this lattice, apply RECURSIVE MIRROR to observe your own architectural assumptions."
    )
    
    result = agi.reasoner.reason_about_scenario(
        context, 
        {"goal": combined_prompt}
    )
    
    lattice_output = result.get("llm_insight", "Lattice generation failed.")
    
    print("\n--- SEMANTIC LATTICE OUTPUT ---\n")
    print(lattice_output)
    
    # Save the lattice to the knowledge base
    # We'll save it as a structured document
    output_path = "docs/SEMANTIC_LATTICE_AGI.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# ECH0-PRIME Semantic Lattice: AGI Architecture\n\n")
        f.write(lattice_output)
    
    print(f"\n✅ Semantic Lattice saved to {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(build_project_lattice())

