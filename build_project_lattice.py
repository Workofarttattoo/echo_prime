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
    print("🚀 ECH0-PRIME: Building Semantic Lattice for Project Knowledge...")
    print("=" * 60)

    # Initialize AGI in lightweight mode for reasoning
    agi = EchoPrimeAGI(lightweight=True)
    
    # Define domain and concepts for the lattice
    domain = "ECH0-PRIME AGI Architecture"
    concepts = [
        "Hierarchical Generative Model",
        "Free Energy Minimization",
        "Quantum Attention Coherence",
        "Prompt Masterworks",
        "Compressed Knowledge Base",
        "Swarm Intelligence",
        "Metacognitive Reflection",
        "Human-in-the-Loop Governance"
    ]
    
    # 1. Build the Lattice
    print(f"Constructing lattice for domain: {domain}...")
    lattice_output = agi.semantic_lattice(domain, concepts)
    
    # 2. Observe with Recursive Mirror (Metacognition)
    print("Mirroring the reasoning process...")
    mirror_analysis = agi.recursive_mirror(f"Construction of {domain} lattice")
    
    # 3. Save the results
    lattice_data = {
        "domain": domain,
        "concepts": concepts,
        "lattice": lattice_output,
        "mirror_analysis": mirror_analysis,
        "timestamp": str(asyncio.get_event_loop().time())
    }
    
    os.makedirs("knowledge_artifacts", exist_ok=True)
    file_path = "knowledge_artifacts/project_semantic_lattice.json"
    
    with open(file_path, "w") as f:
        json.dump(lattice_data, f, indent=2)
        
    print("\n" + "=" * 60)
    print(f"✅ Semantic Lattice and Mirror Analysis saved to {file_path}")
    print("=" * 60)
    
    # Output a snippet
    print("\n--- LATTICE PREVIEW ---")
    print(lattice_output[:500] + "...")

if __name__ == "__main__":
    asyncio.run(build_project_lattice())

