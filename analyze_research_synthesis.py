import asyncio
import os
import sys
import json
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from main_orchestrator import EchoPrimeAGI

async def analyze_research_synthesis():
    print("🚀 ECH0-PRIME: Performing Recursive Mirror Analysis on Research Synthesis...")
    print("=" * 60)

    # Initialize AGI
    agi = EchoPrimeAGI(lightweight=True)
    
    # 1. Sample Research Content (Filenames)
    pdf_samples = [
        "2505_09774v2.pdf",
        "2505_10544v1.pdf",
        "2505_11043v1.pdf"
    ]
    
    # 2. Query for Synthesis
    print(f"Testing synthesis of core concepts from {len(pdf_samples)} research samples...")
    
    # We'll ask the AGI to synthesize the "Deep Theoretical Contributions" from these papers
    query = f"Analyze the synthesized knowledge and theoretical contributions derived from the research papers: {', '.join(pdf_samples)}. How do these integrate with the Hierarchical Generative Model?"
    
    print("\n[PHASE 1: RECALL & SYNTHESIS]")
    # We use the full cognitive cycle to trigger RAG and LLM reasoning
    import numpy as np
    zero_sensory = np.zeros(1000)
    outcome = agi.cognitive_cycle(zero_sensory, query)
    synthesis_result = outcome.get("llm_insight", "No response generated.")
    
    print(f"\n--- SYNTHESIS OUTPUT ---\n{synthesis_result[:500]}...")

    # 3. Apply RECURSIVE MIRROR (The Reflection)
    print("\n[PHASE 2: RECURSIVE MIRROR REFLECTION]")
    mirror_task = f"Evaluate the depth and accuracy of the following synthesis of research data: {synthesis_result[:1000]}"
    mirror_analysis = agi.recursive_mirror(mirror_task)
    
    # 4. Save Artifact
    artifact = {
        "samples": pdf_samples,
        "synthesis": synthesis_result,
        "mirror_analysis": mirror_analysis,
        "timestamp": str(asyncio.get_event_loop().time())
    }
    
    os.makedirs("knowledge_artifacts", exist_ok=True)
    file_path = "knowledge_artifacts/research_synthesis_mirror.json"
    with open(file_path, "w") as f:
        json.dump(artifact, f, indent=2)
        
    print("\n" + "=" * 60)
    print(f"✅ Research Synthesis Analysis and Mirror saved to {file_path}")
    print("=" * 60)
    
    print("\n--- MIRROR ANALYSIS SNIPPET ---")
    print(mirror_analysis[:1000] + "...")

if __name__ == "__main__":
    asyncio.run(analyze_research_synthesis())

