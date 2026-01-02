import sys
import os
import numpy as np

# Add the project root to sys.path
sys.path.append('/Users/noone/.gemini/antigravity/scratch/echo_prime')

from reasoning.orchestrator import ReasoningOrchestrator

def test_llm_reasoning():
    print("--- Verifying ECH0-PRIME Low-Cost LLM Reasoner ---")
    
    # Initialize with LLM enabled
    orchestrator = ReasoningOrchestrator(use_llm=True, model_name="llama3.2")
    
    print("\n[Querying Local LLM (Llama 3.2)]")
    
    source = {
        "status": "Sensory overload at Level 0",
        "priority": "High",
        "error_magnitude": 0.85
    }
    target = {
        "action": "Attenuate sensory gain",
        "expected_result": "Lower prediction error"
    }
    
    result = orchestrator.reason_about_scenario(source, target)
    
    print(f"\nAnalogical Similarity: {result['analogical_similarity']:.4f}")
    print(f"Causal Influence: {result['estimated_causal_influence']:.4f}")
    print("\n--- LLM INSIGHT (Prefrontal Cortex Perspective) ---")
    print(result['llm_insight'])
    print("\n--- End of Insight ---")
    
    if "BRIDGE ERROR" in result['llm_insight']:
        print("\nFAILURE: Could not connect to Ollama.")
    else:
        print("\nSUCCESS: Real LLM integrated into the reasoning loop.")

if __name__ == "__main__":
    test_llm_reasoning()
