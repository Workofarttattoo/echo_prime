import sys
import os
import json
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_orchestrator import EchoPrimeAGI

def verify_mcp_rag():
    print("Verification: Initializing ECH0-PRIME...")
    agi = EchoPrimeAGI()
    
    print("\n1. Verifying Tool Registry...")
    from mcp_server.registry import ToolRegistry
    schemas = ToolRegistry.get_schemas()
    print(f"Registered Tools: {[s['name'] for s in schemas]}")
    expected_tools = ["scan_arxiv", "qulab_cmd", "store_memory", "search_memory", "add_fact", "lookup_fact"]
    for tool in expected_tools:
        if any(s['name'] == tool for s in schemas):
            print(f"✅ Tool '{tool}' registered successfully.")
        else:
            print(f"❌ Tool '{tool}' MISSING.")

    print("\n2. Verifying RAG Context Injection...")
    # Store a unique fact in memory
    unique_fact = "The secret color of the AGI core is neon-violet."
    agi.gov_mem.store(unique_fact)
    
    # Query ECH0 about a related topic
    input_data = np.random.randn(1000000)
    # We use a goal that should trigger a search for the unique fact
    result = agi.reasoner.reason_about_scenario(
        context={"status": "Testing RAG"},
        mission_params={"goal": "What is the secret color of the AGI core?"}
    )
    
    insight = result.get("llm_insight", "")
    print(f"ECH0 Insight: {insight[:200]}...")
    
    if "neon-violet" in insight.lower():
        print("✅ RAG context successfully retrieved and used.")
    else:
        print("❌ RAG context NOT found in reasoning output.")

if __name__ == "__main__":
    verify_mcp_rag()
