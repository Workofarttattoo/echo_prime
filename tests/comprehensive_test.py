import os
import sys
import json
import time
import numpy as np
import requests

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from main_orchestrator import EchoPrimeAGI
from training.intelligent_grader import IntelligentGrader
from mcp_server.registry import ToolRegistry

def run_comprehensive_diagnostics():
    print("\n" + "="*60)
    print("🚀 ECH0-PRIME COMPREHENSIVE DIAGNOSTIC v2.0")
    print("="*60)

    # 1. Initialize System
    print("\n[1/6] INITIALIZING CORE ENGINE...")
    try:
        agi = EchoPrimeAGI()
        agi.voice_enabled = False # Mute for test
        print("✅ Core Engine and Bridges initialized.")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # 2. Verify Tool MCP Registry
    print("\n[2/6] VERIFYING MCP TOOL REGISTRY...")
    schemas = ToolRegistry.get_schemas()
    tool_names = [s['name'] for s in schemas]
    expected_tools = ['qulab_cmd', 'scan_arxiv', 'pinecone_store', 'pinecone_search', 'log_hallucination']
    
    for tool in expected_tools:
        if tool in tool_names:
            print(f"✅ Tool '{tool}' registered and discoverable.")
        else:
            print(f"⚠️ Tool '{tool}' MISSING from registry.")

    # 3. Test Reasoning & Tool Use (Closed Loop)
    print("\n[3/6] TESTING CLOSED-LOOP REASONING & RAG...")
    test_intent = "Research the latest papers on 'Topological Quantum Computing' using Arxiv, then store a summary in my deep memory."
    
    outcome = agi.cognitive_cycle(np.random.randn(1000000), test_intent)
    insight = outcome.get("llm_insight", "")
    
    if "arxiv" in insight.lower() or "ACTION" in insight:
        print("✅ ECH0 attempted tool usage or research.")
    else:
        print("⚠️ ECH0 did not trigger tool usage for the research goal.")
    
    # 4. Test Memory Bridges (Local & Pinecone)
    print("\n[4/6] TESTING MEMORY BRIDGES...")
    # Test Local Memory
    agi.gov_mem.store("Joshua's favorite color is neon-violet.")
    recall = agi.gov_mem.search("What is Joshua's favorite color?")
    if "neon-violet" in recall:
        print("✅ Local Semantic Memory working (RAG verified).")
    else:
        print("❌ Local Semantic Memory failed recall.")

    # 5. Test Intelligent Grader
    print("\n[5/6] TESTING INTELLIGENT GRADER...")
    grader = IntelligentGrader()
    q = "Prove that there are infinitely many primes."
    e = "Euclid's proof or similar."
    r = "There are infinitely many primes because if you multiply all known primes and add one, the new number is not divisible by any of them."
    score, just = grader.grade(q, e, r)
    print(f"✅ Grader Response: Score={score}, Justification='{just[:50]}...'")

    # 6. Test Dashboard API
    print("\n[6/6] TESTING DASHBOARD API CONNECTIVITY...")
    try:
        # Start API in a thread for testing if not running
        # But we assume the dev server is likely running if dashboard is active
        res = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if res.status_code == 200:
            print("✅ Dashboard API Service (FastAPI) is ONLINE.")
    except:
        print("⚠️ Dashboard API Service is OFFLINE or unreachable.")

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE.")
    print("="*60)

if __name__ == "__main__":
    run_comprehensive_diagnostics()
