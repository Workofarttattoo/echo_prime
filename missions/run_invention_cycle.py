import sys
import os
import json
import time

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning.llm_bridge import OllamaBridge
from reasoning.tools.arxiv_scanner import ArxivScanner
from ech0_governance.evaluators import Parliament

def run_invention_cycle():
    print("\n--- 🚀 STARTING ECH0 INVENTION RUN ---")
    
    # 1. Initialize Tools
    llm = OllamaBridge(model="ech0-unified-14b-enhanced")
    scanner = ArxivScanner()
    parliament = Parliament(llm)
    
    # 2. Arxiv Ingestion
    print("\n[PHASE 1] Scanning Arxiv for 'Wisdom'...")
    queries = ["quantum computing hardware", "cancer metabolic treatment", "neuromorphic metamaterials"]
    papers = []
    for q in queries:
        papers.extend(scanner.scan(q, max_results=3)) # Keep small for speed in demo
    
    context = "\n".join([f"- {p['title']}: {p['summary'][:200]}..." for p in papers])
    print(f"Ingested {len(papers)} papers.")

    # 3. Ideation (Recursive Reasoning)
    print("\n[PHASE 2] Generating Inventions...")
    prompt = (
        f"Synthesize the following scientific papers into 5 novel, ground-breaking invention concepts.\n"
        f"PAPERS:\n{context}\n\n"
        "For each invention, provide:\n"
        "1. Title\n2. Scientific Principle (Intersection of fields)\n3. Proof of Concept (How to build it)\n4. Expected Impact\n"
        "Output as valid JSON list."
    )
    
    raw_ideas = llm.query(prompt)
    # Mocking parsing for robustness if LLM output isn't perfect JSON
    # In a real run, we'd use a parser. For now, assuming the model complies or we extract.
    
    # 4. Parliament Filtering (Top 15%)
    print("\n[PHASE 3] Parliament Governance & Filtering...")
    
    refined_inventions = []
    
    # Simulating the list for the sake of the script structure if LLM fails to return list
    # ideally we parse 'raw_ideas'
    
    print(f"Raw Output: {raw_ideas[:100]}...")
    
    # Ask LLM to format/filter strictly now
    filter_prompt = (
        f"Review these ideas and select the top 15% (best 1) based on feasibility and impact.\n"
        f"Ideas: {raw_ideas}\n"
        "Output the selected idea(s) in this EXACT JSON format:\n"
        "[\n"
        "  {\n"
        "    'title': '...',\n"
        "    'scientific_principle': '...',\n"
        "    'proof_of_concept': '...',\n"
        "    'impact': '...'\n"
        "  }\n"
        "]"
    )
    final_json_str = llm.query(filter_prompt)
    
    # 5. Output
    output_path = "ech0_invention_run.json"
    with open(output_path, "w") as f:
        f.write(final_json_str)
        
    print(f"\n✅ Invention Run Complete. Saved to {output_path}")
    print(final_json_str)

if __name__ == "__main__":
    run_invention_cycle()
