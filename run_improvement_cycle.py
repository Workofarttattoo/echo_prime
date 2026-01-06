import asyncio
import os
import sys
import json
import time
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from main_orchestrator import EchoPrimeAGI
from missions.self_modification import CodeAnalyzer

async def run_comprehensive_improvement():
    print("🚀 ECH0-PRIME: Initiating Autonomous Self-Improvement Cycle...")
    print("=" * 60)

    # Initialize AGI in lightweight mode
    agi = EchoPrimeAGI(lightweight=True)
    
    # 1. Internal Orchestrator Improvement
    print("\n[STEP 1] Running Orchestrator Internal Improvement Cycle...")
    # Manually trigger the cycle
    await agi._run_self_improvement_cycle()
    
    # 2. Code Analysis of Core Engine
    print("\n[STEP 2] Analyzing Core Engine for Optimization...")
    analyzer = CodeAnalyzer()
    core_engine_path = os.path.join(project_root, "core", "engine.py")
    
    if os.path.exists(core_engine_path):
        with open(core_engine_path, "r") as f:
            code = f.read()
        
        print(f"Analyzing {core_engine_path}...")
        analysis = analyzer.analyze_code(code, filename="core/engine.py")
        
        print(f"   Complexity: {analysis.get('complexity', {}).get('cyclomatic_complexity', 'N/A')}")
        print(f"   Suggestions Found: {len(analysis.get('suggestions', []))}")
        
        # Log top 3 suggestions
        for i, suggestion in enumerate(analysis.get("suggestions", [])[:3]):
            print(f"   Suggestion {i+1}: {suggestion.get('type')} in {suggestion.get('name')}")
    
    # 3. Knowledge Base Optimization
    print("\n[STEP 3] Optimizing Massive Knowledge Base...")
    from learning.compressed_knowledge_base import CompressedKnowledgeBase
    kb = CompressedKnowledgeBase("./massive_kb")
    await kb.load_async()
    
    before_stats = kb.get_statistics()
    print(f"   Initial Nodes: {before_stats['total_nodes']}")
    
    await kb.optimize_storage()
    
    after_stats = kb.get_statistics()
    print(f"   Optimized Nodes: {after_stats['total_nodes']}")
    print(f"   Efficiency Improvement: {after_stats['storage_efficiency'] - before_stats['storage_efficiency']:.2f}x")

    # 4. Generate Final Improvement Report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "orchestrator_status": agi.get_self_improvement_status(),
        "code_analysis": {
            "file": "core/engine.py",
            "complexity": analysis.get('complexity', {}).get('cyclomatic_complexity', 'N/A') if 'analysis' in locals() else 'N/A'
        },
        "kb_optimization": {
            "nodes_before": before_stats['total_nodes'],
            "nodes_after": after_stats['total_nodes']
        }
    }
    
    os.makedirs("improvement_logs", exist_ok=True)
    report_path = f"improvement_logs/improvement_report_{int(time.time())}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "=" * 60)
    print(f"✅ Self-Improvement Cycle Complete. Report saved to {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_comprehensive_improvement())

