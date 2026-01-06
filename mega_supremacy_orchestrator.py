#!/usr/bin/env python3
"""
🚀 ECH0-PRIME MEGA SUPREMACY ORCHESTRATOR
========================================
Performs complete knowledge integration, fine-tuning, benchmarking, 
and system validation across all available local datasets and intelligence benchmarks.
"""

import os
import sys
import time
import json
import asyncio
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

async def run_command(command, description):
    print(f"\n⚡ [PHASE] {description}")
    print(f"Running: {command}")
    start_time = time.time()
    try:
        # Use subprocess to run the command and stream output
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        duration = time.time() - start_time
        if process.returncode == 0:
            print(f"✅ Success: {description} (Duration: {duration:.2f}s)")
            return True, stdout.decode()
        else:
            print(f"❌ Failed: {description} (Exit code: {process.returncode})")
            print(f"Error: {stderr.decode()}")
            return False, stderr.decode()
    except Exception as e:
        print(f"❌ Exception in {description}: {e}")
        return False, str(e)

async def main():
    print("="*60)
    print("🌟 ECH0-PRIME SUPREMACY: FULL SYSTEM EXECUTION")
    print("="*60)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # HARDCODED VERIFIED PYTHON PATH
    python_cmd = "/Users/noone/miniconda3/bin/python"
    print(f"Using Verified Python: {python_cmd}")
    print("="*60 + "\n")
    
    summary = {
        "start_time": time.time(),
        "phases": {}
    }

    # PHASE 1: Knowledge Ingestion & Wisdom Processing
    print("🟢 PHASE 1: KNOWLEDGE INTEGRATION")
    
    # 1.1 Process Wisdom Files (PDFs, JSONs in research_drop)
    res, out = await run_command(f"{python_cmd} wisdom_processor.py", "Processing Research & Wisdom")
    summary["phases"]["wisdom_processing"] = res

    # 1.2 Ingest Local Workspace
    # Focus on 'noone' home dir but skim as requested
    home_dir = "/Users/noone"
    res, out = await run_command(f"{python_cmd} training/ingest_local.py --dir {home_dir}", "Skimming /Users/noone for Pertinent Files")
    summary["phases"]["local_ingestion"] = res

    # PHASE 2: Comprehensive Benchmarking
    print("\n🟢 PHASE 2: COMPREHENSIVE BENCHMARKING")
    
    # 2.1 Full HLE (Humanity's Last Exam)
    res, out = await run_command(f"{python_cmd} tests/benchmark_hle.py", "Full HLE Benchmark (100 Samples)")
    summary["phases"]["hle_benchmark"] = res

    # 2.2 Comprehensive AI Suite (GSM8K, ARC, MMLU, etc.)
    res, out = await run_command(f"{python_cmd} ai_benchmark_suite.py --use-ech0 --full", "Comprehensive AI Suite")
    summary["phases"]["ai_suite"] = res

    # PHASE 3: System Validation & Tests
    print("\n🟢 PHASE 3: SYSTEM VALIDATION")
    
    # 3.1 Run core tests
    res, out = await run_command(f"{python_cmd} -m pytest tests/test_phase_1.py tests/test_phase_2.py", "Core System Validation")
    summary["phases"]["system_validation"] = res

    # PHASE 4: Fine-tuning & Optimization
    print("\n🟢 PHASE 4: WEIGHT OPTIMIZATION")
    res, out = await run_command(f"{python_cmd} force_consolidation.py", "Cognitive Weight Consolidation")
    summary["phases"]["fine_tuning"] = res

    # FINAL REPORT
    print("\n" + "="*60)
    print("📊 SUPREMACY RUN SUMMARY")
    print("="*60)
    for phase, status in summary["phases"].items():
        icon = "✅" if status else "❌"
        print(f"{phase:<25} : {icon}")
    
    total_duration = time.time() - summary["start_time"]
    print(f"\nTotal Execution Time: {total_duration/60:.2f} minutes")
    print("="*60)

    # Save summary
    with open("mega_supremacy_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Report saved to: mega_supremacy_report.json")

if __name__ == "__main__":
    asyncio.run(main())

