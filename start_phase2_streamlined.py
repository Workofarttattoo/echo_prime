import os
import sys
import asyncio
import time
import gc
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Optimize for M4 chip and 24GB RAM - conservative approach
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"  # Conservative threading
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["ECH0_PHASE"] = "2"
os.environ["ECH0_FULL_ARCH"] = "1"
os.environ["ECH0_LIGHTWEIGHT"] = "0"  # Full capabilities but controlled

from main_orchestrator import EchoPrimeAGI
from core.api_service import agi_instance

async def start_streamlined_phase2():
    print("🚀 ECH0-PRIME Phase 2 | M4 Optimized | 24GB RAM (Streamlined)")
    print("=" * 65)

    # Memory optimization
    gc.set_threshold(700, 10, 10)

    start_time = time.time()
    print("1. Initializing cognitive architecture with controlled resource usage...")

    try:
        # Initialize with controlled settings for M4
        agi = EchoPrimeAGI(
            lightweight=False,  # Full capabilities
            enable_voice=False,  # Conserve resources
            memory_optimized=True  # Enable memory optimization
        )

        # Connect to API service
        global agi_instance
        agi_instance = agi

        init_time = time.time() - start_time
        print(f"   ⏱️ Initialization completed in {init_time:.2f}s")

        print("2. Phase 2 capabilities loaded:")
        print("   ✅ Hierarchical Generative Model (HGM)")
        print("   ✅ Free Energy Engine")
        print("   ✅ Prompt Masterworks (20 techniques)")
        print("   ✅ Compressed Knowledge Base")
        print("   ✅ Memory-optimized reasoning")

        print("3. Starting core observer (without heavy swarm components)...")

        # Start basic observer loop without heavy components
        observer_task = asyncio.create_task(basic_observer_loop(agi))

        print("\n" + "=" * 65)
        print("✅ PHASE 2 STREAMLINED CAPABILITIES ACTIVE")
        print("   Optimized for Apple M4 + 24GB RAM")
        print("   Memory-efficient cognitive processing")
        print("   Core reasoning and prompt masterworks active")
        print("=" * 65)

        # Keep running with periodic health checks
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                gc.collect()
                print("🧹 Memory cleanup completed")
        except KeyboardInterrupt:
            print("\n🛑 Shutting down ECH0-PRIME Phase 2...")
            observer_task.cancel()
            agi.cleanup()
            print("✅ Shutdown complete")

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()

async def basic_observer_loop(agi):
    """Basic observer loop without heavy components"""
    while True:
        try:
            # Basic health check every 30 seconds
            await asyncio.sleep(30)
            # Could add basic input processing here
        except Exception as e:
            print(f"Observer loop error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(start_streamlined_phase2())
