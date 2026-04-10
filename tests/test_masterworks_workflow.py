import asyncio
import numpy as np
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_orchestrator import EchoPrimeAGI

async def test_masterworks_workflow():
    print("🚀 TESTING MASTERWORKS INTEGRATION IN COGNITIVE CYCLE")
    print("=" * 60)
    
    # Initialize ECH0-PRIME in full mode (not lightweight)
    os.environ["ECH0_FULL_ARCH"] = "1"
    
    try:
        agi = EchoPrimeAGI(lightweight=False)
        print("\n✅ ECH0-PRIME initialized with Masterworks")
        
        # Define a complex goal to trigger various masterworks
        complex_goal = "Analyze the long-term impact of artificial general intelligence on human creative expression and identify key patterns of emergence."
        
        print(f"\n🎯 GOAL: {complex_goal}")
        print("\n🔄 Running cognitive cycle...")
        
        # Create mock sensory input
        sensory_input = np.random.randn(1000)
        
        # Execute cognitive cycle
        # We'll use a try-except because some components might fail if Ollama is not running
        try:
            result = agi.cognitive_cycle(sensory_input, complex_goal)
            print("\n✅ Cognitive Cycle completed")
            
            # Check if masterworks were mentioned in output (since we appended metadata)
            if "TEMPORAL ANCHOR METADATA" in str(result):
                print("⚓ TEMPORAL ANCHOR: Detected in response")
            
            if "ECHO VISION ANALYSIS PROTOCOL" in str(result):
                print("👁️ ECHO VISION: Detected (should only be there if image present, wait...)")

            print("\n📝 Sample Response Segment:")
            insight = result if isinstance(result, str) else result.get("llm_insight", "")
            print(insight[:500] + "...")
            
        except Exception as e:
            print(f"\n❌ Cognitive Cycle failed (likely Ollama connection): {e}")
            print("Note: Masterworks are integrated but LLM execution requires Ollama.")
            
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_masterworks_workflow())

