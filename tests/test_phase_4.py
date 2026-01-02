import sys
import os
import numpy as np

# Add the project root to sys.path
sys.path.append('/Users/noone/.gemini/antigravity/scratch/echo_prime')

from training.pipeline import TrainingPipeline, IntrinsicCuriosityModule, SelfImprovementLoop

def test_phase_4():
    print("--- Verifying ECH0-PRIME Phase 4 Training & Infrastructure ---")
    
    pipeline = TrainingPipeline(model_params=500_000_000_000)

    # 1. Test Pre-training
    print(f"\n[Testing Pre-training Phase]")
    pipeline.run_pretraining(tokens_count=10**12)
    assert pipeline.state == "PRETRAINED"

    # 2. Test Curiosity Module
    print(f"\n[Testing Intrinsic Curiosity]")
    icm = IntrinsicCuriosityModule(state_dim=5)
    s = np.array([1, 0, 0, 0, 0])
    s_next = np.array([0, 1, 0, 0, 0])
    bonus = icm.compute_bonus(s, s_next)
    print(f"Curiosity Bonus (Novel state transition): {bonus:.4f}")
    assert bonus > 0

    # 3. Test RL Pipeline
    print(f"\n[Testing RL Pipeline]")
    pipeline.run_reinforcement_learning(tasks=["Navigation", "Tool Use"])
    assert pipeline.state == "RL_TUNED"

    # 4. Test Self-Improvement Scaffold
    print(f"\n[Testing Recursive Self-Improvement]")
    loop = SelfImprovementLoop()
    code = "def calculate_pi(): return 3.14"
    new_code = loop.propose_modification(code)
    print(f"Proposed Code Modification: {new_code}")
    is_safe = loop.formal_verification(new_code)
    print(f"Formal Verification Passed: {is_safe}")
    assert is_safe

    print("\nPhase 4 Verification Complete.")

if __name__ == "__main__":
    test_phase_4()
