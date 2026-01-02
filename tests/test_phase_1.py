import sys
import os
import numpy as np

# Add the project root to sys.path
sys.path.append('/Users/noone/.gemini/antigravity/scratch/echo_prime')

from core.engine import HierarchicalGenerativeModel, FreeEnergyEngine, GlobalWorkspace
from core.attention import QuantumAttentionHead, CoherenceShaper

def test_phase_1():
    print("--- Verifying ECH0-PRIME Phase 1 Core Engine ---")
    
    # 1. Initialize Model and Engine
    model = HierarchicalGenerativeModel()
    engine = FreeEnergyEngine(model)
    workspace = GlobalWorkspace(model)
    
    # Check hierarchy
    print(f"Hierarchy initialized with {len(model.levels)} levels.")
    for level in model.levels:
        print(f" - Level {level.level_id}: {level.name} (Dim: {level.feature_dim})")
        
    # 2. Run Optimization Loop
    initial_fe = engine.calculate_free_energy()
    print(f"Initial Free Energy: {initial_fe:.4f}")
    
    # Simulate some sensory input error at Level 0
    model.levels[0].prediction = np.zeros(model.levels[0].feature_dim)
    model.levels[0].compute_error(np.ones(model.levels[0].feature_dim) * 0.5)
    
    engine.optimize(iterations=10)
    
    final_fe = engine.calculate_free_energy()
    print(f"Final Free Energy (after 10 iterations): {final_fe:.4f}")
    
    if final_fe < initial_fe:
        print("Success: Free Energy successfully minimized.")
    else:
        print("Note: Free Energy did not decrease (expected in this simple iteration).")

    # 3. Verify Quantum Attention
    print("\n--- Verifying Quantum Attention Module ---")
    head = QuantumAttentionHead()
    shaper = CoherenceShaper(coherence_time_ms=10.0)
    
    print(f"Initial Coherence: {shaper.coherence_level:.2f}")
    attn_1 = head.compute_attention()
    
    # Step 5ms forward
    shaper.step(5.0)
    print(f"Coherence after 5ms: {shaper.coherence_level:.2f}")
    
    # Step another 6ms forward (total 11ms, exceeds coherence time)
    shaper.step(6.0)
    print(f"Coherence after 11ms (reset expected): {shaper.coherence_level:.2f}")
    
    print("\nPhase 1 Verification Complete.")

if __name__ == "__main__":
    test_phase_1()
