import sys
import os
import numpy as np

# Add the project root to sys.path
sys.path.append('/Users/noone/.gemini/antigravity/scratch/echo_prime')

from memory.manager import MemoryManager, WorkingMemory, EpisodicMemory, SemanticMemory
from learning.meta import CSALearningSystem, MetaLearningController

def test_phase_2():
    print("--- Verifying ECH0-PRIME Phase 2 Memory & Learning ---")
    
    # 1. Test Working Memory Capacity
    print("\n[Testing Working Memory]")
    wm = WorkingMemory(capacity=5)
    for i in range(10):
        wm.store(np.random.randn(100))
    print(f"Stored 10 items in capacity-5 WM. Final count: {len(wm.retrieve_all())}")
    assert len(wm.retrieve_all()) == 5

    # 2. Test Semantic Binding (Circular Convolution)
    print("\n[Testing Semantic Binding]")
    sm = SemanticMemory(dimension=1000)
    vec_a = np.random.randn(1000)
    vec_b = np.random.randn(1000)
    
    bound = sm.bind(vec_a, vec_b)
    print(f"Bound vector magnitude: {np.linalg.norm(bound):.2f}")
    
    # Approx unbinding
    unbound_b = sm.unbind(bound, vec_a)
    similarity = np.dot(vec_b, unbound_b) / (np.linalg.norm(vec_b) * np.linalg.norm(unbound_b))
    print(f"Unbinding recovery similarity (cosine): {similarity:.4f}")
    # Cosine similarity for unbinding in HD computing is usually positive but not 1.0 without high dims
    assert similarity > 0.1 

    # 3. Test Meta-Learning Consolidation
    print("\n[Testing Meta-Learning]")
    controller = MetaLearningController(param_dim=100, alpha=0.1, beta=0.5)
    initial_slow = np.copy(controller.theta_slow)
    
    # Adapt fast weights
    grad = np.ones(100) * 0.5
    controller.adapt(grad)
    
    # Consolidate
    controller.consolidate()
    diff = np.linalg.norm(controller.theta_slow - initial_slow)
    print(f"Slow weights shift after one consolidation (beta=0.5): {diff:.4f}")
    assert diff > 0

    print("\nPhase 2 Verification Complete.")

if __name__ == "__main__":
    test_phase_2()
