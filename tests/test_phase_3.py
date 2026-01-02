import sys
import os
import numpy as np

# Add the project root to sys.path
sys.path.append('/Users/noone/.gemini/antigravity/scratch/echo_prime')

from reasoning.orchestrator import ReasoningOrchestrator, ProbabilisticReasoning

def test_phase_3():
    print("--- Verifying ECH0-PRIME Phase 3 Reasoning & Causal Discovery ---")
    
    orchestrator = ReasoningOrchestrator()

    # 1. Test Probabilistic Inference (ELBO)
    print("\n[Testing Probabilistic Reasoning]")
    prob = ProbabilisticReasoning(latent_dim=10)
    
    def dummy_log_likelihood(x, z):
        # High likelihood if z is close to x
        return -np.sum((x - z)**2)

    x_input = np.ones(10)
    elbo = prob.calculate_elbo(x_input, dummy_log_likelihood)
    print(f"Calculated ELBO for sample input: {elbo:.4f}")
    assert isinstance(elbo, float)

    # 2. Test Analogical Reasoning
    print("\n[Testing Analogical Structure Mapping]")
    vec_a = np.array([1, 0, 0, 1])
    vec_b = np.array([1, 0, 0.1, 0.9])
    
    scenario_1 = {"feature_alpha": vec_a, "type": "A"}
    scenario_2 = {"feature_alpha": vec_b, "type": "A"}
    
    sim_score = orchestrator.analogy.structure_mapping(scenario_1, scenario_2)
    print(f"Analogical Similarity Score: {sim_score:.4f}")
    assert sim_score > 0.8

    # 3. Test Causal Discovery
    print("\n[Testing Causal Effect Estimation]")
    x_data = np.linspace(0, 10, 100)
    y_data = 2 * x_data + np.random.normal(0, 1, 100) # Strong causal link
    
    effect = orchestrator.causal.estimate_causal_effect(x_data, y_data)
    print(f"Estimated Causal Correlation: {effect:.4f}")
    assert effect > 0.9

    # Test Intervention string output
    intervention = orchestrator.causal.perform_intervention("p_sensor", 0.5)
    print(f"Intervention Triggered: {intervention}")

    print("\nPhase 3 Verification Complete.")

if __name__ == "__main__":
    test_phase_3()
