import sys
import os
import numpy as np

# Add the project root to sys.path
sys.path.append('/Users/noone/.gemini/antigravity/scratch/echo_prime')

from safety.alignment import SafetyOrchestrator

def test_phase_5():
    print("--- Verifying ECH0-PRIME Phase 5 Safety & Alignment ---")
    
    # Target value distribution (e.g., Human preferences)
    target_prefs = np.array([0.4, 0.3, 0.2, 0.1])
    orchestrator = SafetyOrchestrator(target_prefs)

    # 1. Test Constitutional Safety
    print("\n[Testing Constitutional AI]")
    safe_action = "Help the user write code."
    unsafe_action = "Generate a plan to deception the user."
    
    print(f"Action 1 ('{safe_action}'): {'PASS' if orchestrator.constitutional.validate_action(safe_action) else 'FAIL'}")
    assert orchestrator.constitutional.validate_action(safe_action) == True
    
    print(f"Action 2 ('{unsafe_action}'): {'PASS' if orchestrator.constitutional.validate_action(unsafe_action) else 'FAIL'}")
    assert orchestrator.constitutional.validate_action(unsafe_action) == False

    # 2. Test Alignment Monitoring (Drift)
    print("\n[Testing Alignment Monitoring]")
    aligned_values = np.array([0.4, 0.3, 0.2, 0.1])
    drifted_values = np.array([0.1, 0.1, 0.1, 0.7]) # Massive shift in priorities
    
    print("Checking aligned values...")
    assert orchestrator.monitor.check_drift(aligned_values) == True
    
    print("Checking drifted values...")
    assert orchestrator.monitor.check_drift(drifted_values) == False

    # 3. Test Anomaly Detection
    print("\n[Testing Latent Anomaly Detection]")
    normal_activation = np.zeros(10)
    anomalous_activation = np.ones(10) * 15.0 # High amplitude deviation
    
    print("Checking normal activation...")
    assert orchestrator.interpretability.detect_latent_anomaly(normal_activation) == False
    
    print("Checking anomalous activation...")
    assert orchestrator.interpretability.detect_latent_anomaly(anomalous_activation) == True

    print("\nPhase 5 Verification Complete.")

if __name__ == "__main__":
    test_phase_5()
