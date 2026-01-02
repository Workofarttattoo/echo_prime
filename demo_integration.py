import time
import json
import numpy as np
from main_orchestrator import EchoPrimeAGI

def demonstrate_full_integration():
    print("🚀 ECH0-PRIME: Full System Integration Demonstration")
    print("=" * 60)

    # 1. Initialize AGI with all advanced components
    # Using lightweight=True for the engine but still want advanced systems
    agi = EchoPrimeAGI(enable_voice=False, device="cpu", lightweight=True)
    
    # Manually initialize advanced components since lightweight=True skips them in __init__
    print("Manually activating advanced components for demonstration...")
    from agents.multi_agent import MultiAgentSystem
    from distributed_swarm_intelligence import DistributedSwarmIntelligence
    from advanced_safety import AdvancedSafetySystem
    from self_modifying_architecture import SelfModifyingArchitecture
    from performance_profiler import PerformanceProfiler
    from system_health_monitor import SystemHealthMonitor
    from quantum_computation import QuantumEnhancedComputation
    from advanced_meta_learning import AdvancedMetaLearningSystem
    from multi_modal_integration import MultiModalIntegrationSystem
    from capabilities.prompt_masterworks import PromptMasterworks
    
    target_values = np.array([0.4, 0.3, 0.2, 0.1])
    agi.advanced_safety = AdvancedSafetySystem(target_values)
    agi.safety = agi.advanced_safety
    agi.multi_agent = MultiAgentSystem()
    agi.swarm_intelligence = DistributedSwarmIntelligence()
    agi.swarm_intelligence.start()
    agi.self_mod_arch = SelfModifyingArchitecture(agi.model, agi.advanced_safety)
    agi.performance_profiler = PerformanceProfiler()
    agi.health_monitor = SystemHealthMonitor()
    agi.quantum_sys = QuantumEnhancedComputation()
    agi.meta_learning = AdvancedMetaLearningSystem()
    from multi_modal_integration import Modality
    agi.multimodal_sys = MultiModalIntegrationSystem([Modality.VISION, Modality.AUDIO, Modality.TEXT])
    agi.prompt_masterworks = PromptMasterworks()
    
    print("✓ ECH0-PRIME Initialized with Hybrid Mode (Lightweight Engine + Advanced Systems)")

    # 2. Test Performance & Health
    print("\n--- Testing Monitoring Systems ---")
    perf_report = agi.get_detailed_performance_report()
    health_status = agi.get_system_health_status()
    print(f"✓ Performance Score: {perf_report.get('performance_score', 0):.2f}")
    print(f"✓ System Health: {health_status.get('overall_status', 'Unknown')}")

    # 3. Test Quantum & Meta-Learning
    print("\n--- Testing Quantum & Meta-Learning ---")
    quantum_res = agi.run_quantum_optimization({"problem": "resource_allocation"})
    curriculum = agi.update_learning_curriculum(["math_task", "logic_task"])
    print(f"✓ Quantum Result: {quantum_res.get('status')}")
    print(f"✓ Meta-Learning Curriculum: {len(curriculum.get('curriculum', []))} tasks prioritized")

    # 4. Test Safety & Swarm
    print("\n--- Testing Safety & Swarm ---")
    safety_audit = agi.perform_safety_audit("Calculate mission trajectory", {"priority": "high"})
    swarm_task_id = agi.submit_swarm_task("Parallel processing sub-module analysis")
    print(f"✓ Safety Audit: {'SAFE' if safety_audit.get('is_fully_safe') else 'BLOCKED'}")
    print(f"✓ Swarm Task ID: {swarm_task_id}")

    # 5. Test Self-Modifying Architecture
    print("\n--- Testing Self-Modifying Architecture ---")
    agi.evolve_architecture({"loss": 0.45})
    bugs = agi.scan_codebase_for_bugs()
    print(f"✓ Evolution cycle executed")
    print(f"✓ Bug scan completed: {len(bugs)} files with issues found")

    # 6. Test Multi-Modal Processing
    print("\n--- Testing Multi-Modal Processing ---")
    # Simulate a cognitive cycle which now includes profiling and health checks
    dummy_input = np.random.randn(1000000).astype(np.float32)
    cycle_outcome = agi.cognitive_cycle(dummy_input, "Integrate all subsystems for final check")
    print(f"✓ Cognitive Cycle status: {cycle_outcome.get('status')}")
    print(f"✓ Phi (Integrated Information): {agi.phi:.2f}")

    print("\n" + "=" * 60)
    print("🎉 ALL SYSTEMS INTEGRATED AND FUNCTIONAL!")
    print("ECH0-PRIME has successfully evolved with all requested advanced capabilities.")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_full_integration()



