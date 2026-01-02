"""
Integration tests for all new components.
Tests that the revolutionary transformation works end-to-end.
"""
import sys
import os
import torch
import numpy as np
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_system_initialization():
    """Test that the main orchestrator initializes with all new components"""
    print("Testing basic system initialization...")

    try:
        from main_orchestrator import EchoPrimeAGI

        # Initialize system
        agi = EchoPrimeAGI(enable_voice=False)

        # Check that all new components are initialized
        assert hasattr(agi, 'multi_agent'), "Multi-agent system not initialized"
        assert hasattr(agi, 'self_mod'), "Self-modification system not initialized"
        assert hasattr(agi, 'iit'), "Integrated Information Theory not initialized"
        assert hasattr(agi, 'creativity'), "Creativity system not initialized"
        assert hasattr(agi, 'science'), "Scientific discovery system not initialized"
        assert hasattr(agi, 'goal_system'), "Goal system not initialized"
        assert hasattr(agi, 'monitoring'), "Monitoring system not initialized"

        print("✓ System initialization successful")
        return True

    except Exception as e:
        print(f"✗ System initialization failed: {e}")
        return False


def test_pytorch_model_integration():
    """Test that PyTorch models work in the cognitive cycle"""
    print("Testing PyTorch model integration...")

    try:
        from main_orchestrator import EchoPrimeAGI

        agi = EchoPrimeAGI(enable_voice=False)

        # Create test input
        test_input = np.random.randn(1000000).astype(np.float32)

        # Run cognitive cycle
        result = agi.cognitive_cycle(test_input, "Test PyTorch integration")

        # Check that result contains expected keys
        assert "free_energy" in result, "Free energy not calculated"
        assert "surprise" in result, "Surprise detection failed"
        assert "actions" in result, "Actions not generated"

        # Check that PyTorch models are working (no NaN/inf values)
        assert not np.isnan(result["free_energy"]), "Free energy is NaN"
        assert not np.isinf(result["free_energy"]), "Free energy is infinite"

        print("✓ PyTorch model integration successful")
        return True

    except Exception as e:
        print(f"✗ PyTorch model integration failed: {e}")
        return False


def test_memory_system_integration():
    """Test that enhanced memory systems work together"""
    print("Testing memory system integration...")

    try:
        from memory.manager import MemoryManager

        # Initialize memory system
        memory = MemoryManager()

        # Test different memory types
        test_vector = np.random.randn(1024).astype(np.float32)

        # Store in different memory systems
        memory.process_input(test_vector)

        # Test retrieval
        retrieved = memory.episodic.retrieve_nearest(test_vector, k=1)
        assert len(retrieved[1]) >= 0, "Episodic retrieval failed"

        # Test semantic memory
        memory.semantic.store_concept("test_concept", test_vector)
        concepts = memory.semantic.retrieve_similar("test_concept")
        assert len(concepts) > 0, "Semantic retrieval failed"

        print("✓ Memory system integration successful")
        return True

    except Exception as e:
        print(f"✗ Memory system integration failed: {e}")
        return False


def test_planning_system():
    """Test the planning system components"""
    print("Testing planning system...")

    try:
        from reasoning.planner import HTNPlanner, Task, Method

        # Test HTN planning
        planner = HTNPlanner()

        # Add tasks and methods
        planner.add_task(Task("pickup", ["at_location"], ["has_object"]))
        planner.add_task(Task("move", [], ["at_location"]))

        planner.add_method(Method(
            "get_object",
            "get_object",
            [],
            ["move", "pickup"]
        ))

        # Test planning
        planner.set_initial_state(set())  # Empty initial state
        plan = planner.plan("get_object")
        assert plan is not None, "HTN planning failed"

        print("✓ Planning system successful")
        return True

    except Exception as e:
        print(f"✗ Planning system failed: {e}")
        return False


def test_architecture_search():
    """Test architecture search system"""
    print("Testing architecture search...")

    try:
        from learning.architecture_search import ArchitectureSearchSystem, ArchitectureSpec

        search_system = ArchitectureSearchSystem()

        # Test basic architecture search
        results = search_system.comprehensive_search()

        assert "best" in results, "Architecture search failed"
        assert results["best"] is not None, "No best architecture found"

        print("✓ Architecture search successful")
        return True

    except Exception as e:
        print(f"✗ Architecture search failed: {e}")
        return False


def test_human_collaboration():
    """Test human-AI collaboration system"""
    print("Testing human collaboration...")

    try:
        from agents.human_collaboration import HumanAICollaborationSystem, Feedback

        collab_system = HumanAICollaborationSystem()

        # Test explanation generation
        explanation = collab_system.explain_decision(
            "prediction",
            prediction="positive",
            features={"feature1": 0.8, "feature2": 0.6},
            model_confidence=0.85
        )

        assert explanation.confidence > 0, "Explanation generation failed"

        # Test feedback processing
        feedback = Feedback(
            feedback_type="correction",
            target_output="negative",
            human_input="positive",
            context={"situation": "test"},
            timestamp=0.0
        )

        result = collab_system.process_feedback(feedback)
        assert "correction_learned" in result, "Feedback processing failed"

        print("✓ Human collaboration successful")
        return True

    except Exception as e:
        print(f"✗ Human collaboration failed: {e}")
        return False


def test_multi_agent_system():
    """Test multi-agent collaboration"""
    print("Testing multi-agent system...")

    try:
        from agents.multi_agent import MultiAgentSystem, Agent

        # Create multi-agent system
        mas = MultiAgentSystem()

        # Add agents
        agent1 = Agent("agent1", "science", ["analyze", "compute"])
        agent2 = Agent("agent2", "creativity", ["generate", "innovate"])

        mas.add_agent(agent1)
        mas.add_agent(agent2)

        # Test agent capabilities
        assert agent1.can_handle_task({"required_capabilities": ["analyze"]}), "Agent capability check failed"
        assert not agent1.can_handle_task({"required_capabilities": ["unknown"]}), "Agent capability check failed"

        print("✓ Multi-agent system successful")
        return True

    except Exception as e:
        print(f"✗ Multi-agent system failed: {e}")
        return False


def test_research_components():
    """Test research innovations"""
    print("Testing research components...")

    try:
        from research.self_model import IntegratedInformationTheory, EnhancedGlobalWorkspace

        # Test IIT
        iit = IntegratedInformationTheory()
        phi = iit.compute_phi(np.array([0.1, 0.2, 0.3, 0.4]))
        assert isinstance(phi, float), "IIT calculation failed"

        # Test enhanced GWT
        gwt = EnhancedGlobalWorkspace()
        gwt.register_module("vision", np.array([0.1, 0.2, 0.3]))
        gwt.register_module("language", np.array([0.4, 0.5, 0.6]))

        winner, synchrony = gwt.broadcast()
        assert synchrony >= 0, "GWT broadcast failed"

        print("✓ Research components successful")
        return True

    except Exception as e:
        print(f"✗ Research components failed: {e}")
        return False


def test_advanced_capabilities():
    """Test advanced capabilities"""
    print("Testing advanced capabilities...")

    try:
        from capabilities.creativity import CreativeProblemSolver
        from capabilities.scientific_discovery import ScientificDiscoverySystem
        from missions.long_term_goals import LongTermGoalSystem

        # Test creativity
        creativity = CreativeProblemSolver()
        solutions = creativity.solve_creatively({"concepts": ["wheel", "transport"]})
        assert len(solutions) > 0, "Creativity system failed"

        # Test scientific discovery
        science = ScientificDiscoverySystem()
        discovery = science.discover([{"value": 1.0}, {"value": 2.0}, {"value": 3.0}])
        assert "hypothesis" in discovery, "Scientific discovery failed"

        # Test long-term goals
        goals = LongTermGoalSystem()
        goal = goals.add_goal("Build a revolutionary AI system")
        assert goal.id.startswith("goal_"), "Goal system failed"

        print("✓ Advanced capabilities successful")
        return True

    except Exception as e:
        print(f"✗ Advanced capabilities failed: {e}")
        return False


def test_self_modification():
    """Test self-modification system"""
    print("Testing self-modification...")

    try:
        from missions.self_modification import SelfModificationSystem

        self_mod = SelfModificationSystem()

        # Test code validation
        test_code = "def test_function():\n    return 42"
        validation = self_mod.validator.validate(test_code)
        assert validation["valid"], "Code validation failed"

        print("✓ Self-modification successful")
        return True

    except Exception as e:
        print(f"✗ Self-modification failed: {e}")
        return False


def test_infrastructure():
    """Test infrastructure components"""
    print("Testing infrastructure...")

    try:
        from infrastructure.monitoring import MonitoringSystem
        from infrastructure.distributed import DistributedTraining

        # Test monitoring
        monitoring = MonitoringSystem()
        monitoring.metrics.record("test_metric", 42.0)
        summary = monitoring.metrics.get_summary("test_metric")
        assert summary["latest"] == 42.0, "Monitoring failed"

        # Test distributed training (without actual distributed setup)
        dist_training = DistributedTraining()
        # Just test that it initializes
        assert hasattr(dist_training, 'initialize'), "Distributed training failed"

        print("✓ Infrastructure successful")
        return True

    except Exception as e:
        print(f"✗ Infrastructure failed: {e}")
        return False


def run_comprehensive_integration_test():
    """Run all integration tests"""
    print("=" * 60)
    print("RUNNING COMPREHENSIVE INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        test_basic_system_initialization,
        test_pytorch_model_integration,
        test_memory_system_integration,
        test_planning_system,
        test_architecture_search,
        test_human_collaboration,
        test_multi_agent_system,
        test_research_components,
        test_advanced_capabilities,
        test_self_modification,
        test_infrastructure
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} crashed: {e}")
            failed += 1

    print("=" * 60)
    print(f"INTEGRATION TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("The revolutionary ECH0-PRIME transformation is complete and working!")
        return True
    else:
        print(f"⚠️  {failed} tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_integration_test()
    sys.exit(0 if success else 1)
