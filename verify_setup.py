#!/usr/bin/env python3
"""
ECH0-PRIME Setup Verification Script
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Quick verification that all components are working.
"""

import sys
import os

def test_imports():
    """Test all module imports."""
    print("🔍 Testing module imports...")
    try:
        import numpy as np
        from core import (
            HierarchicalGenerativeModel, FreeEnergyEngine, GlobalWorkspace,
            QuantumAttentionHead, CoherenceShaper, VisionBridge, ActuatorBridge
        )
        from memory import MemoryManager
        from learning import CSALearningSystem
        from reasoning import ReasoningOrchestrator, OllamaBridge
        from safety import SafetyOrchestrator
        from training import TrainingPipeline
        print("✓ All core modules import successfully")
        print(f"  NumPy version: {np.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_dependencies():
    """Test critical dependencies."""
    print("\n🔍 Testing dependencies...")
    try:
        import numpy
        import PIL
        import requests
        import speech_recognition
        print("✓ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Dependency missing: {e}")
        return False

def test_ollama():
    """Test Ollama connectivity."""
    print("\n🔍 Testing Ollama connection...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✓ Ollama is running ({len(models)} models available)")

            # Check for llama3.2
            has_llama = any("llama3.2" in m.get("name", "") for m in models)
            if has_llama:
                print("  ✓ llama3.2 model found")
            else:
                print("  ⚠️  llama3.2 not found (run: ollama pull llama3.2)")
            return True
        else:
            print(f"⚠️  Ollama responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️  Ollama not accessible: {e}")
        print("   Start with: ollama serve")
        return False

def test_basic_functionality():
    """Test basic system functionality."""
    print("\n🔍 Testing basic functionality...")
    try:
        from core import HierarchicalGenerativeModel, FreeEnergyEngine
        import numpy as np

        lightweight_mode = os.environ.get("ECH0_VERIFY_LIGHTWEIGHT", "1").lower() not in (
            "0",
            "false",
            "no",
        )

        # Create model
        model = HierarchicalGenerativeModel(lightweight=lightweight_mode)
        fe_engine = FreeEnergyEngine(model)

        # Test prediction
        sensory_dim = model.levels[0].input_dim
        test_input = np.random.randn(sensory_dim)
        fe = fe_engine.optimize(sensory_input=test_input, iterations=1)

        profile_label = "lite" if lightweight_mode else "full"
        print(f"✓ Core engine functional (FE: {fe:.2f}, profile: {profile_label})")
        return True
    except Exception as e:
        print(f"❌ Core engine test failed: {e}")
        return False

def test_audio_system():
    """Test audio system (non-blocking)."""
    print("\n🔍 Testing audio system...")
    try:
        from core import AudioBridge
        # Don't actually initialize (can hang on permissions)
        print("✓ AudioBridge module available")
        print("  (Microphone permission required for actual use)")
        return True
    except Exception as e:
        print(f"❌ AudioBridge test failed: {e}")
        return False

def test_directories():
    """Test required directories exist."""
    print("\n🔍 Checking directories...")
    dirs = [
        "sensory_input",
        "audio_input",
        "memory_data",
        "dashboard/data"
    ]
    all_exist = True
    for d in dirs:
        if os.path.exists(d):
            print(f"  ✓ {d}")
        else:
            print(f"  ⚠️  {d} (will be created on first run)")
            all_exist = False
    return True  # Non-critical

def main():
    """Run all verification tests."""
    print("=" * 50)
    print("ECH0-PRIME Setup Verification")
    print("=" * 50)

    results = {
        "Imports": test_imports(),
        "Dependencies": test_dependencies(),
        "Ollama": test_ollama(),
        "Core Engine": test_basic_functionality(),
        "Audio System": test_audio_system(),
        "Directories": test_directories(),
    }

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_passed = True
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test}")
        if not result:
            all_passed = False

    print("=" * 50)

    if all_passed:
        print("\n🎉 All systems ready! You can run:")
        print("   source venv/bin/activate")
        print("   python main_orchestrator.py")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        print("   Run ./setup.sh to fix common issues")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
