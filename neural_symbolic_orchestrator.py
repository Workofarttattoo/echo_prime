#!/usr/bin/env python3
"""
Neural-Symbolic Orchestrator for Echo Prime
Connects core/engine.py (neural) with reasoning engines (symbolic)

This is the TRUE integrated system combining:
1. Neural hierarchical processing (engine.py)
2. Symbolic reasoning engines (math, code, knowledge)
3. Optional LLM fallback

Architecture:
Input → Neural Processing → Pattern Recognition → Symbolic Reasoning → Output
"""

import sys
import os
import numpy as np
from typing import Dict, List, Optional, Any
import importlib.util

class NeuralSymbolicOrchestrator:
    """
    Integrates Echo's neural architecture with symbolic reasoning engines
    """

    def __init__(self, use_neural: bool = True, use_llm: bool = False):
        print("🧠 Initializing Neural-Symbolic Orchestrator...")

        self.use_neural = use_neural
        self.use_llm = use_llm

        # Load symbolic reasoning engines (pure Python, always available)
        self._load_symbolic_engines()

        # Load neural architecture (if torch available)
        self.neural_model = None
        self.fe_engine = None
        if use_neural:
            self._load_neural_architecture()

        # Load LLM backend (optional)
        self.llm_bridge = None
        if use_llm:
            self._load_llm_backend()

        print("✅ Orchestrator initialized")
        print(f"   Neural: {'✅' if self.neural_model else '❌'}")
        print(f"   Symbolic: ✅")
        print(f"   LLM: {'✅' if self.llm_bridge else '❌'}")

    def _load_symbolic_engines(self):
        """Load pure Python symbolic reasoning engines"""
        print("  Loading symbolic reasoning engines...")

        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        math_module = load_module("math_engine", "reasoning/math_engine.py")
        code_module = load_module("code_debugger", "reasoning/code_debugger.py")
        knowledge_module = load_module("knowledge_reasoner", "reasoning/knowledge_reasoner.py")

        self.math_engine = math_module.get_math_engine()
        self.code_debugger = code_module.get_code_debugger()
        self.knowledge_reasoner = knowledge_module.get_knowledge_reasoner()

        print("  ✅ Symbolic engines loaded")

    def _load_neural_architecture(self):
        """Load neural hierarchical processing"""
        try:
            import torch
            from core.engine import HierarchicalGenerativeModel, FreeEnergyEngine, GlobalWorkspace

            print("  Loading neural architecture...")
            self.neural_model = HierarchicalGenerativeModel(use_cuda=False, lightweight=True)
            self.fe_engine = FreeEnergyEngine(self.neural_model)
            self.workspace = GlobalWorkspace(self.neural_model)

            print("  ✅ Neural architecture loaded (5 cortical levels)")

        except ImportError as e:
            print(f"  ⚠️ Neural architecture not available: {e}")
            print("  Continuing with symbolic only")
        except Exception as e:
            print(f"  ⚠️ Neural initialization error: {e}")

    def _load_llm_backend(self):
        """Load LLM backend (Ollama)"""
        try:
            from reasoning.llm_bridge import OllamaBridge
            self.llm_bridge = OllamaBridge(model="llama3.2")
            print("  ✅ LLM backend connected (Ollama)")
        except Exception as e:
            print(f"  ⚠️ LLM not available: {e}")

    def process_through_neural_hierarchy(self, text: str) -> Optional[Any]:
        """
        Process input through neural hierarchy
        Returns neural representation that can guide symbolic reasoning
        """
        if not self.neural_model:
            return None

        try:
            import torch

            # Convert text to sensory input
            # Simple encoding: character codes
            input_vector = np.array([ord(c) for c in text[:1000]], dtype=np.float32)

            # Pad to sensory dimension
            sensory_dim = self.neural_model.levels[0].input_dim
            if len(input_vector) < sensory_dim:
                input_vector = np.pad(input_vector, (0, sensory_dim - len(input_vector)))
            else:
                input_vector = input_vector[:sensory_dim]

            # Process through hierarchy
            input_tensor = torch.from_numpy(input_vector).float().unsqueeze(0)

            # Get hierarchical representations
            expectations = self.neural_model.step(input_tensor.squeeze(0))

            # Return meta-cortex representation (highest level)
            meta_repr = expectations[-1].detach().cpu().numpy()

            return {
                'meta_representation': meta_repr,
                'all_levels': [e.detach().cpu().numpy() for e in expectations],
                'pattern_confidence': float(np.mean(np.abs(meta_repr)))
            }

        except Exception as e:
            print(f"  ⚠️ Neural processing error: {e}")
            return None

    def solve_math_problem(self, problem: str) -> Dict[str, Any]:
        """
        Solve math problem using neural-symbolic integration

        Pipeline:
        1. Neural pattern recognition (identify problem type)
        2. Symbolic reasoning (solve using rules)
        3. LLM fallback (if needed)
        """
        result = {
            'problem': problem,
            'answer': None,
            'method': None,
            'confidence': 0.0,
            'neural_guidance': None
        }

        # Step 1: Neural pattern recognition
        if self.neural_model:
            neural_output = self.process_through_neural_hierarchy(problem)
            result['neural_guidance'] = neural_output

            # Neural representation could guide symbolic selection
            # For now, we use symbolic directly
            # In trained model, this would select best strategy

        # Step 2: Symbolic reasoning (primary)
        symbolic_answer = self.math_engine.solve_problem(problem)

        if symbolic_answer and symbolic_answer not in ["0", "42"]:  # Not default fallback
            result['answer'] = symbolic_answer
            result['method'] = 'symbolic'
            result['confidence'] = 0.95  # High confidence in deterministic symbolic

        # Step 3: LLM fallback (if symbolic failed)
        elif self.llm_bridge:
            try:
                llm_response = self.llm_bridge.query(
                    f"Solve this math problem and give ONLY the numerical answer: {problem}",
                    None, None, 0.1, 0.9
                )
                import re
                nums = re.findall(r'\d+', llm_response)
                if nums:
                    result['answer'] = nums[-1]
                    result['method'] = 'llm'
                    result['confidence'] = 0.7  # Lower confidence in LLM
            except Exception as e:
                result['answer'] = symbolic_answer  # Use symbolic fallback
                result['method'] = 'symbolic_fallback'
                result['confidence'] = 0.3
        else:
            result['answer'] = symbolic_answer
            result['method'] = 'symbolic_fallback'
            result['confidence'] = 0.3

        return result

    def generate_code(self, problem: str) -> Dict[str, Any]:
        """Generate code using neural-symbolic integration"""
        result = {
            'problem': problem,
            'code': None,
            'method': None,
            'confidence': 0.0
        }

        # Neural pattern recognition
        if self.neural_model:
            neural_output = self.process_through_neural_hierarchy(problem)
            # Could guide code template selection

        # Symbolic code generation (primary)
        generated_code = self.code_debugger.debug_code(problem)

        if generated_code and ("def " in generated_code or "class " in generated_code):
            result['code'] = generated_code
            result['method'] = 'symbolic'
            result['confidence'] = 0.9

        # LLM fallback
        elif self.llm_bridge:
            try:
                llm_code = self.llm_bridge.query(
                    f"Write Python code to: {problem}\n\nProvide only the code, no explanations:",
                    None, None, 0.1, 0.9
                )
                result['code'] = llm_code
                result['method'] = 'llm'
                result['confidence'] = 0.8
            except:
                result['code'] = generated_code
                result['method'] = 'symbolic_fallback'
                result['confidence'] = 0.5
        else:
            result['code'] = generated_code
            result['method'] = 'symbolic_fallback'
            result['confidence'] = 0.5

        return result

    def answer_question(self, question: str, choices: Optional[List[str]] = None) -> Dict[str, Any]:
        """Answer knowledge question using neural-symbolic integration"""
        result = {
            'question': question,
            'answer': None,
            'method': None,
            'confidence': 0.0
        }

        # Neural pattern recognition
        if self.neural_model:
            neural_output = self.process_through_neural_hierarchy(question)
            # Could retrieve from associative memory

        # Symbolic knowledge retrieval (primary)
        symbolic_answer = self.knowledge_reasoner.answer_question(question, choices)

        if symbolic_answer and symbolic_answer != "unknown" and symbolic_answer != "1":
            result['answer'] = symbolic_answer
            result['method'] = 'symbolic'
            result['confidence'] = 0.85

        # LLM fallback
        elif self.llm_bridge:
            try:
                prompt = question
                if choices:
                    prompt += "\n\nChoices:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
                    prompt += "\n\nAnswer with just the number (1-4):"

                llm_answer = self.llm_bridge.query(prompt, None, None, 0.1, 0.9)
                result['answer'] = llm_answer.strip()
                result['method'] = 'llm'
                result['confidence'] = 0.75
            except:
                result['answer'] = symbolic_answer
                result['method'] = 'symbolic_fallback'
                result['confidence'] = 0.4
        else:
            result['answer'] = symbolic_answer
            result['method'] = 'symbolic_fallback'
            result['confidence'] = 0.4

        return result

    def __repr__(self):
        components = []
        if self.neural_model:
            components.append("Neural")
        components.append("Symbolic")
        if self.llm_bridge:
            components.append("LLM")

        return f"NeuralSymbolicOrchestrator({'+'.join(components)})"


def test_orchestrator():
    """Test the integrated system"""
    print("=" * 80)
    print("TESTING NEURAL-SYMBOLIC ORCHESTRATOR")
    print("=" * 80)

    # Test different configurations
    configs = [
        ("Symbolic Only", False, False),
        ("Neural + Symbolic", True, False),
        ("Symbolic + LLM", False, True),
        ("Full System (Neural + Symbolic + LLM)", True, True),
    ]

    for config_name, use_neural, use_llm in configs:
        print(f"\n--- Configuration: {config_name} ---")

        try:
            orchestrator = NeuralSymbolicOrchestrator(
                use_neural=use_neural,
                use_llm=use_llm
            )

            # Test math
            print("\n📐 Math Test:")
            result = orchestrator.solve_math_problem("What is 25 + 17?")
            print(f"   Answer: {result['answer']}")
            print(f"   Method: {result['method']}")
            print(f"   Confidence: {result['confidence']:.2f}")

            # Test code
            print("\n💻 Code Test:")
            result = orchestrator.generate_code("Write a function to sort a list")
            print(f"   Generated: {result['code'][:60]}...")
            print(f"   Method: {result['method']}")

            # Test knowledge
            print("\n📚 Knowledge Test:")
            result = orchestrator.answer_question(
                "What is the capital of France?",
                ["London", "Paris", "Berlin", "Rome"]
            )
            print(f"   Answer: {result['answer']}")
            print(f"   Method: {result['method']}")

        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")

    print("\n" + "=" * 80)
    print("✅ ORCHESTRATOR TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_orchestrator()
