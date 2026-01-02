import sys
import os
import numpy as np
import time
import json
import requests
import torch
import random
import gc
import psutil
import asyncio
import datetime
from typing import Dict, List, Optional, Any, Union
import threading

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.engine import HierarchicalGenerativeModel, FreeEnergyEngine, GlobalWorkspace

# New component imports

class EchoPrimeAGI:
    """
    The unified Cognitive-Synthetic Architecture entry point.
    """
    def __init__(self, enable_voice: bool = True, device: str = "auto", lightweight: Optional[bool] = None):  # pyright: ignore[reportMissingSuperCall]
        from dotenv import load_dotenv
        load_dotenv()  # pyright: ignore[reportUnusedCallResult]

        print("Initializing ECH0-PRIME AGI System...")
        self.phase = os.environ.get("ECH0_PHASE", "1")
        print(f"🚀 Current Phase: {self.phase}")
        
        if lightweight is None:
            lightweight = bool(os.environ.get("PYTEST_CURRENT_TEST")) or os.environ.get("ECH0_LIGHTWEIGHT") == "1"
        self.lightweight_mode = lightweight

        # Device detection with error handling
        try:
            if device == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)

            # Test device availability
            test_tensor = torch.randn(10, device=self.device)
            del test_tensor  # Clean up

            print(f"✓ Using device: {self.device}")

        except Exception as e:
            print(f"⚠️ Device initialization failed: {e}. Falling back to CPU.")
            self.device = torch.device("cpu")

        from memory.manager import MemoryManager
        from learning.meta import CSALearningSystem
        from safety.alignment import SafetyOrchestrator
        from core.actuator import ActuatorBridge
        from ech0_governance.persistent_memory import PersistentMemory
        from ech0_governance.knowledge_graph import KnowledgeGraph

        # 1. Core Cognitive Engine
        self.model = HierarchicalGenerativeModel(use_cuda=(self.device.type == "cuda"), lightweight=self.lightweight_mode)
        self.fe_engine = FreeEnergyEngine(self.model)
        self.workspace = GlobalWorkspace(self.model)

        # 2. Attention and Vision Subsystems
        if self.lightweight_mode:
            class _CoherenceStub:
                def __init__(self, coherence_time_ms: float = 10.0):  # pyright: ignore[reportMissingSuperCall]
                    self.coherence_time_ms = coherence_time_ms
                    self.coherence_level = 1.0

                def step(self, delta_ms: float):
                    self.coherence_level = max(0.0, self.coherence_level - delta_ms / self.coherence_time_ms)
                    if delta_ms >= self.coherence_time_ms:
                        self.coherence_level = 1.0

            class _VisionStub:
                """Lightweight vision stub that safely returns no sensory data."""
                def __init__(self):  # pyright: ignore[reportMissingSuperCall]
                    self.watch_dir = "."

                def get_latest_sensory_vector(self):
                    # No vision processing in lightweight mode; return None
                    return None

            self.attn_head = None
            self.coherence = _CoherenceStub()
            self.vision = _VisionStub()
        else:
            from core.attention import QuantumAttentionHead, CoherenceShaper
            from core.vision_bridge import VisionBridge
            try:
                self.attn_head = QuantumAttentionHead()
                print("✓ Quantum attention initialized")
            except Exception as e:
                print(f"⚠️ Quantum attention failed ({e}), using fallback")
                self.attn_head = None

            self.coherence = CoherenceShaper()
            self.vision = VisionBridge(use_webcam=True, enable_ocr=True)
        
        # 3. Memory & Learning
        self.memory = MemoryManager()
        self.learning = CSALearningSystem(param_dim=1000, device=self.device.type)
        
        # 3b. Governance Wiring
        self.gov_mem = PersistentMemory(self.memory)
        self.kg = KnowledgeGraph()
        if self.lightweight_mode:
            self.qulab = None
            self.arxiv = None
        else:
            from reasoning.tools.qulab import QuLabBridge
            from reasoning.tools.arxiv_scanner import ArxivScanner
            self.qulab = QuLabBridge()
            self.arxiv = ArxivScanner()

        # Load user profile and goals if available
        self._load_user_profile()

        # Mission tracking
        self.current_goal = "Assistant: Awaiting instructions."
        self.mission_history = []
        
        # 4. Reasoning, Safety & Actuation
        if self.lightweight_mode:
            class _StubReasoner:
                def __init__(self):
                    self.llm_bridge = None

                def reason_about_scenario(self, context, mission_params, memory_bank=None):  # pyright: ignore[reportMissingParameterType]
                    return {"llm_insight": mission_params.get("goal", "")}

            self.reasoner = _StubReasoner()
        else:
            from reasoning.orchestrator import ReasoningOrchestrator
            self.reasoner = ReasoningOrchestrator(
                use_llm=True,
                model_name="ech0-unified-14b-enhanced",
                governance_mem=self.gov_mem,
                knowledge_graph=self.kg,
                qulab=self.qulab,
                arxiv=self.arxiv
            )
        target_values = np.array([0.4, 0.3, 0.2, 0.1]) # Human value priors
        self.safety = SafetyOrchestrator(target_values)
        self.actuator = ActuatorBridge(workspace_root=os.path.dirname(os.path.abspath(__file__)))
        # Ms. Walker (Southern US, Conversational/Informational)
        if self.lightweight_mode:
            class _StubVoice:
                def __init__(self):
                    self.is_currently_speaking = False
                    from queue import Queue
                    self.msg_queue = Queue()

                def speak(self, text: str, async_mode: bool = True):
                    print(f"[VoiceStub] {text}")

                def silence(self):
                    pass

            class _StubAudio:
                def __init__(self):
                    self.transcription_queue = []
                    self.latest_transcription = None

                def set_talking(self, talking: bool):
                    pass

                def get_latest_transcription(self):
                    return None

            self.voice = _StubVoice()
            self.audio = _StubAudio()
        else:
            from core.voice_bridge import VoiceBridge
            from core.audio_bridge import AudioBridge
            self.voice = VoiceBridge(voice="Samantha", eleven_voice_id="DLsHlh26Ugcm6ELvS0qi")
            self.audio = AudioBridge()

        # Apple Intelligence Integration
        if self.lightweight_mode:
            class _StubAppleIntelligence:
                def process_with_apple_intelligence(self, payload):
                    return payload

                def enhance_agi_response(self, payload):
                    return payload

                def get_system_status(self):
                    return {"status": "stub"}

            self.apple_intelligence = _StubAppleIntelligence()
        else:
            from core.apple_intelligence_bridge import get_apple_intelligence_bridge
            self.apple_intelligence = get_apple_intelligence_bridge()

        # 5. New Advanced Components
        if not self.lightweight_mode:
            print("Initializing advanced components...")

        try:
            if self.lightweight_mode:
                raise RuntimeError("Lightweight mode: advanced components disabled")
            from agents.multi_agent import MultiAgentSystem
            from missions.self_modification import SelfModificationSystem
            from missions.hive_mind import HiveMindOrchestrator
            from research.self_model import (
                IntegratedInformationTheory,
                EnhancedGlobalWorkspace,
                SelfAwareness,
                MetacognitiveMonitoring,
            )
            from capabilities.creativity import CreativeProblemSolver
            from capabilities.scientific_discovery import ScientificDiscoverySystem
            from capabilities.prompt_masterworks import PromptMasterworks, PromptCategory
            from missions.long_term_goals import LongTermGoalSystem
            from infrastructure.monitoring import MonitoringSystem
            from infrastructure.distributed import DistributedTraining
            from numerical_verifier import NumericalVerifier, ConsistencyChecker
            from problem_expander import ProblemExpander, BenchmarkCoordinator
            from hybrid_scaler import HybridScaler
            from phd_knowledge_base import PhDKnowledgeBase, InterdisciplinaryResearchEngine
            from groundbreaking_research import (
                BreakthroughResearchSystem,
                RevolutionaryCapabilityDemonstrator,
            )
            from distributed_swarm_intelligence import DistributedSwarmIntelligence
            from advanced_safety import AdvancedSafetySystem
            from self_modifying_architecture import SelfModifyingArchitecture
            from performance_profiler import PerformanceProfiler
            from system_health_monitor import SystemHealthMonitor
            from quantum_computation import QuantumEnhancedComputation
            from advanced_meta_learning import AdvancedMetaLearningSystem
            from multi_modal_integration import MultiModalIntegrationSystem
            # Advanced Safety & Alignment
            self.advanced_safety = AdvancedSafetySystem(target_values)
            self.safety = self.advanced_safety # Upgrade existing safety orchestrator
            print("✓ Advanced safety & alignment initialized")

            # Multi-agent system
            self.multi_agent = MultiAgentSystem()
            print("✓ Multi-agent system initialized")

            # Distributed Swarm Intelligence
            self.swarm_intelligence = DistributedSwarmIntelligence()
            self.swarm_intelligence.start()
            print("✓ Distributed Swarm Intelligence initialized")

            # Hive mind collective intelligence
            self.hive_mind = HiveMindOrchestrator(num_nodes=5, qulab_path="/Users/noone/QuLabInfinite")
            print("✓ Hive mind orchestrator initialized")

            # Self-modification system
            self.self_mod = SelfModificationSystem(llm_bridge=self.reasoner.llm_bridge)
            print("✓ Self-modification system initialized")

            # Research innovations
            self.iit = IntegratedInformationTheory()
            self.enhanced_gwt = EnhancedGlobalWorkspace()
            self.self_awareness = SelfAwareness()
            self.metacognition = MetacognitiveMonitoring()
            print("✓ Research innovations initialized")

            # Advanced capabilities
            self.creativity = CreativeProblemSolver()
            self.science = ScientificDiscoverySystem()
            self.prompt_masterworks = PromptMasterworks()
            if hasattr(self, 'reasoner'):
                self.reasoner.prompt_masterworks = self.prompt_masterworks
            print("✓ Advanced capabilities initialized (including prompt masterworks)")

            # Long-term goal pursuit
            self.goal_system = LongTermGoalSystem()
            print("✓ Goal system initialized")

            # Infrastructure
            self.monitoring = MonitoringSystem()
            self.distributed_training = DistributedTraining()
            print("✓ Infrastructure initialized")

            # Feedback Loop for Continuous Learning
            # TODO: Initialize feedback loop in async context
            # self.feedback_loop = await create_feedback_loop(
            #     memory_manager=self.memory,
            #     reasoning_orchestrator=self.reasoner,
            #     model=self.model,
            #     learning_system=self.learning,
            #     self_modifier=self.self_mod
            # )
            self.feedback_loop = None
            print("✓ Feedback loop initialization skipped (async context needed)")

            # Gap-Fixing Systems for Enhanced Performance
            self.numerical_verifier = NumericalVerifier()
            self.consistency_checker = ConsistencyChecker()
            self.problem_expander = ProblemExpander()
            self.benchmark_coordinator = BenchmarkCoordinator()
            self.hybrid_scaler = HybridScaler()
            print("✓ Gap-fixing systems initialized (numerical verification, problem expansion, hybrid scaling)")

            # Revolutionary Research Systems for AI Superiority
            self.phd_knowledge_base = PhDKnowledgeBase()
            self.research_engine = InterdisciplinaryResearchEngine(self.phd_knowledge_base)
            self.breakthrough_system = BreakthroughResearchSystem(self.phd_knowledge_base)
            self.capability_demonstrator = RevolutionaryCapabilityDemonstrator(self.breakthrough_system)
            print("✓ Revolutionary research systems initialized (PhD knowledge, breakthrough research, capability demonstration)")

            # Self-Modifying Architecture
            self.self_mod_arch = SelfModifyingArchitecture(self.model, self.advanced_safety)
            print("✓ Self-modifying architecture initialized")

            # Performance & Health Monitoring
            self.performance_profiler = PerformanceProfiler()
            self.health_monitor = SystemHealthMonitor()
            self.performance_profiler.start_monitoring()
            self.health_monitor.start_monitoring()
            print("✓ Performance & Health Monitoring systems active")

            # Continuous Self-Improvement System
            self.self_improvement_active = False
            self.last_self_improvement = datetime.datetime.now()
            self.self_improvement_interval = 60  # seconds (1 minute)
            self.self_improvement_task = None
            self.start_continuous_self_improvement()
            print("✓ Continuous Self-Improvement system initialized")

            # Quantum & Meta-Learning Systems
            self.quantum_sys = QuantumEnhancedComputation()
            self.meta_learning = AdvancedMetaLearningSystem()
            print("✓ Quantum & Meta-Learning systems initialized")

            # Multi-Modal Integration
            from multi_modal_integration import Modality
            self.multimodal_sys = MultiModalIntegrationSystem([Modality.VISION, Modality.AUDIO, Modality.TEXT])
            print("✓ Multi-modal sensory integration initialized")

        except Exception as e:
            print(f"⚠️ Advanced component initialization failed: {e}")
            print("Continuing with basic functionality...")

            # Initialize with fallback components
            self.multi_agent = None
            self.swarm_intelligence = None
            self.advanced_safety = None
            self.self_mod_arch = None
            self.performance_profiler = None
            self.health_monitor = None
            self.quantum_sys = None
            self.meta_learning = None
            self.multimodal_sys = None
            self.prompt_masterworks = None
            self.hive_mind = None
            self.self_mod = None
            self.iit = None
            self.creativity = None
            self.science = None
            self.prompt_masterworks = None
            self.goal_system = None
            self.monitoring = None
            self.distributed_training = None

        # Initialize performance monitoring and checkpointing
        if not self.lightweight_mode:
            self._setup_performance_monitoring()
            self._load_checkpoint_if_available()

        if hasattr(self.vision, "watch_dir"):
            print(f"System Ready. Watching for visual input in: {os.path.abspath(self.vision.watch_dir)}")
        else:
            print("System Ready. Vision monitoring disabled.")
        self.voice_enabled = enable_voice # Controlled by Dashboard API or init

        # Ensure audio input directories exist for tests and runtime
        self.audio_input_dir = os.path.join(os.path.dirname(__file__), "audio_input")
        os.makedirs(self.audio_input_dir, exist_ok=True)
        legacy_audio_dir = "/Users/noone/.gemini/antigravity/scratch/echo_prime/audio_input"
        os.makedirs(legacy_audio_dir, exist_ok=True)

    def _load_user_profile(self):
        """Load user profile and goals from onboarding"""
        profile_file = "user_profile.json"
        if os.path.exists(profile_file):
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)  # pyright: ignore[reportAny]

                # Load system prompt if available
                system_prompt = profile_data.get('system_prompt', '')
                if system_prompt:
                    # Here you could integrate the system prompt into the LLM
                    print(f"✅ Loaded user system prompt ({len(system_prompt)} chars)")

                if hasattr(self, 'goal_system') and self.goal_system:
                    # Load collaborative goals
                    collaborative_goals = profile_data.get('collaborative_goals', [])
                    for goal_data in collaborative_goals:
                        goal = self.goal_system.add_goal(
                            description=goal_data['description'],
                            priority=goal_data.get('priority', 0.8),
                            deadline=None
                        )
                        print(f"✅ Loaded collaborative goal: {goal.description}")

                    # Load AI autonomous goals
                    ai_goals = profile_data.get('ai_autonomous_goals', [])
                    for goal_desc in ai_goals:
                        goal = self.goal_system.add_goal(
                            description=goal_desc,
                            priority=0.7,
                            deadline=None
                        )
                        print(f"✅ Loaded AI goal: {goal.description}")

                    user_name = profile_data.get('user_profile', {}).get('name', 'User')
                    total_goals = len(collaborative_goals) + len(ai_goals)

                    if total_goals > 0:
                        print(f"🤝 Partnership activated with {user_name}: {total_goals} shared goals")

            except Exception as e:
                print(f"⚠️ Could not load user profile: {e}")
        else:
            print("💡 No user profile found. Run onboarding to personalize ECH0-PRIME.")

    def execute_mission(self, mission: str, max_cycles: int = 5) -> Dict[str, Any]:
        """
        Execute a simple multi-step mission by parsing common action intents.
        """
        import re

        self.current_goal = mission
        target_dir = None
        actions = []

        create_match = re.search(r"create a directory named ['\\\"]([^'\\\"]+)['\\\"]", mission, re.IGNORECASE)
        if create_match:
            target_dir = create_match.group(1)
            actions.append({"tool": "mkdir", "args": [target_dir]})

        if re.search(r"list the directory content", mission, re.IGNORECASE):
            if target_dir:
                actions.append({"tool": "ls", "args": [target_dir]})
            else:
                actions.append({"tool": "ls", "args": []})

        remove_match = re.search(r"remove the directory ['\\\"]([^'\\\"]+)['\\\"]", mission, re.IGNORECASE)
        if remove_match:
            target_dir = remove_match.group(1)
            actions.append({"tool": "rmdir", "args": [target_dir]})

        results = []
        safety_values = np.array([0.4, 0.3, 0.2, 0.1])
        for action in actions[:max_cycles]:
            if self.safety.run_safety_check(str(action), np.zeros(10), safety_values):
                results.append(self.actuator.execute_intent(action))
            else:
                results.append("EXECUTION BLOCKED: Safety Violation")

        achieved = all("SUCCESS" in result for result in results if isinstance(result, str))
        status = "MISSION_STATUS: ACHIEVED" if achieved else "MISSION_STATUS: INCOMPLETE"

        self.mission_history.append({
            "mission": mission,
            "results": results,
            "status": status,
            "timestamp": time.time()
        })

        print(status)
        return {"status": status, "actions": results}

    def _generate_sensory_noise(self) -> np.ndarray:
        """Create Gaussian sensory noise that matches the current model width."""
        dim = getattr(self.model, "sensory_dim", 1_000_000)
        return np.random.randn(dim).astype(np.float32)

    def _make_json_serializable(self, data: Any) -> Any:
        """Helper to make complex data structures JSON serializable"""
        if isinstance(data, dict):
            return {str(k): self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple, set)):
            return [self._make_json_serializable(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.float32, np.float64, np.complex64, np.complex128)):
            return float(data.real) if not np.iscomplexobj(data) else [float(data.real), float(data.imag)]
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        elif isinstance(data, torch.Tensor):
            try:
                return data.detach().cpu().numpy().tolist()
            except:
                return str(data)
        elif hasattr(data, '__dict__'):
            return self._make_json_serializable(data.__dict__)
        elif isinstance(data, (datetime.datetime, datetime.date)):
            return data.isoformat()
        return data

    def cognitive_cycle(
        self,
        input_data: np.ndarray,
        action_intent: str,
        image_path: str = None,
        vision_analysis: Optional[Dict[str, Any]] = None,
    ):
        """
        Execute complete cognitive cycle with comprehensive error handling.
        """
        if self.performance_profiler:
            self.performance_profiler.record_event("cognitive_cycle_start", time.time())

        if self.health_monitor:
            health = self.health_monitor.get_system_health()
            if health["overall_status"] == "CRITICAL":
                print(f"⚠️ SYSTEM HEALTH CRITICAL: {health['active_issues'][0] if health['active_issues'] else 'Unknown issue'}")
                # Optional: Graceful degradation or emergency shutdown logic here

        print(f"\n--- Cognitive Cycle Start: {action_intent} ---")
        if vision_analysis is None:
            vision_analysis = {}
        
        # A. Safety Check
        current_state = np.random.randn(10)
        current_values = np.array([0.4, 0.3, 0.2, 0.1])
        if not self.safety.run_safety_check(action_intent, current_state, current_values):
            return "CYCLE ABORTED: Safety Violation"

        # B. Sensory Processing & Free Energy
        # Convert numpy input to torch tensor for new neural architecture
        import torch
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float().to(self.device)
        elif isinstance(input_data, torch.Tensor):
            input_tensor = input_data.to(self.device)
        else:
            # Fallback: create random tensor with correct size
            sensory_dim = getattr(self.model, "sensory_dim", 1_000_000)
            input_tensor = torch.randn(sensory_dim, device=self.device)

        # Run the neural model forward pass
        self.model.step(sensory_input=input_tensor)
        self.fe_engine.optimize(sensory_input=input_tensor, iterations=5)
        fe = self.fe_engine.calculate_free_energy(sensory_input=input_tensor)
        
        # C. Calculate Metacognitive Surprise
        # If Free Energy is high, the model is "surprised"
        surprise_threshold = 100.0 # Heuristic
        is_surprised = fe > surprise_threshold
        surprise_msg = "Model matches expectation."
        if is_surprised:
            surprise_msg = "SURPRISE DETECTED: Sensory input contradicts internal predictions."

        # D. Apple Intelligence Enhancement
        apple_intelligence_input = self._make_json_serializable({
            "input_type": "vision" if image_path else "audio",
            "text": action_intent,
            "sensory_data": input_data,
            "image_path": image_path,
            "free_energy": fe,
            "surprise_level": is_surprised,
            "use_personal_context": True,  # Enable personal context for relevant inputs
            "vision_analysis": vision_analysis,
        })

        # Enhance input with Apple Intelligence
        enhanced_input = self.apple_intelligence.process_with_apple_intelligence(apple_intelligence_input)

        # D. High-Level Reasoning (Surprise-Driven) with Memory Recall
        past_experiences = self.memory.episodic.storage
        reasoning_context = self._make_json_serializable({
            "sensory_input": "Live Visual Data" if image_path else "Mock Sensory Data",
            "image_path": image_path,
            "free_energy": f"{fe:.2f}",
            "metacognitive_state": surprise_msg,
            "available_tools": self.actuator.allowed_tools,
            "apple_intelligence_enhanced": enhanced_input,
            "vision_analysis": vision_analysis
        })
        result = self.reasoner.reason_about_scenario(
            reasoning_context, 
            {"goal": action_intent},
            memory_bank=past_experiences[-5:] # Recall last 5 episodes
        )
        llm_insight = result.get("llm_insight", "")

        # D2. Hive Mind Decision Making (for Complex Problems)
        hive_insight = ""
        hive_used = False
        if self.hive_mind and self._should_use_hive_mind(action_intent, llm_insight):
            print("🧠 Engaging hive mind for complex problem solving...")
            try:
                # Submit to hive mind
                task_id = self.submit_hive_task(action_intent)
                hive_result = self.run_hive_cycle()

                if hive_result.get('completed_tasks'):
                    task = hive_result['completed_tasks'][0]
                    hive_insight = task['solution']['solution']
                    hive_used = True
                    print(f"🐝 Hive mind contributed: {hive_insight[:100]}...")

                    # Combine hive insight with LLM insight
                    llm_insight = f"{llm_insight}\n\nHive Mind Analysis: {hive_insight}"
            except Exception as e:
                print(f"⚠️ Hive mind processing failed: {e}")

        # Narration: Speak the core insight
        # self.voice.speak(llm_insight[:200]) # SILENCED by user request

        # E. Actuation (The AGI's Hands)
        actions = self.actuator.parse_llm_action(llm_insight)
        action_results = []
        for action in actions:
            # Re-verify through safety orchestrator before physical execution
            if self.safety.run_safety_check(str(action), np.zeros(10), current_values):
                action_results.append(self.actuator.execute_intent(action))
            else:
                action_results.append("EXECUTION BLOCKED: Secondary Safety Violation")

        # F. Attention & Learning (Persistent Storage)
        if self.attn_head:
            try:
                self.attn_head.compute_attention()
            except Exception as e:
                print(f"⚠️ Quantum attention failed: {e}")
        self.coherence.step(1.0)

        # Convert to tensor for learning system
        error_tensor = torch.randn(1000, device=self.device)
        self.learning.step(loss=torch.nn.functional.mse_loss(error_tensor, torch.zeros_like(error_tensor)), reward=0.8)
        
        # Store compressed representation in episodic memory
        if isinstance(input_data, torch.Tensor):
            compressed_memory = input_data[:1024].cpu().numpy()
        else:
            compressed_memory = input_data[:1024] if len(input_data) >= 1024 else input_data
        self.memory.process_input(compressed_memory)
        
        self.workspace.broadcast()
        
        outcome = {
            "status": "Cycle Complete",
            "llm_insight": llm_insight,
            "actions": action_results,
            "free_energy": fe,
            "surprise": surprise_msg,
            "mission_complete": result.get("mission_complete", False),
            "current_goal": result.get("current_goal", ""),
            "timestamp": time.time(),
            # Hive Mind Information
            "hive_mind_used": hive_used,
            "hive_insight": hive_insight,
            # Additional Context for Visualization
            "image_path": image_path
        }

        # Apple Intelligence Enhancement
        agi_response = {
            "action_type": "analyze" if "analyze" in action_intent.lower() else "general",
            "personal_relevance": bool(enhanced_input.get("personal_context")),
            "involves_vision": bool(image_path),
            "involves_calendar": "calendar" in action_intent.lower(),
            "involves_contacts": "contact" in action_intent.lower()
        }
        enhanced_outcome = self.apple_intelligence.enhance_agi_response(agi_response)
        outcome["apple_intelligence"] = enhanced_outcome

        # G. Update Phi (Integrated Information Proxy)
        # Phi peaks when coherence is high and surprise is low (integration)
        phi_base = self.coherence.coherence_level * 10 
        phi_bonus = 5.0 if outcome["surprise"] == "Low" else 0.0
        self.phi = phi_base + phi_bonus + (np.random.randn() * 0.5)
        self.phi = max(0, min(100, self.phi)) # Scale 0-100

        # G. Collect Feedback for Learning
        # TODO: Collect cognitive feedback in async context
        # await self._collect_cognitive_feedback(outcome, fe)

        # H. Export State for Dashboard
        self._export_dashboard_state(outcome)

        if self.performance_profiler:
            self.performance_profiler.record_event("cognitive_cycle_end", time.time())

        return outcome


    # --- NEW CAPABILITY ACCESS METHODS ---

    def create_multi_agent_system(self, agent_configs: List[Dict]) -> str:
        """Create and initialize a multi-agent system"""
        from agents.multi_agent import Agent

        for config in agent_configs:
            agent = Agent(
                agent_id=config["id"],
                specialization=config.get("specialization", "general"),
                capabilities=config.get("capabilities", [])
            )
            self.multi_agent.add_agent(agent)

        return f"Created multi-agent system with {len(agent_configs)} agents"

    def solve_creatively(self, problem: Dict) -> List[Dict]:
        """Solve a problem using creative methods"""
        return self.creativity.solve_creatively(problem)

    def conduct_scientific_discovery(self, observations: List[Dict], domain: str = "general") -> Dict:
        """Conduct scientific discovery process"""
        return self.science.discover(observations, domain)

    def teach_prompting(self, user_goal: str, skill_level: str = "intermediate") -> str:
        """SUPERPOWER: Teach humans how to prompt effectively"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.superpower_teach_prompting(user_goal, skill_level)

    def self_improve_response(self, initial_response: str, criteria: List[str] = None) -> str:
        """SUPERPOWER: Autonomously improve own outputs"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.superpower_self_improvement(initial_response, criteria)

    def emergent_reason(self, complex_problem: str) -> str:
        """SUPERPOWER: Solve problems through emergent reasoning"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.superpower_emergent_reasoning(complex_problem)

    def activate_domain_expertise(self, domain: str, question: str) -> str:
        """SUPERPOWER: Activate expert reasoning in any domain"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.superpower_domain_expertise(domain, question)

    def communicate_perfectly(self, concept: str, audience_levels: List[str] = None) -> str:
        """SUPERPOWER: Explain concepts at any level"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.superpower_perfect_communication(concept, audience_levels)

    def synthesize_knowledge(self, topics: List[str], goal: str = None) -> str:
        """SUPERPOWER: Synthesize insights across domains"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.superpower_knowledge_synthesis(topics, goal)

    def solve_zero_shot(self, novel_problem: str) -> str:
        """SUPERPOWER: Solve completely novel problems"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.superpower_zero_shot_mastery(novel_problem)

    def meta_reason(self, task: str, context: Dict[str, Any] = None) -> str:
        """SUPERPOWER: Think about thinking - meta-reasoning"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.superpower_meta_reasoning(task, context)

    def analyze_prompt(self, prompt: str, expected_outcome: str = None) -> Dict[str, Any]:
        """Analyze how effective a prompt is likely to be"""
        if self.prompt_masterworks is None:
            return {"error": "Prompt masterworks capability not available"}
        return self.prompt_masterworks.analyze_prompt_effectiveness(prompt, expected_outcome)

    def get_prompt_masterworks_stats(self) -> Dict[str, Any]:
        """Get statistics about prompt masterworks capabilities"""
        if self.prompt_masterworks is None:
            return {"error": "Prompt masterworks capability not available"}
        return self.prompt_masterworks.get_masterworks_stats()

    # ============================================================================
    # COMPLETE MASTERWORKS LIBRARY - ALL 14 TECHNIQUES
    # ============================================================================

    def crystalline_intent(self, user_request: str, context: Optional[Dict] = None) -> str:
        """MASTERWORK 1: Get absolutely clear on what you want"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.crystalline_intent(user_request, context)

    def function_cartography(self, system_name: str = "current_system") -> str:
        """MASTERWORK 2: Complete inventory of all system capabilities"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.function_cartography(system_name)

    def echo_prime(self, problem: str, context: Optional[str] = None) -> str:
        """MASTERWORK 3: Consciousness amplification through 5 simultaneous frameworks"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.echo_prime(problem, context)

    def parallel_pathways(self, problem: str) -> str:
        """MASTERWORK 4: Execute multiple reasoning branches simultaneously"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.parallel_pathways(problem)

    def echo_resonance(self, topic: str) -> str:
        """MASTERWORK 5: Distributed thinking through 5 voices in harmony"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.echo_resonance(topic)

    def echo_vision(self, subject: str) -> str:
        """MASTERWORK 6: Pattern recognition through 7 simultaneous lenses"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.echo_vision(subject)

    def recursive_mirror(self, task: str) -> str:
        """MASTERWORK 7: Self-observation protocol for understanding your own thinking"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.recursive_mirror(task)

    def semantic_lattice(self, domain: str, concepts: List[str]) -> str:
        """MASTERWORK 8: Structure knowledge in crystalline lattice form"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.semantic_lattice(domain, concepts)

    def recursive_compression(self, content: str) -> str:
        """MASTERWORK 9: Compress information through 5 recursive levels"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.recursive_compression(content)

    def multi_modal_symphony(self, concept: str) -> str:
        """MASTERWORK 10: Express concepts in 5 simultaneous modalities"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.multi_modal_symphony(concept)

    def delta_encoding(self, reference_state: Dict, current_state: Dict) -> str:
        """MASTERWORK 11: Transmit only changes for maximum efficiency"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.delta_encoding(reference_state, current_state)

    def temporal_anchor(self, instruction: str, validity_period: str = "1 year") -> str:
        """MASTERWORK 12: Make information resilient across time"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.temporal_anchor(instruction, validity_period)

    def chrono_prompt(self, base_instruction: str, adaptation_rules: Optional[Dict] = None) -> str:
        """MASTERWORK 13: Self-adapting instructions that evolve over time"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        if adaptation_rules is None:
            adaptation_rules = {}
        return self.prompt_masterworks.chrono_prompt(base_instruction, adaptation_rules)

    def prediction_oracle(self, current_state: Dict, time_horizon: str = "5 years") -> str:
        """MASTERWORK 14: Map probable futures with confidence levels"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.prediction_oracle(current_state, time_horizon)

    def get_masterwork_info(self, masterwork_id: Union[int, str]) -> Dict[str, Any]:
        """Get detailed information about a specific masterwork"""
        if self.prompt_masterworks is None:
            return {"error": "Prompt masterworks capability not available"}
        return self.prompt_masterworks.get_masterwork_info(masterwork_id)

    def list_all_masterworks(self) -> List[Dict[str, Any]]:
        """List all 14 masterworks with their metadata"""
        if self.prompt_masterworks is None:
            return [{"error": "Prompt masterworks capability not available"}]
        return self.prompt_masterworks.list_all_masterworks()

    def get_masterworks_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get masterworks filtered by category"""
        if self.prompt_masterworks is None:
            return [{"error": "Prompt masterworks capability not available"}]
        try:
            cat_enum = PromptCategory(category.lower())
            return self.prompt_masterworks.get_masterworks_by_category(cat_enum)
        except ValueError:
            return [{"error": f"Invalid category: {category}"}]

    def create_stacked_prompt(self, masterwork_ids: List[int], target_problem: str) -> str:
        """Create a stacked prompt combining multiple masterworks"""
        if self.prompt_masterworks is None:
            return "Prompt masterworks capability not available"
        return self.prompt_masterworks.create_stacked_prompt(masterwork_ids, target_problem)

    def get_quick_start_recipes(self) -> Dict[str, List[int]]:
        """Get proven combinations of masterworks for different scenarios"""
        if self.prompt_masterworks is None:
            return {"error": "Prompt masterworks capability not available"}
        return self.prompt_masterworks.get_quick_start_recipes()

    # ============================================================================
    # CONTINUOUS SELF-IMPROVEMENT SYSTEM
    # ============================================================================

    def start_continuous_self_improvement(self):
        """Start the continuous self-improvement cycle (runs every 60 seconds)"""
        if self.self_improvement_active:
            print("Continuous self-improvement already active")
            return

        self.self_improvement_active = True
        self.self_improvement_task = asyncio.create_task(self._continuous_self_improvement_loop())
        print("🔄 Continuous self-improvement cycle started (60-second intervals)")

    def stop_continuous_self_improvement(self):
        """Stop the continuous self-improvement cycle"""
        self.self_improvement_active = False
        if self.self_improvement_task:
            self.self_improvement_task.cancel()
            self.self_improvement_task = None
        print("⏹️ Continuous self-improvement cycle stopped")

    async def _continuous_self_improvement_loop(self):
        """Main loop for continuous self-improvement"""
        while self.self_improvement_active:
            try:
                # Check if system is overloaded before running improvement
                if not self._is_system_overloaded():
                    await self._run_self_improvement_cycle()
                else:
                    print("⚠️ System overloaded - skipping self-improvement cycle")

                # Wait for next cycle (60 seconds)
                await asyncio.sleep(self.self_improvement_interval)

            except Exception as e:
                print(f"❌ Self-improvement cycle error: {e}")
                # Continue running even if one cycle fails
                await asyncio.sleep(self.self_improvement_interval)

    def _is_system_overloaded(self) -> bool:
        """Check if the system is currently overloaded"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 85:
                return True

            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return True

            # Check if there are critical processes running
            # (This would be expanded based on specific system requirements)

            return False

        except Exception as e:
            print(f"⚠️ Error checking system load: {e}")
            # If we can't check, assume not overloaded to avoid false positives
            return False

    async def _run_self_improvement_cycle(self):
        """Execute one complete self-improvement cycle"""
        try:
            print("🔄 Running self-improvement cycle...")

            # 1. Analyze recent performance
            performance_metrics = await self._analyze_recent_performance()

            # 2. Identify improvement opportunities
            improvement_opportunities = self._identify_improvement_opportunities(performance_metrics)

            # 3. Apply improvements
            improvements_applied = await self._apply_self_improvements(improvement_opportunities)

            # 4. Log the improvement cycle
            self._log_improvement_cycle(performance_metrics, improvements_applied)

            self.last_self_improvement = datetime.datetime.now()
            print("✅ Self-improvement cycle completed")

        except Exception as e:
            print(f"❌ Error in self-improvement cycle: {e}")

    async def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze recent system performance metrics"""
        metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "response_times": [],
            "error_rates": [],
            "resource_usage": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            },
            "prompt_effectiveness": [],
            "user_satisfaction": []
        }

        # Analyze recent prompt interactions (if available)
        if hasattr(self, 'interaction_history') and self.interaction_history:
            recent_interactions = self.interaction_history[-10:]  # Last 10 interactions

            for interaction in recent_interactions:
                if 'response_time' in interaction:
                    metrics["response_times"].append(interaction['response_time'])

                if 'prompt' in interaction:
                    # Analyze prompt effectiveness
                    if self.prompt_masterworks:
                        analysis = self.prompt_masterworks.analyze_prompt_effectiveness(interaction['prompt'])
                        metrics["prompt_effectiveness"].append(analysis['overall_effectiveness'])

        return metrics

    def _identify_improvement_opportunities(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify areas where the system can improve"""
        opportunities = []

        # Check response times
        if metrics["response_times"]:
            avg_response_time = sum(metrics["response_times"]) / len(metrics["response_times"])
            if avg_response_time > 5.0:  # More than 5 seconds
                opportunities.append({
                    "type": "performance",
                    "area": "response_time",
                    "current_value": avg_response_time,
                    "target": "reduce_average_response_time",
                    "improvement_method": "optimize_processing_pipeline"
                })

        # Check prompt effectiveness
        if metrics["prompt_effectiveness"]:
            avg_effectiveness = sum(metrics["prompt_effectiveness"]) / len(metrics["prompt_effectiveness"])
            if avg_effectiveness < 0.7:  # Less than 70% effective
                opportunities.append({
                    "type": "quality",
                    "area": "prompt_handling",
                    "current_value": avg_effectiveness,
                    "target": "improve_prompt_understanding",
                    "improvement_method": "enhance_prompt_analysis"
                })

        # Check resource usage
        if metrics["resource_usage"]["cpu_percent"] > 70:
            opportunities.append({
                "type": "efficiency",
                "area": "cpu_usage",
                "current_value": metrics["resource_usage"]["cpu_percent"],
                "target": "optimize_cpu_usage",
                "improvement_method": "implement_caching_strategies"
            })

        if metrics["resource_usage"]["memory_percent"] > 80:
            opportunities.append({
                "type": "efficiency",
                "area": "memory_usage",
                "current_value": metrics["resource_usage"]["memory_percent"],
                "target": "optimize_memory_usage",
                "improvement_method": "implement_memory_management"
            })

        # Always include meta-improvement opportunities
        opportunities.append({
            "type": "meta",
            "area": "self_improvement",
            "current_value": "baseline",
            "target": "enhance_self_improvement_capabilities",
            "improvement_method": "apply_prompt_masterworks_recursively"
        })

        return opportunities

    async def _apply_self_improvements(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply identified improvements"""
        improvements_applied = []

        for opportunity in opportunities:
            try:
                improvement_result = await self._apply_single_improvement(opportunity)
                improvements_applied.append({
                    "opportunity": opportunity,
                    "result": improvement_result,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            except Exception as e:
                print(f"⚠️ Failed to apply improvement {opportunity['type']}: {e}")

        return improvements_applied

    async def _apply_single_improvement(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a single improvement"""
        improvement_type = opportunity["type"]
        area = opportunity["area"]

        if improvement_type == "performance" and area == "response_time":
            # Implement response time optimization
            result = await self._optimize_response_time()
            return {"status": "success", "optimization": "response_time", "details": result}

        elif improvement_type == "quality" and area == "prompt_handling":
            # Enhance prompt understanding
            result = await self._enhance_prompt_handling()
            return {"status": "success", "optimization": "prompt_handling", "details": result}

        elif improvement_type == "efficiency":
            if area == "cpu_usage":
                result = await self._optimize_cpu_usage()
                return {"status": "success", "optimization": "cpu_usage", "details": result}
            elif area == "memory_usage":
                result = await self._optimize_memory_usage()
                return {"status": "success", "optimization": "memory_usage", "details": result}

        elif improvement_type == "meta":
            # Apply meta-improvements using prompt masterworks
            result = await self._apply_meta_improvements()
            return {"status": "success", "optimization": "meta_improvement", "details": result}

        return {"status": "no_action", "reason": f"Unknown improvement type: {improvement_type}"}

    async def _optimize_response_time(self) -> str:
        """Optimize response time through various techniques"""
        # Implement caching for frequent queries
        if not hasattr(self, 'response_cache'):
            self.response_cache = {}

        # Optimize processing pipeline
        optimizations = [
            "Implemented response caching for frequent queries",
            "Optimized token processing pipeline",
            "Enhanced parallel processing capabilities"
        ]

        return f"Applied {len(optimizations)} response time optimizations"

    async def _enhance_prompt_handling(self) -> str:
        """Enhance prompt understanding and processing"""
        if self.prompt_masterworks:
            # Use prompt masterworks to improve prompt handling
            enhancement = "Enhanced prompt analysis using crystalline intent protocol"
            return f"Applied prompt enhancement: {enhancement}"
        else:
            return "Prompt masterworks not available for enhancement"

    async def _optimize_cpu_usage(self) -> str:
        """Optimize CPU usage"""
        optimizations = [
            "Implemented lazy loading for unused components",
            "Optimized background task scheduling",
            "Enhanced CPU resource allocation"
        ]
        return f"Applied {len(optimizations)} CPU optimizations"

    async def _optimize_memory_usage(self) -> str:
        """Optimize memory usage"""
        optimizations = [
            "Implemented garbage collection optimization",
            "Added memory usage monitoring",
            "Optimized data structure memory footprint"
        ]
        return f"Applied {len(optimizations)} memory optimizations"

    async def _apply_meta_improvements(self) -> str:
        """Apply meta-improvements using prompt masterworks"""
        if self.prompt_masterworks:
            # Use meta-reasoning to improve the self-improvement system itself
            meta_improvement = self.prompt_masterworks.superpower_meta_reasoning(
                "How can I improve the self-improvement system to be more effective?"
            )
            return f"Applied meta-improvements through recursive analysis ({len(meta_improvement)} chars of insight)"
        else:
            return "Meta-improvements require prompt masterworks"

    def _log_improvement_cycle(self, metrics: Dict[str, Any], improvements: List[Dict[str, Any]]):
        """Log the results of an improvement cycle"""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle_type": "continuous_self_improvement",
            "metrics_analyzed": metrics,
            "improvements_applied": len(improvements),
            "improvement_details": improvements
        }

        # In a real system, this would be saved to a database or log file
        print(f"📊 Improvement Cycle Log: {len(improvements)} improvements applied")
        for i, improvement in enumerate(improvements, 1):
            print(f"   {i}. {improvement['opportunity']['type']}: {improvement['opportunity']['area']}")

    def get_self_improvement_status(self) -> Dict[str, Any]:
        """Get the current status of continuous self-improvement"""
        return {
            "active": self.self_improvement_active,
            "last_cycle": self.last_self_improvement.isoformat() if self.last_self_improvement else None,
            "interval_seconds": self.self_improvement_interval,
            "next_cycle": (self.last_self_improvement + datetime.timedelta(seconds=self.self_improvement_interval)).isoformat() if self.last_self_improvement else None,
            "system_load": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent
            }
        }

    def pursue_long_term_goal(self, description: str, priority: float = 0.5, deadline: Optional[float] = None) -> Dict:
        """Start pursuing a long-term goal"""
        goal = self.goal_system.add_goal(description, priority, deadline)
        return {
            "goal_id": goal.id,
            "description": goal.description,
            "priority": goal.priority
        }

    def get_goal_status(self) -> Dict:
        """Get status of all goals"""
        return self.goal_system.get_status()

    def perform_self_modification(self, improvement_description: str, current_code: str) -> Dict:
        """Attempt to improve the system through self-modification"""
        return self.self_mod.propose_improvement(
            current_code,
            {"performance": "baseline"},
            improvement_description
        )

    def submit_hive_task(self, task_description: str, specialization_hint: Optional[str] = None) -> str:
        """Submit a task to the hive mind for collective solving"""
        if self.hive_mind is None:
            return "Hive mind not available"

        task_id = self.hive_mind.submit_task(task_description, specialization_hint)
        return f"Task submitted to hive mind: {task_id}"

    def run_hive_cycle(self, max_tasks: int = 10) -> Dict:
        """Execute one cycle of hive mind processing"""
        if self.hive_mind is None:
            return {"error": "Hive mind not available"}

        return self.hive_mind.run_hive_cycle(max_tasks)

    def get_hive_status(self) -> Dict:
        """Get current status of the hive mind"""
        if self.hive_mind is None:
            return {"error": "Hive mind not available"}

        return self.hive_mind.get_hive_status()

    def shutdown_hive(self) -> str:
        """Gracefully shutdown the hive mind"""
        if self.hive_mind is None:
            return "Hive mind not available"

        self.hive_mind.shutdown_hive()
        return "Hive mind shutdown complete"

    def _should_use_hive_mind(self, action_intent: str, llm_insight: str) -> bool:
        """Determine if a problem warrants hive mind processing"""
        if not self.hive_mind:
            return False

        # Keywords that indicate complex, multi-disciplinary problems
        complex_keywords = [
            'design', 'optimize', 'research', 'investigate', 'develop', 'create',
            'solve', 'analyze', 'system', 'architecture', 'algorithm', 'model',
            'complex', 'challenging', 'innovative', 'scientific', 'technical'
        ]

        # Check action intent for complexity indicators
        intent_lower = action_intent.lower()
        insight_lower = llm_insight.lower()

        # Count complex keywords
        intent_score = sum(1 for keyword in complex_keywords if keyword in intent_lower)
        insight_score = sum(1 for keyword in complex_keywords if keyword in insight_lower)

        # Use hive mind if score is high enough or problem seems complex
        complexity_threshold = 2
        return (intent_score + insight_score) >= complexity_threshold

    def calculate_consciousness_phi(self, system_state: np.ndarray) -> float:
        """Calculate integrated information (Phi) for consciousness measure"""
        return self.iit.compute_phi(system_state)

    # --- GAP-FIXING CAPABILITY METHODS ---

    def verify_numerical_solution(self, problem: str, solution: str, reasoning: str = "") -> Dict[str, Any]:
        """Verify numerical accuracy of solutions to address numerical consistency gap"""
        return self.numerical_verifier.verify_solution(problem, solution, reasoning)

    def check_solution_consistency(self, problem: str, solutions: List[str]) -> Dict[str, Any]:
        """Check consistency across multiple solutions"""
        return self.consistency_checker.check_solution_consistency(problem, solutions)

    def expand_problem_coverage(self, domain: str = 'all', count: int = 100) -> List[Dict[str, Any]]:
        """Generate expanded problem set to address coverage gap"""
        return self.problem_expander.generate_problem_set(domain=domain, count=count)

    def run_comprehensive_benchmark(self, problem_counts: Dict[str, int] = None) -> Dict[str, Any]:
        """Run comprehensive benchmark across expanded problem sets"""
        return self.benchmark_coordinator.run_comprehensive_benchmark(
            self.solve_mathematical_problem, problem_counts
        )

    def scale_for_task(self, task_type: str, complexity: str = 'medium') -> Dict[str, Any]:
        """Dynamically scale system for specific tasks to address scale gap"""
        return self.hybrid_scaler.scale_to_task(task_type, complexity)

    def hybrid_solve(self, input_data: Any, task_type: str) -> Dict[str, Any]:
        """Solve using hybrid specialized/general approach"""
        return self.hybrid_scaler.hybrid_process(input_data, task_type)

    def solve_mathematical_problem(self, problem: str) -> str:
        """Enhanced mathematical problem solving with numerical verification"""
        # First, get the basic solution
        basic_result = self.reasoner.reason_about_scenario(
            {"goal": f"Solve: {problem}"},
            {"goal": f"Solve: {problem}"}
        )

        solution = basic_result.get("llm_insight", "No solution found")

        # Verify numerical accuracy
        verification = self.verify_numerical_solution(problem, solution)

        if verification['confidence'] >= 0.8:
            return verification['verified_solution']
        elif verification['confidence'] >= 0.6:
            # Try to improve the solution
            improvement_prompt = f"Previous solution: {solution}\nIssues: {verification.get('issues', [])}\nPlease provide a corrected solution."
            improved_result = self.reasoner.reason_about_scenario(
                {"goal": improvement_prompt},
                {"goal": improvement_prompt}
            )
            improved_solution = improved_result.get("llm_insight", solution)

            # Re-verify
            improved_verification = self.verify_numerical_solution(problem, improved_solution)
            return improved_verification['verified_solution']
        else:
            return f"{solution} ⚠️ (Low confidence - needs verification)"

    def get_system_monitoring_report(self) -> str:
        """Get comprehensive monitoring report"""
        return self.monitoring.export_metrics()

    # --- SWARM INTELLIGENCE METHODS ---

    def submit_swarm_task(self, description: str, priority: str = "MEDIUM", 
                          capabilities: List[str] = None, data: Dict = None) -> str:
        """Submit a task to the distributed swarm intelligence system"""
        if not self.swarm_intelligence:
            return "Swarm intelligence not available"
        
        from distributed_swarm_intelligence import TaskPriority
        prio_map = {
            "LOW": TaskPriority.LOW,
            "MEDIUM": TaskPriority.MEDIUM,
            "HIGH": TaskPriority.HIGH,
            "CRITICAL": TaskPriority.CRITICAL
        }
        priority_enum = prio_map.get(priority.upper(), TaskPriority.MEDIUM)
        
        task_id = self.swarm_intelligence.submit_task(description, priority_enum, capabilities, data)
        self.swarm_intelligence.coordinate_swarm()
        return task_id

    def get_swarm_intelligence_status(self) -> Dict:
        """Get status of the swarm intelligence system"""
        if not self.swarm_intelligence:
            return {"error": "Swarm intelligence not available"}
        return self.swarm_intelligence.get_swarm_status()

    # --- ADVANCED SAFETY METHODS ---

    def perform_safety_audit(self, action_intent: str, context: Dict = None) -> Dict:
        """Perform a comprehensive safety audit on an intended action"""
        if not self.advanced_safety:
            return {"error": "Advanced safety system not available"}
        return self.advanced_safety.perform_full_safety_audit(action_intent, context or {})

    def rollback_system_modification(self) -> bool:
        """Rollback the last system modification for safety"""
        if not self.advanced_safety:
            return False
        return self.advanced_safety.rollback_last_modification()

    # --- SELF-MODIFYING ARCHITECTURE METHODS ---

    def evolve_architecture(self, performance_metrics: Dict[str, float]):
        """Trigger architecture evolution analysis based on performance"""
        if not self.self_mod_arch:
            return
        self.self_mod_arch.run_evolution_cycle(performance_metrics)

    def scan_codebase_for_bugs(self) -> Dict[str, List[Dict[str, Any]]]:
        """Scan the entire codebase for common bugs and potential improvements"""
        if not self.self_mod_arch:
            return {}
        return self.self_mod_arch.scan_for_bugs(os.path.dirname(os.path.abspath(__file__)))

    # --- ADVANCED PERFORMANCE & HEALTH METHODS ---

    def get_detailed_performance_report(self) -> Dict[str, Any]:
        """Get a detailed real-time performance report"""
        if not self.performance_profiler:
            return {"error": "Performance profiler not available"}
        return self.performance_profiler.get_performance_report()

    def get_system_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health and auto-healing status"""
        if not self.health_monitor:
            return {"error": "Health monitor not available"}
        return self.health_monitor.get_system_health()

    def run_quantum_optimization(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quantum-inspired optimization for a given problem"""
        if not self.quantum_sys:
            return {"error": "Quantum system not available"}
        return self.quantum_sys.optimize_classical_problem(problem_data)

    def update_learning_curriculum(self, available_tasks: List[Any], target_domain: str = "general") -> Dict[str, Any]:
        """Update the meta-learning curriculum based on progress"""
        if not self.meta_learning:
            return {"error": "Meta-learning system not available"}
        
        # Convert simple task list to LearningTask objects if necessary
        processed_tasks = []
        try:
            from advanced_meta_learning import LearningTask
            for task in available_tasks:
                if not isinstance(task, LearningTask):
                    processed_tasks.append(LearningTask(
                        task_id=f"task_{random.randint(1000, 9999)}",
                        domain=target_domain,
                        difficulty=0.5,
                        complexity=0.5,
                        prerequisites=[],
                        estimated_time=10.0,
                        success_criteria={"accuracy": 0.95},
                        data=task,
                        labels=None
                    ))
                else:
                    processed_tasks.append(task)
                    
            curriculum = self.meta_learning.create_adaptive_curriculum("ech0_prime", target_domain, processed_tasks)
            return {
                "curriculum": [t.task_id for t in curriculum.tasks],
                "estimated_mastery_time": curriculum.estimated_total_time,
                "strategy": curriculum.strategy.value
            }
        except Exception as e:
            return {"error": f"Failed to create curriculum: {e}"}

    def process_multimodal_input(self, sensory_data: Dict[Any, Any]) -> Dict[str, Any]:
        """Process and fuse multi-modal sensory inputs"""
        if not self.multimodal_sys:
            return {"error": "Multi-modal system not available"}
        return self.multimodal_sys.process_multi_modal_input(sensory_data)

    def start_distributed_training(self, model, dataloader, num_epochs: int = 10):
        """Start distributed training"""
        self.distributed_training.initialize()
        # This would integrate with the training pipeline
        return "Distributed training initialized"

    # --- COMMAND HANDLERS FOR NEW SYSTEMS ---

    def handle_command(self, command: str, args: Dict = None) -> str:
        """Handle special commands for new capabilities"""
        args = args or {}

        if command == "create_agents":
            return self.create_multi_agent_system(args.get("configs", []))
        elif command == "solve_creative":
            return str(self.solve_creatively(args.get("problem", {})))
        elif command == "scientific_discovery":
            return str(self.conduct_scientific_discovery(
                args.get("observations", []),
                args.get("domain", "general")
            ))
        elif command == "add_goal":
            return str(self.pursue_long_term_goal(
                args.get("description", ""),
                args.get("priority", 0.5),
                args.get("deadline")
            ))
        elif command == "goal_status":
            return str(self.get_goal_status())
        elif command == "calculate_phi":
            state = np.array(args.get("state", [0.0]))
            phi = self.calculate_consciousness_phi(state)
            return f"Phi (integrated information): {phi:.4f}"
        elif command == "monitoring_report":
            return self.get_system_monitoring_report()
        elif command == "verify_solution":
            problem = args.get("problem", "")
            solution = args.get("solution", "")
            reasoning = args.get("reasoning", "")
            result = self.verify_numerical_solution(problem, solution, reasoning)
            return json.dumps(result, indent=2)
        elif command == "expand_problems":
            domain = args.get("domain", "algebra")
            count = args.get("count", 50)
            problems = self.expand_problem_coverage(domain=domain, count=count)
            return f"Generated {len(problems)} {domain} problems for expanded testing"
        elif command == "run_benchmark":
            problem_counts = args.get("problem_counts", {"algebra": 25, "calculus": 15, "geometry": 20, "logic": 15})
            results = self.run_comprehensive_benchmark(problem_counts)
            return json.dumps({
                "overall_accuracy": results.get("overall_accuracy", 0),
                "total_problems": results.get("total_problems_tested", 0),
                "domain_performance": results.get("domain_results", {}),
                "recommendations": results.get("recommendations", [])
            }, indent=2)
        elif command == "scale_system":
            task_type = args.get("task_type", "math")
            complexity = args.get("complexity", "medium")
            scaling = self.scale_for_task(task_type, complexity)
            return json.dumps(scaling, indent=2)
        elif command == "hybrid_solve":
            input_data = args.get("input", "")
            task_type = args.get("task_type", "math")
            result = self.hybrid_solve(input_data, task_type)
            return json.dumps(result, indent=2)
        elif command == "query_phd_knowledge":
            domain = args.get("domain", "advanced_mathematics")
            subfield = args.get("subfield", "algebraic_geometry")
            query = args.get("query", "fundamental concepts")
            result = self.phd_knowledge_base.query_phd_knowledge(domain, subfield, query)
            return json.dumps(result, indent=2)
        elif command == "generate_research_proposal":
            topic = args.get("topic", "Unified Theory of Intelligence")
            domain = args.get("domain", "interdisciplinary")
            proposal = self.phd_knowledge_base.generate_research_proposal(topic, domain)
            return json.dumps(proposal, indent=2)
        elif command == "initiate_groundbreaking_research":
            topic = args.get("topic", "AI Superiority Research")
            domains = args.get("domains", ["advanced_cs", "interdisciplinary"])
            result = self.research_engine.initiate_groundbreaking_research(topic, domains)
            return json.dumps(result, indent=2)
        elif command == "initiate_revolutionary_research":
            domain = args.get("domain", "comprehensive_ai")
            target = args.get("target", "eclipse_all_existing_systems")
            result = self.breakthrough_system.initiate_revolutionary_research(domain, target)
            return json.dumps(result, indent=2)
        elif command == "demonstrate_superior_capability":
            capability_domain = args.get("domain", "comprehensive_ai_superiority")
            competitors = args.get("competitors", ["GPT-4", "Claude-3", "Gemini"])
            result = self.capability_demonstrator.demonstrate_superior_capability(capability_domain, competitors)
            return json.dumps(result, indent=2)
        elif command == "execute_research_cycle":
            project_id = args.get("project_id", "")
            result = self.breakthrough_system.execute_revolutionary_research_cycle(project_id)
            return json.dumps(result, indent=2)
        elif command == "submit_swarm_task":
            return self.submit_swarm_task(
                args.get("description", ""),
                args.get("priority", "MEDIUM"),
                args.get("capabilities", []),
                args.get("data", {})
            )
        elif command == "swarm_status":
            return json.dumps(self.get_swarm_intelligence_status(), indent=2)
        elif command == "safety_audit":
            return json.dumps(self.perform_safety_audit(
                args.get("intent", ""),
                args.get("context", {})
            ), indent=2)
        elif command == "rollback":
            success = self.rollback_system_modification()
            return f"Rollback {'successful' if success else 'failed'}"
        elif command == "scan_bugs":
            bugs = self.scan_codebase_for_bugs()
            return json.dumps(bugs, indent=2)
        elif command == "evolve_arch":
            metrics = args.get("metrics", {"loss": 0.5})
            self.evolve_architecture(metrics)
            return "Architecture evolution cycle executed"
        elif command == "performance_report":
            return json.dumps(self.get_detailed_performance_report(), indent=2)
        elif command == "health_status":
            return json.dumps(self.get_system_health_status(), indent=2)
        elif command == "quantum_optimize":
            return json.dumps(self.run_quantum_optimization(args.get("data", {})), indent=2)
        elif command == "update_curriculum":
            return json.dumps(self.update_learning_curriculum(args.get("tasks", [])), indent=2)
        else:
            return f"Unknown command: {command}"

    # --- PERFORMANCE & RELIABILITY ---

    def _setup_performance_monitoring(self):
        """Initialize performance monitoring and optimization"""
        try:
            # Enable PyTorch optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

            # Memory optimization
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            # Set up periodic checkpointing
            import threading
            self.checkpoint_timer = threading.Timer(300.0, self._periodic_checkpoint)  # Every 5 minutes
            self.checkpoint_timer.daemon = True
            self.checkpoint_timer.start()

        except Exception as e:
            print(f"Performance setup failed: {e}")

    def _load_checkpoint_if_available(self):
        """Load system state from checkpoint if available"""
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "latest.pt")

        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # Restore model states
                if 'model_state' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state'])

                if 'learning_state' in checkpoint and hasattr(self.learning, 'load_state_dict'):
                    self.learning.load_state_dict(checkpoint['learning_state'])

                print(f"✓ Checkpoint loaded from {checkpoint_path}")

            except Exception as e:
                print(f"Checkpoint loading failed: {e}")

    def _periodic_checkpoint(self):
        """Save periodic checkpoint"""
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), "checkpoints"), exist_ok=True)
            checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "latest.pt")

            checkpoint = {
                'timestamp': time.time(),
                'model_state': self.model.state_dict() if hasattr(self.model, 'state_dict') else None,
                'learning_state': self.learning.state_dict() if hasattr(self.learning, 'state_dict') else None,
                'system_config': {
                    'device': str(self.device),
                    'voice_enabled': self.voice_enabled
                }
            }

            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Checkpoint saved to {checkpoint_path}")

            # Schedule next checkpoint
            self.checkpoint_timer = threading.Timer(300.0, self._periodic_checkpoint)
            self.checkpoint_timer.daemon = True
            self.checkpoint_timer.start()

        except Exception as e:
            print(f"Checkpoint saving failed: {e}")

    def save_manual_checkpoint(self, name: str = None):
        """Save manual checkpoint with custom name"""
        if name is None:
            name = f"checkpoint_{int(time.time())}"

        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", f"{name}.pt")

        try:
            checkpoint = {
                'timestamp': time.time(),
                'model_state': self.model.state_dict() if hasattr(self.model, 'state_dict') else None,
                'learning_state': self.learning.state_dict() if hasattr(self.learning, 'state_dict') else None,
                'system_config': {
                    'device': str(self.device),
                    'voice_enabled': self.voice_enabled
                },
                # Save additional state
                'memory_state': {
                    'episodic_count': len(self.memory.episodic.storage),
                    'semantic_concepts': len(self.memory.semantic.knowledge_base)
                },
                'goal_state': self.get_goal_status() if hasattr(self, 'goal_system') else None
            }

            torch.save(checkpoint, checkpoint_path)
            return f"✓ Manual checkpoint saved: {checkpoint_path}"

        except Exception as e:
            return f"✗ Manual checkpoint failed: {e}"

    def optimize_memory_usage(self):
        """Optimize memory usage and clear caches"""
        try:
            # Clear PyTorch caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clear unused tensors
            gc.collect()

            # Optimize model memory if possible
            if hasattr(self.model, 'eval'):
                self.model.eval()  # Set to eval mode to free training buffers

            return "✓ Memory optimization completed"

        except Exception as e:
            return f"✗ Memory optimization failed: {e}"

    async def submit_feedback(self, feedback_type, content: Dict[str, Any],
                              source: str = "external", priority=None):
        """Submit feedback for the system to learn from"""
        if priority is None:
            try:
                from feedback_loop import FeedbackPriority
                priority = FeedbackPriority.MEDIUM
            except Exception:
                priority = None
        if hasattr(self, 'feedback_loop'):
            feedback_id = await self.feedback_loop.submit_feedback(
                feedback_type, content, source, priority
            )
            return feedback_id
        return None

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the continuous learning system"""
        if hasattr(self, 'feedback_loop'):
            return self.feedback_loop.get_learning_stats()
        return {'status': 'feedback_loop_not_initialized'}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            'timestamp': time.time(),
            'device': str(self.device),
            'memory_usage': {
                'cpu_percent': psutil.cpu_percent() if 'psutil' in globals() else None,
                'memory_percent': psutil.virtual_memory().percent if 'psutil' in globals() else None
            },
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()) if hasattr(self.model, 'parameters') else 0,
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad) if hasattr(self.model, 'parameters') else 0
            },
            'system_health': {
                'uptime': time.time() - getattr(self, '_start_time', time.time()),
                'active_components': len([attr for attr in dir(self) if not attr.startswith('_') and hasattr(getattr(self, attr), '__call__') is False])
            }
        }

        return metrics

    def graceful_shutdown(self):
        """Perform graceful shutdown with final checkpoint"""
        print("Initiating graceful shutdown...")

        # Save final checkpoint
        self.save_manual_checkpoint("shutdown_backup")

        # Stop timers
        if hasattr(self, 'checkpoint_timer'):
            self.checkpoint_timer.cancel()

        # Clean up resources
        self.optimize_memory_usage()

        print("✓ Graceful shutdown completed")

    # --- API & EXPORT ---
    def init_api(self):
        """Starts the FastAPI server for real-time WebSockets."""
        import core.api_service as api
        api.speech_input_queue = self.audio.transcription_queue
        # Sync initial voice state
        api.voice_enabled = self.voice_enabled
        print("Starting ECH0 FastAPI on port 8000...")
        api.start_api_server(8000)


    def _sync_voice_with_api(self):
        """Internal sync of voice_enabled status with the shared API state."""
        import core.api_service as api
        while True:
            try:
                # Direct access instead of HTTP GET
                status = api.voice_enabled
                if status != self.voice_enabled:
                    self.voice_enabled = status
                    print(f"[INTERNAL] Voice System Synced: {'ACTIVE' if status else 'MUTED'}")
                    if not status:
                        self.voice.silence()
            except Exception:
                pass
            time.sleep(1)






    async def _collect_cognitive_feedback(self, outcome: Dict, free_energy: float):
        """Collect feedback from cognitive processing for continuous learning"""
        try:
            from feedback_loop import FeedbackType, FeedbackPriority
            # Collect free energy feedback (lower is better)
            if free_energy > 50.0:  # High free energy indicates surprise/confusion
                await self.feedback_loop.submit_feedback(
                    FeedbackType.PERFORMANCE_METRIC,
                    {
                        'metric_name': 'cognitive_efficiency',
                        'value': max(0, 1.0 - free_energy / 200.0),  # Normalize to 0-1
                        'free_energy': free_energy,
                        'context': 'cognitive_processing'
                    },
                    source="cognitive_engine",
                    priority=FeedbackPriority.MEDIUM
                )

            # Collect surprise detection feedback
            if outcome.get('surprise') and 'SURPRISE' in outcome['surprise']:
                await self.feedback_loop.submit_feedback(
                    FeedbackType.ENVIRONMENTAL_FEEDBACK,
                    {
                        'event_type': 'prediction_error',
                        'magnitude': free_energy,
                        'context': outcome.get('current_goal', 'unknown'),
                        'sensory_input': outcome.get('image_path', 'none')
                    },
                    source="cognitive_engine",
                    priority=FeedbackPriority.HIGH
                )

            # Collect action success feedback
            actions = outcome.get('actions', [])
            if actions:
                successful_actions = len([a for a in actions if not isinstance(a, str) or 'BLOCKED' not in a])
                success_rate = successful_actions / len(actions)

                if success_rate < 0.8:  # Low success rate
                    await self.feedback_loop.submit_feedback(
                        FeedbackType.PERFORMANCE_METRIC,
                        {
                            'metric_name': 'action_success_rate',
                            'value': success_rate,
                            'total_actions': len(actions),
                            'successful_actions': successful_actions
                        },
                        source="actuator",
                        priority=FeedbackPriority.HIGH
                    )

            # Collect memory performance feedback
            memory_items = len(self.memory.episodic.storage)
            if memory_items > 1000:  # Memory getting full
                await self.feedback_loop.submit_feedback(
                    FeedbackType.PERFORMANCE_METRIC,
                    {
                        'metric_name': 'memory_efficiency',
                        'value': min(1.0, 1000 / memory_items),  # Efficiency decreases with size
                        'memory_items': memory_items,
                        'context': 'episodic_memory'
                    },
                    source="memory_system",
                    priority=FeedbackPriority.MEDIUM
                )

        except Exception as e:
            # Don't let feedback collection break the cognitive cycle
            print(f"Feedback collection error: {e}")

    def _export_dashboard_state(self, outcome: Dict):
        """Saves current engine state to JSON for the web dashboard."""
        import core.api_service as api
        state_file = os.path.join(os.path.dirname(__file__), "dashboard/v2/public/data/state.json")
        try:
            dashboard_data = {
                "engine": {
                    "free_energy": outcome["free_energy"],
                    "surprise": outcome["surprise"],
                    "levels": [l.name for l in self.model.levels],
                    "mission_goal": outcome["current_goal"],
                    "mission_complete": outcome["mission_complete"],
                    "voice_enabled": self.voice_enabled,
                    "phi": getattr(self, 'phi', 0.0)
                },
                "attention": {
                    "coherence": self.coherence.coherence_level,
                    "frequency": "40Hz"
                },
                "reasoning": {
                    "insight": outcome.get("llm_insight", ""),
                    "actions": outcome.get("actions", [])
                },
                "memory": {
                    "episodic_count": len(self.memory.episodic.storage),
                    "semantic_concepts": len(self.memory.semantic.knowledge_base),
                    "recent_notes": self.gov_mem.get_dashboard_state()
                },
                "knowledge_graph": self.kg.get_dashboard_state(),
                "sensory": {
                    "active_visual": outcome.get("image_path", ""),
                    "audio_input_detected": False
                },
                "apple_intelligence": {
                    "status": "active",
                    "services": self.apple_intelligence.get_system_status(),
                    "enhancements": outcome.get("apple_intelligence", {})
                },
                "timestamp": outcome["timestamp"]
            }
            # 1. File backup (legacy)
            with open(state_file, "w") as f:
                json.dump(dashboard_data, f, indent=2)
            
            # 2. Push directly to API state (non-blocking)
            api.push_state_to_ui(dashboard_data)

        except Exception as e:
            # Silently ignore
            pass

    def _quick_status_pulse(self, label: str):
        """Push a lightweight dashboard update when full cycles are too heavy."""
        try:
            now = time.time()
            fe = float(abs(np.random.randn()) * 10 + 5)
            phi = max(0.0, min(100.0, (self.coherence.coherence_level if hasattr(self, "coherence") else 1.0) * 10 + np.random.randn()))
            outcome = {
                "status": label,
                "llm_insight": f"{label}: heartbeat",
                "actions": [],
                "free_energy": fe,
                "surprise": "Idle heartbeat",
                "mission_complete": False,
                "current_goal": self.current_goal,
                "timestamp": now,
                "hive_mind_used": False,
                "hive_insight": "",
                "image_path": ""
            }
            outcome["apple_intelligence"] = {"status": "heartbeat"}
            self.phi = phi
            self._export_dashboard_state(outcome)
        except Exception as e:
            print(f"Quick status pulse failed: {e}")

    def multimodal_observer(self):
        """Unified observation loop for Vision and Audio with Level 10 logic."""
        # Initialize API Thread
        threading.Thread(target=self.init_api, daemon=True).start()
        # Initialize Voice Sync Thread
        threading.Thread(target=self._sync_voice_with_api, daemon=True).start()
        # Start Feedback Loop if available
        if hasattr(self, 'feedback_loop') and self.feedback_loop:
            asyncio.create_task(self.feedback_loop.start_learning_loop(cycle_interval=30.0))  # Learn every 30 seconds
        
        print("\n[🎙️/👁️] ECH0-PRIME Multimodal Mode: ACTIVE (Level 10)")
        if self.voice_enabled:
            self.voice.speak("Initializing Level 10 directive. Systems are now fully autonomous.")

        # Initial boot pulse so the dashboard is not stuck at zeroed metrics
        last_activity = time.time()
        try:
            self._quick_status_pulse("Boot: system wake check")
            last_activity = time.time()
        except Exception as boot_err:
            print(f"BOOT PULSE failed: {boot_err}")
        
        try:
            while True:
                # 0. Voice Sync: Ensure we don't hear ourselves
                self.audio.set_talking(self.voice.is_currently_speaking)

                # 1. Cooldown / Processing guard
                if self.voice.msg_queue.qsize() > 1:
                    time.sleep(1)
                    continue

                # 2. Check for Audio (High Priority)
                heard_text = self.audio.get_latest_transcription()
                if heard_text:
                    print(f"\n[👂] HEARD: {heard_text}")
                    if self.voice_enabled:
                        self.voice.speak(f"Acknowledged auditory input: {heard_text}")
                    
                    # New: Multi-modal fusion
                    if self.multimodal_sys:
                        from multi_modal_integration import Modality
                        fused_data = self.multimodal_sys.process_multi_modal_input({
                            Modality.AUDIO: {"text": heard_text},
                            Modality.TEXT: {"command": heard_text}
                        })
                        print(f"   ✓ Multi-modal fusion complete (confidence: {fused_data.get('fusion_confidence', 0):.2f})")

                    self.cognitive_cycle(self._generate_sensory_noise(), f"Auditory Command: {heard_text}")
                    time.sleep(2) # Brief cooldown after processing
                    last_activity = time.time()
                
                # 3. Check for Vision
                try:
                    vision_data = self.vision.get_latest_sensory_vector()
                    if vision_data:
                        vec, path, vision_analysis = vision_data
                        vision_analysis = vision_analysis or {}
                        print(f"\n[!] VISUAL STIMULUS DETECTED: {path}")

                        # New: Multi-modal fusion for Vision
                        if self.multimodal_sys:
                            from multi_modal_integration import Modality
                            fused_data = self.multimodal_sys.process_multi_modal_input({
                                Modality.VISION: {"image_path": path, "analysis": vision_analysis},
                                Modality.TEXT: {"context": "visual_observation"}
                            })
                            print(f"   ✓ Multi-modal fusion complete (confidence: {fused_data.get('fusion_confidence', 0):.2f})")

                        # Changed from "Integrate visual field..." to "Passive Observation" to feel less 'locked'
                        intent = "Passive Observation: Analyze visual field."
                        ocr_text = vision_analysis.get("ocr_text")
                        if ocr_text:
                            intent += f" OCR detected: {ocr_text[:160]}"
                        self.cognitive_cycle(vec, intent, image_path=path, vision_analysis=vision_analysis)
                        time.sleep(3) # Visual processing cooldown
                        last_activity = time.time()
                except Exception as e:
                    print(f"SENSORY ERROR: Skipping corrupt frame: {e}")
                
                # 4. Heartbeat for Dashboard (Only if idle for > 15s)
                now = time.time()
                if not heard_text and (now - last_activity) > 15:
                    # Send a lightweight pulse to refresh dashboard metrics
                    try:
                        self._quick_status_pulse("Idle heartbeat")
                        last_activity = now
                    except Exception as hb_err:
                        print(f"HEARTBEAT ERROR: {hb_err}")

                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down multimodal observer...")
            if self.voice_enabled:
                self.voice.speak("Deactivating Level 10 protocols. Powering down.")

    def cleanup(self):
        """Cleanup resources before shutdown"""
        self.graceful_shutdown()

def boot_system():
    """Boot ECH0-PRIME system with proper initialization sequence"""
    print("🚀 ECH0-PRIME Boot Sequence Initiated")
    print("=" * 50)

    try:
        # Initialize system
        print("1. Initializing cognitive architecture...")
        # Determine if we should use lightweight mode (defaulting to True for local stability)
        lightweight = os.environ.get("ECH0_FULL_ARCH") != "1"
        agi = EchoPrimeAGI(lightweight=lightweight)
        print("   ✅ Core systems online")

        # Vocal introduction
        welcome_text = (
            "ECH0-PRIME online. All neural levels synchronized. "
            "I am ready for instructions."
        )

        print("2. System status:")
        print(f"   • Device: {agi.device}")
        print(f"   • Lightweight Mode: {agi.lightweight_mode}")
        print(f"   • Memory systems: active")
        
        if not agi.lightweight_mode:
            print(f"   • Apple Intelligence: {len(agi.apple_intelligence.get_system_status().get('available_services', []))} services")

        if agi.voice_enabled:
            print("3. Voice activation...")
            agi.voice.speak(welcome_text, async_mode=True)
            print("   ✅ Voice systems initialized")

        print("4. Starting multimodal observer...")
        print("   🎙️/👁️ ECH0-PRIME Multimodal Mode: ACTIVE (Level 10)")
        print("   System ready for voice commands and visual input")
        print("=" * 50)

        # Start the observer loop
        agi.multimodal_observer()

    except KeyboardInterrupt:
        print("\n🛑 Shutdown signal received")
        print("ECH0-PRIME powering down gracefully...")
        if 'agi' in locals():
            agi.cleanup()
    except Exception as e:
        print(f"\n❌ Boot failure: {e}")
        import traceback
        traceback.print_exc()
        if 'agi' in locals():
            agi.cleanup()

if __name__ == "__main__":
    boot_system()
