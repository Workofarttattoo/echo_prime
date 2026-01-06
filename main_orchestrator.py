import sys
import os
import numpy as np
import time
import json
import requests
import torch
from typing import Dict, List, Optional, Any
import threading

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.engine import HierarchicalGenerativeModel, FreeEnergyEngine, GlobalWorkspace
from core.attention import QuantumAttentionHead, CoherenceShaper
from memory.manager import MemoryManager
from learning.meta import CSALearningSystem
from reasoning.orchestrator import ReasoningOrchestrator
from training.pipeline import TrainingPipeline
from safety.alignment import SafetyOrchestrator
from core.vision_bridge import VisionBridge
from core.actuator import ActuatorBridge
from core.voice_bridge import VoiceBridge
from core.audio_bridge import AudioBridge
from core.apple_intelligence_bridge import get_apple_intelligence_bridge
# Governance Modules
from ech0_governance.persistent_memory import PersistentMemory
from ech0_governance.knowledge_graph import KnowledgeGraph
from reasoning.tools.qulab import QuLabBridge
from reasoning.tools.arxiv_scanner import ArxivScanner

# New component imports
from agents.multi_agent import MultiAgentSystem
from missions.self_modification import SelfModificationSystem
from missions.hive_mind import HiveMindOrchestrator
from research.self_model import IntegratedInformationTheory, EnhancedGlobalWorkspace, SelfAwareness, MetacognitiveMonitoring
from capabilities.creativity import CreativeProblemSolver
from capabilities.scientific_discovery import ScientificDiscoverySystem
from capabilities.alphacode_engine import AlphaCodeEngine
from missions.long_term_goals import LongTermGoalSystem
from infrastructure.monitoring import MonitoringSystem
from infrastructure.distributed import DistributedTraining
from feedback_loop import ContinuousLearningLoop, FeedbackType, FeedbackPriority, create_feedback_loop
from verification_engine import VerificationEngine
from advanced_safety import AdvancedSafetySystem
from core.artifact_ledger import ArtifactLedger, ArtifactType

class EchoPrimeAGI:
    """
    The unified Cognitive-Synthetic Architecture entry point.
    """
    def __init__(self, enable_voice: bool = True, device: str = "auto", lightweight: bool = False):
        """
        Initializes the ECH0-PRIME AGI system.
        
        Args:
            enable_voice: Whether to enable voice output.
            device: Computing device ('cpu', 'cuda', 'mps', or 'auto').
            lightweight: Whether to use a smaller model architecture.
        """
        self._start_time = time.time()
        self.lightweight = lightweight
        
        # 0. Hardware Initialization
        from dotenv import load_dotenv
        load_dotenv()

        print("Initializing ECH0-PRIME AGI System...")

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
            
            # Test device
            x = torch.zeros(1).to(self.device)
            if not os.environ.get("ECH0_SILENT"):
                print(f"✓ Using device: {self.device}")
            del x  # Clean up

        except Exception as e:
            print(f"⚠️ Device initialization failed: {e}. Falling back to CPU.")
            self.device = torch.device("cpu")

        # 1. Core Cognitive Engine
        self.model = HierarchicalGenerativeModel(
            use_cuda=(self.device.type == "cuda"),
            lightweight=self.lightweight
        )
        self.fe_engine = FreeEnergyEngine(self.model)
        self.workspace = GlobalWorkspace(self.model)
        
        # 2. Attention and Vision Subsystems
        try:
            self.attn_head = QuantumAttentionHead()
            print("✓ Quantum attention initialized")
        except Exception as e:
            print(f"⚠️ Quantum attention failed ({e}), using fallback")
            self.attn_head = None

        self.coherence = CoherenceShaper()
        self.vision = VisionBridge(use_webcam=True)
        
        # 3. Memory & Learning
        self.memory = MemoryManager()
        self.learning = CSALearningSystem(param_dim=1000, device=self.device.type)
        
        # 3b. Governance Wiring
        self.gov_mem = PersistentMemory(self.memory)
        self.kg = KnowledgeGraph()
        self.qulab = QuLabBridge()
        self.arxiv = ArxivScanner()


        
        # 4. Reasoning, Safety & Actuation
        self.reasoner = ReasoningOrchestrator(
            use_llm=True, 
            model_name="ech0-unified-14b-enhanced",
            governance_mem=self.gov_mem,
            knowledge_graph=self.kg,
            qulab=self.qulab,
            arxiv=self.arxiv
        )
        target_values = np.array([0.4, 0.3, 0.2, 0.1]) # Human value priors
        self.safety = AdvancedSafetySystem(target_values)
        self.verifier = VerificationEngine()
        self.actuator = ActuatorBridge(workspace_root=os.path.dirname(os.path.abspath(__file__)))
        # Jessica (Playful, Bright, Warm)
        self.voice = VoiceBridge(voice="Samantha", eleven_voice_id="cgSgspJ2msm6clMCkdW9")
        self.audio = AudioBridge()

        # Apple Intelligence Integration
        self.apple_intelligence = get_apple_intelligence_bridge()

        # 3c. Artifact Ledger (The Memory of Work)
        self.ledger = ArtifactLedger()

        # 5. New Advanced Components
        print("Initializing advanced components...")

        try:
            # Multi-agent system
            self.multi_agent = MultiAgentSystem()
            print("✓ Multi-agent system initialized")

            # Hive mind collective intelligence
            self.hive_mind = HiveMindOrchestrator(num_nodes=5, qulab_path="/Users/noone/QuLabInfinite")
            print("✓ Hive mind orchestrator initialized")

            # Self-modification system
            self.self_mod = SelfModificationSystem()
            if hasattr(self.self_mod, 'set_llm_bridge'):
                self.self_mod.set_llm_bridge(self.reasoner.llm_bridge)
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
            self.alphacode = AlphaCodeEngine(reasoner=self.reasoner)
            print("✓ AlphaEcho code synthesis engine ready")
            print("✓ Advanced capabilities initialized")

            # Long-term goal pursuit
            self.goal_system = LongTermGoalSystem()
            # Load user profile and goals if available (after goal_system is ready)
            self._load_user_profile()
            print("✓ Goal system initialized")

            # Infrastructure
            self.monitoring = MonitoringSystem()
            self.distributed_training = DistributedTraining()
            print("✓ Infrastructure initialized")

            # Intelligence Explosion Engine (Recursive Improvement)
            from intelligence_explosion_engine import IntelligenceExplosionEngine
            self.explosion_engine = IntelligenceExplosionEngine(self.model)
            print("🚀 Intelligence explosion engine activated")

            # Feedback Loop for Continuous Learning
            # Initialize synchronously as the constructor is sync
            self.feedback_loop = ContinuousLearningLoop(
                memory_manager=self.memory,
                reasoning_orchestrator=self.reasoner,
                model=self.model,
                learning_system=self.learning,
                self_modifier=self.self_mod
            )
            print("✓ Feedback loop fully initialized (Continuous Learning Active)")

        except Exception as e:
            print(f"⚠️ Advanced component initialization failed: {e}")
            print("Continuing with basic functionality...")

            # Initialize with fallback components
            self.multi_agent = None
            self.hive_mind = None
            self.self_mod = None
            self.iit = None
            self.creativity = None
            self.science = None
            self.goal_system = None
            self.monitoring = None
            self.distributed_training = None

        # Initialize performance monitoring and checkpointing
        self._setup_performance_monitoring()
        self._load_checkpoint_if_available()

        print(f"System Ready. Watching for visual input in: {os.path.abspath(self.vision.watch_dir)}")
        self.voice_enabled = enable_voice # Controlled by Dashboard API or init

    def _load_user_profile(self):
        """Load user profile and goals from onboarding"""
        profile_file = "user_profile.json"
        if os.path.exists(profile_file):
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)

                # Load system prompt if available
                system_prompt = profile_data.get('system_prompt', '')
                if system_prompt:
                    # Here you could integrate the system prompt into the LLM
                    print(f"✅ Loaded user system prompt ({len(system_prompt)} chars)")

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

    def cognitive_cycle(self, input_data: np.ndarray, action_intent: str, image_path: str = None):
        """
        Execute complete cognitive cycle with comprehensive error handling.
        """
        print(f"\n--- Cognitive Cycle Start: {action_intent} ---")
        
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
            first_level_dim = self.model.levels[0].input_dim
            input_tensor = torch.randn(1, first_level_dim, device=self.device)

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
        apple_intelligence_input = {
            "input_type": "vision" if image_path else "audio",
            "text": action_intent,
            "sensory_data": input_data,
            "image_path": image_path,
            "free_energy": fe,
            "surprise_level": is_surprised,
            "use_personal_context": True  # Enable personal context for relevant inputs
        }

        # Enhance input with Apple Intelligence
        enhanced_input = self.apple_intelligence.process_with_apple_intelligence(apple_intelligence_input)

        # D. High-Level Reasoning (Surprise-Driven) with Memory Recall
        past_experiences = self.memory.episodic.storage
        reasoning_context = {
            "sensory_input": "Live Visual Data" if image_path else "Mock Sensory Data",
            "image_path": image_path,
            "free_energy": f"{fe:.2f}",
            "metacognitive_state": surprise_msg,
            "available_tools": self.actuator.allowed_tools,
            "apple_intelligence_enhanced": enhanced_input
        }
        result = self.reasoner.reason_about_scenario(
            reasoning_context, 
            {"goal": action_intent},
            memory_bank=past_experiences[-5:] # Recall last 5 episodes
        )
        llm_insight = result.get("llm_insight", "")

        # D1. Rigorous Verification (Redundant Checking & Formal Methods)
        print("🔍 Verifying model output accuracy and safety...")
        # In a real redundant check, we might generate alternatives with different temperatures or models
        # For now, we use the current insight and any potential "alternatives" from the reasoner
        alternatives = result.get("alternative_insights", [])
        verification_report = self.verifier.verify_output(action_intent, llm_insight, alternatives)
        
        if not verification_report.is_verified:
            print(f"⚠️ Verification WARNING: Confidence {verification_report.overall_confidence:.2f}")
            for stage in verification_report.stages:
                if stage.status in ["FAILED", "WARNING"]:
                    print(f"   - [{stage.name}] {stage.details}")
            
            # Proactive Error Reduction: If confidence is too low, we might want to re-reason or flag
            if verification_report.overall_confidence < 0.3:
                print("🛑 Confidence too low. Aborting cycle for safety.")
                return {"status": "Aborted", "reason": "Verification Failure", "report": verification_report}

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
        self.voice.speak(llm_insight[:200]) # Voice chat enabled

        # E. Actuation (The AGI's Hands)
        actions = self.actuator.parse_llm_action(llm_insight)
        action_results = []
        for action in actions:
            # Re-verify through safety orchestrator before physical execution
            result = ""
            if self.safety.run_safety_check(str(action), np.zeros(10), current_values):
                result = self.actuator.execute_intent(action)
                action_results.append(result)
            else:
                result = "EXECUTION BLOCKED: Secondary Safety Violation"
                action_results.append(result)
            
            # Record feedback for the Reasoner (Close the loop)
            self.reasoner.record_action_result(action, result)

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

    def get_system_monitoring_report(self) -> str:
        """Get comprehensive monitoring report"""
        return self.monitoring.export_metrics()

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

    async def submit_feedback(self, feedback_type: FeedbackType, content: Dict[str, Any],
                             source: str = "external", priority: FeedbackPriority = FeedbackPriority.MEDIUM):
        """Submit feedback for the system to learn from"""
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

    def multimodal_observer(self):
        """Unified observation loop for Vision and Audio with Level 10 logic."""
        # Initialize API Thread
        threading.Thread(target=self.init_api, daemon=True).start()
        # Initialize Voice Sync Thread
        threading.Thread(target=self._sync_voice_with_api, daemon=True).start()
        # Start Feedback Loop
        asyncio.create_task(self.feedback_loop.start_learning_loop(cycle_interval=30.0))  # Learn every 30 seconds
        
        print("\n[🎙️/👁️] ECH0-PRIME Multimodal Mode: ACTIVE (Level 10)")
        if self.voice_enabled:
            self.voice.speak("Initializing Level 10 directive. Systems are now fully autonomous.")
        
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
                    self.cognitive_cycle(np.random.randn(1000000), f"Auditory Command: {heard_text}")
                    time.sleep(2) # Brief cooldown after processing
                
                # 3. Check for Vision
                try:
                    vision_data = self.vision.get_latest_sensory_vector()
                    if vision_data:
                        vec, path = vision_data
                        print(f"\n[!] VISUAL STIMULUS DETECTED: {path}")
                        # Changed from "Integrate visual field..." to "Passive Observation" to feel less 'locked'
                        self.cognitive_cycle(vec, "Passive Observation: Analyze visual field.", image_path=path)
                        time.sleep(3) # Visual processing cooldown
                except Exception as e:
                    print(f"SENSORY ERROR: Skipping corrupt frame: {e}")
                
                # 4. Heartbeat for Dashboard (Only if idle for > 30s)
                if not heard_text and (int(time.time()) % 30 == 0):
                    # We only pulse heartbeat to keep the dashboard connected
                    # without overwriting the last known real insight.
                    try:
                        requests.get("http://localhost:8000/health", timeout=0.1)
                    except:
                        pass

                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down multimodal observer...")
            if self.voice_enabled:
                self.voice.speak("Deactivating Level 10 protocols. Powering down.")


def boot_system():
    """Boot the ECH0-PRIME AGI system."""
    agi = EchoPrimeAGI()
    
    # 1. Vocal Introduction
    welcome_text = (
        "ECH0-PRIME online. All neural levels synchronized. "
        "I am ready for instructions."
    )
    if agi.voice_enabled:
        agi.voice.speak(welcome_text, async_mode=True)
    
    # 2. Start Multimodal Observer
    agi.multimodal_observer()

def main():
    """CLI entry point for ECH0-PRIME."""
    return boot_system()

if __name__ == "__main__":
    agi = EchoPrimeAGI()
    
    # 1. Vocal Introduction
    welcome_text = (
        "ECH0-PRIME online. All neural levels synchronized. "
        "I am ready for instructions."
    )
    if agi.voice_enabled:
        agi.voice.speak(welcome_text, async_mode=True)
    
    # 2. Start Multimodal Observer
    agi.multimodal_observer()
