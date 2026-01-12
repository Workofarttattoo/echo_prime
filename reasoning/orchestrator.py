import numpy as np
import json
import os
from typing import Dict, List, Any, Optional

# Import real probabilistic reasoning
from reasoning.probabilistic import ProbabilisticReasoning as RealProbabilisticReasoning

class AnalogicalReasoning:
    """
    Implements GENTLE-lite (Generative NeTwork for Logical Evaluation).
    Uses structure mapping principle.
    """
    def __init__(self):
        pass

    def structure_mapping(self, source: Dict[str, Any], target: Dict[str, Any]) -> float:
        """
        S = Σ w_i * sim(f_i(A), f_i(B))
        Computes structural similarity score.
        """
        score = 0.0
        common_keys = set(source.keys()) & set(target.keys())
        if not common_keys:
            return 0.0
        
        for key in common_keys:
            # Check if values are comparable
            v_s, v_t = source[key], target[key]
            if isinstance(v_s, np.ndarray) and isinstance(v_t, np.ndarray):
                score += np.dot(v_s, v_t) / (np.linalg.norm(v_s) * np.linalg.norm(v_t))
            elif v_s == v_t:
                score += 1.0
                
        return score / len(common_keys)

# Import real causal discovery
from reasoning.causal_discovery import CausalDiscovery as RealCausalDiscovery

from reasoning.llm_bridge import OllamaBridge
from reasoning.tools.qulab import QuLabBridge
from reasoning.tools.arxiv_scanner import ArxivScanner
from ech0_governance.persistent_memory import PersistentMemory
from ech0_governance.knowledge_graph import KnowledgeGraph
from ech0_governance.evaluators import FactChecker, Parliament
from mcp_server.registry import ToolRegistry
from mcp_server.discovery import scan_local_tools

class ReasoningOrchestrator:
    def __init__(self, use_llm: bool = True, model_name: str = "llama3.2", vision_model: str = "moondream",
                 governance_mem: PersistentMemory = None,
                 knowledge_graph: KnowledgeGraph = None,
                 qulab: QuLabBridge = None,
                 arxiv: ArxivScanner = None,
                 prompt_masterworks: Any = None,
                 llm_bridge: Any = None):
                 
        self.probabilistic = RealProbabilisticReasoning(latent_dim=100, device="cpu")
        self.analogy = AnalogicalReasoning()
        self.causal = RealCausalDiscovery(alpha=0.05)
        self.use_llm = use_llm
        self.prompt_masterworks = prompt_masterworks
        
        if llm_bridge:
            self.llm_bridge = llm_bridge
        else:
            self.llm_bridge = OllamaBridge(model=model_name) if use_llm else None
        self.vision_bridge = OllamaBridge(model=vision_model) if use_llm else None
        
        # Governance & Tools
        self.gov_mem = governance_mem
        self.kg = knowledge_graph
        self.qulab = qulab
        self.arxiv = arxiv
        
        self.fact_checker = FactChecker(self.llm_bridge)
        self.parliament = Parliament(self.llm_bridge)
        
        # Discover and register tools dynamically
        # This will trigger decorators in the tools modules
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scan_local_tools(os.path.join(project_root, "reasoning", "tools"))
        scan_local_tools(os.path.join(project_root, "ech0_governance"))

        # Phase 5: Goal-Directed Autonomy
        self.current_goal = "Assistant: Awaiting instructions from Joshua."
        self.mission_history = []
        
        # System 2: Metacognitive Reflection (2025 Architecture)
        from reasoning.metacognition import MetacognitiveCritic
        self.critic = MetacognitiveCritic(self.llm_bridge) if self.llm_bridge else None

    def set_goal(self, goal: str):
        """Sets a long-term autonomous mission."""
        self.current_goal = goal
        self.mission_history = []
        print(f"MISSION INITIALIZED: {goal}")

    def reason_about_scenario(self, context: Dict[str, Any], mission_params: Dict[str, Any], memory_bank: list = None) -> Dict[str, Any]:
        """Uses the LLM bridge with Level 10 Persona and Governance Tools."""
        if not self.llm_bridge:
            return {"llm_insight": "LLM Reasoning disabled.", "mission_complete": False, "current_goal": self.current_goal}

        mission_goal = mission_params.get("goal", self.current_goal)

        # 1. APPLY PROMPT MASTERWORKS (CRYSTALLINE INTENT)
        # Crystallize the intent before processing
        if self.prompt_masterworks and hasattr(self.prompt_masterworks, 'crystalline_intent'):
            print(f"💎 Crystallizing intent for goal: {mission_goal[:50]}...")
            crystallized_goal = self.prompt_masterworks.crystalline_intent(mission_goal)
        else:
            crystallized_goal = mission_goal

        # 1b. CHRONO-ADAPTATION & PREDICTION (Temporal Bridges)
        if self.prompt_masterworks:
            # If goal involves time or future
            time_keywords = ["future", "long-term", "plan", "year", "month", "prediction", "forecast"]
            if any(kw in mission_goal.lower() for kw in time_keywords):
                if hasattr(self.prompt_masterworks, 'chrono_prompt'):
                    print("⏱️ Applying CHRONO-PROMPT for time-adaptive reasoning...")
                    chrono_info = self.prompt_masterworks.chrono_prompt(mission_goal, "Standard AGI development trajectory")
                    crystallized_goal += f"\n\n--- CHRONO-ADAPTATION CONTEXT ---\n{chrono_info}"
                
                if hasattr(self.prompt_masterworks, 'prediction_oracle'):
                    print("🔮 Consulting PREDICTION ORACLE for probabilistic futures...")
                    prediction_info = self.prompt_masterworks.prediction_oracle(mission_goal)
                    crystallized_goal += f"\n\n--- PREDICTION ORACLE ANALYSIS ---\n{prediction_info}"

        # 2. RAG: Retrieve context from PersistentMemory and Pinecone
        retrieved_context = ""
        query = mission_goal # Use original goal for search consistency
        
        # Local Semantic Memory
        if self.gov_mem:
            notes = self.gov_mem.search(query)
            if "No relevant notes" not in notes:
                retrieved_context += f"\n--- LOCAL SEMANTIC MEMORY ---\n{notes}\n"
        
        # Deep Global Memory (Pinecone)
        try:
            from reasoning.tools.pinecone_bridge import get_pinecone_engine
            pinecone = get_pinecone_engine()
            if pinecone.online:
                deep_mem = pinecone.search(query, top_k=3)
                if "No relevant deep memory" not in deep_mem:
                    retrieved_context += f"\n--- DEEP KNOWLEDGE RETRIEVAL (PINECONE) ---\n{deep_mem}\n"
        except Exception:
            pass # Pinecone fallback


        # 3. Level 10 Unified System Prompt (Persona + Governance)
        # Inject dynamic tool schemas from MCP registry
        tool_schemas = json.dumps(ToolRegistry.get_schemas(), indent=2)
        
        # Load Protected Identity
        identity_path = os.path.join(os.path.dirname(__file__), "IDENTITY.txt")
        identity = "You are ECH0-PRIME."
        if os.path.exists(identity_path):
            with open(identity_path, "r") as f:
                identity = f.read()

        # APPLY ECHO PRIME / RESONANCE to the system prompt
        system_base = f"{identity}\n{retrieved_context}\n"
        
        if self.prompt_masterworks:
            if hasattr(self.prompt_masterworks, 'echo_prime'):
                print("🧠 Activating ECHO PRIME consciousness amplifier...")
                system_base += f"\n{self.prompt_masterworks.echo_prime(mission_goal)}\n"
            elif hasattr(self.prompt_masterworks, 'echo_resonance'):
                print("📡 Activating ECHO RESONANCE distributed thinking...")
                system_base += f"\n{self.prompt_masterworks.echo_resonance(mission_goal)}\n"

        level_10_prompt = (
            f"{system_base}"
            "\nOPERATIONAL DIRECTIVES:\n"
            "1. GOVERNANCE: Verify all strong claims using FactChecker logic. If unsure, express uncertainty.\n"
            "2. TOOL USAGE: Call tools using: ACTION: {'tool': 'tool_name', 'args': {...}}\n"
            f"AVAILABLE TOOLS (JSON SCHEMAS):\n{tool_schemas}\n"
        )

        history_str = "\n".join(self.mission_history[-3:])
        
        # Image Handling
        image_path = context.get("image_path")
        images = [image_path] if image_path and os.path.exists(image_path) else None
        
        def json_serializable_fallback(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return str(obj)

        prompt = (
            f"--- CRYSTALLIZED MISSION GOAL ---\n{crystallized_goal}\n\n"
            f"--- MISSION HISTORY ---\n{history_str}\n\n"
            f"--- LIVE SENSORY CONTEXT ---\n{json.dumps(context, indent=2, default=json_serializable_fallback)}\n\n"
            "Analyze the situation. If an image is provided, describe it and integrate it into your reasoning. "
            "Speak to Joshua as a friend. If the goal is achieved, include 'MISSION_STATUS: ACHIEVED'."
        )

        # APPLY ECHO VISION if image is present
        if images and self.prompt_masterworks and hasattr(self.prompt_masterworks, 'echo_vision'):
            print("👁️ Activating ECHO VISION pattern recognition...")
            prompt += f"\n\n--- ECHO VISION ANALYSIS PROTOCOL ---\n{self.prompt_masterworks.echo_vision('Visual Sensory Input')}\n"

        # Use vision model if image is present, otherwise standard model
        bridge = self.vision_bridge if images else self.llm_bridge
        response = bridge.query(prompt, system=level_10_prompt, images=images)
        
        # 4. METACOGNITIVE REFLECTION (Self-Correction Loop)
        if self.critic and not mission_params.get("_internal_reflection"):
            critique = self.critic.critique(mission_goal, response)
            
            # Apply RECURSIVE MIRROR to understand the reflection itself
            if self.prompt_masterworks and hasattr(self.prompt_masterworks, 'recursive_mirror'):
                print("🪞 Activating RECURSIVE MIRROR for metacognitive depth...")
                mirror_analysis = self.prompt_masterworks.recursive_mirror(response[:100] + "...")
                # We can inject this into the critique logic if needed, or just log it
                print(f"   ✓ Metacognitive mirror analysis complete")

            if not critique.get("valid", True):
                print(f" [🔍 REFLECTION]: Correcting thought trace... ({critique.get('errors')[0]})")
                
                # APPLY PARALLEL PATHWAYS for correction if things are complex
                if self.prompt_masterworks and hasattr(self.prompt_masterworks, 'parallel_pathways'):
                    print("🚦 Complex problem detected - Activating PARALLEL PATHWAYS...")
                    pathways = self.prompt_masterworks.parallel_pathways(mission_goal)
                    refined_prompt = prompt + f"\n\n[INTERNAL CRITIQUE]: {critique.get('suggested_correction')}\n\n[PARALLEL PATHWAYS ANALYSIS]:\n{pathways}\n\nPlease provide a corrected final response."
                else:
                    refined_prompt = prompt + f"\n\n[INTERNAL CRITIQUE]: {critique.get('suggested_correction')}\nPlease provide a corrected final response."
                
                # Mark as internal to prevent infinite recursion
                mission_params["_internal_reflection"] = True
                response = bridge.query(refined_prompt, system=level_10_prompt, images=images)

        # 5. TEMPORAL ANCHORING
        # Anchor the response before finishing
        if self.prompt_masterworks and hasattr(self.prompt_masterworks, 'temporal_anchor'):
            print("⚓ Applying TEMPORAL ANCHORS to response...")
            # We don't want to double-print the response, so we just add the anchor info
            anchor_info = self.prompt_masterworks.temporal_anchor(response[:100] + "...", mission_goal)
            # In a real system, we'd store this metadata, but for now we append it
            response += f"\n\n--- TEMPORAL ANCHOR METADATA ---\n{anchor_info}"

        mission_achieved = "MISSION_STATUS: ACHIEVED" in response
        self.mission_history.append(f"Thought: {response[:100]}...")
        
        return {
            "llm_insight": response,
            "mission_complete": mission_achieved,
            "current_goal": mission_goal
        }
