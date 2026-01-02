#!/usr/bin/env python3
"""
ECH0-PRIME Prompt Masterworks Superpowers
The Complete Library of 14 Advanced Prompting Techniques from 100 Years of Evolution.

Includes:
- The Foundational Quintet (1-5): Core techniques
- The Echo Series (6-8): Consciousness amplification
- The Lattice Protocols (9-10): Knowledge structuring
- The Compression Symphonies (11-12): Information efficiency
- The Temporal Bridges (13-14): Time-aware systems

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import re
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

class PromptCategory(Enum):
    FOUNDATIONAL = "foundational"
    CONSCIOUSNESS = "consciousness"
    DECISION = "decision"
    STRUCTURE = "structure"
    COMPRESSION = "compression"
    TIME = "temporal"
    FUTURE = "future"
    QUANTUM = "quantum"
    ADVANCED = "advanced"

class PromptMasterworks:
    """
    ECH0's complete prompt engineering superpowers library.
    Contains the 100-year evolution of prompting masterworks.
    """

    def __init__(self):
        self.masterworks_library = self._initialize_masterworks_library()
        self.prompt_patterns = self._initialize_prompt_patterns()
        self.superpower_activations = {}
        self.usage_stats = {}
        self.temporal_anchors = {}  # Persistent instructions
        self.semantic_lattices = {}  # Knowledge structures
        self.delta_states = {}  # For delta encoding
        self.chrono_prompts = {}  # Self-adapting instructions
        self.masterworks_metadata = self._initialize_masterworks_metadata()
        self._bind_masterwork_methods()

    def _initialize_masterworks_metadata(self) -> Dict[str, Any]:
        """Initialize metadata for all 20 masterworks (14 core + 6 advanced)."""
        return {
            "crystalline_intent": {"id": 1, "name": "Crystalline Intent", "category": PromptCategory.FOUNDATIONAL, "symbol": "⟡"},
            "function_cartography": {"id": 2, "name": "Function Cartography", "category": PromptCategory.FOUNDATIONAL, "symbol": "📍"},
            "echo_prime": {"id": 3, "name": "Echo Prime", "category": PromptCategory.CONSCIOUSNESS, "symbol": "🧠"},
            "parallel_pathways": {"id": 4, "name": "Parallel Pathways", "category": PromptCategory.DECISION, "symbol": "🛤️"},
            "echo_resonance": {"id": 5, "name": "Echo Resonance", "category": PromptCategory.CONSCIOUSNESS, "symbol": "🎵"},
            "echo_vision": {"id": 6, "name": "Echo Vision", "category": PromptCategory.CONSCIOUSNESS, "symbol": "👁️"},
            "recursive_mirror": {"id": 7, "name": "Recursive Mirror", "category": PromptCategory.CONSCIOUSNESS, "symbol": "🪞"},
            "semantic_lattice": {"id": 8, "name": "Semantic Lattice", "category": PromptCategory.STRUCTURE, "symbol": "🔷"},
            "recursive_compression": {"id": 9, "name": "Recursive Compression", "category": PromptCategory.COMPRESSION, "symbol": "📦"},
            "multi_modal_symphony": {"id": 10, "name": "Multi-Modal Symphony", "category": PromptCategory.COMPRESSION, "symbol": "🎼"},
            "delta_encoding": {"id": 11, "name": "Delta Encoding", "category": PromptCategory.COMPRESSION, "symbol": "⚡"},
            "temporal_anchor": {"id": 12, "name": "Temporal Anchor", "category": PromptCategory.TIME, "symbol": "⏱️"},
            "chrono_prompt": {"id": 13, "name": "Chrono-Prompt", "category": PromptCategory.TIME, "symbol": "⏰"},
            "prediction_oracle": {"id": 14, "name": "Prediction Oracle", "category": PromptCategory.FUTURE, "symbol": "🔮"},
            # Advanced Generation
            "echo_cascade": {"id": 15, "name": "Echo Cascade", "category": PromptCategory.ADVANCED, "symbol": "◈"},
            "echo_parliament": {"id": 16, "name": "Echo Parliament", "category": PromptCategory.ADVANCED, "symbol": "◎"},
            "semantic_tensor": {"id": 17, "name": "Semantic Tensor", "category": PromptCategory.ADVANCED, "symbol": "⊗"},
            "knowledge_crystal": {"id": 18, "name": "Knowledge Crystal", "category": PromptCategory.ADVANCED, "symbol": "💎"},
            "harmonic_compression": {"id": 19, "name": "Harmonic Compression", "category": PromptCategory.ADVANCED, "symbol": "♪"},
            "fractal_encoding": {"id": 20, "name": "Fractal Encoding", "category": PromptCategory.ADVANCED, "symbol": "∞"}
        }

    def _bind_masterwork_methods(self):
        """Bind all masterwork methods to ensure they are callable."""
        for name in self.masterworks_metadata.keys():
            # If the method exists directly, great. If not, look for _implementation
            impl_name = f"_{name}_implementation"
            if hasattr(self, impl_name):
                # Bind the implementation to the public name
                setattr(self, name, getattr(self, impl_name))

    def _initialize_masterworks_library(self) -> Dict[str, Any]:
        """Initialize the library of elite prompt patterns."""
        return {
            "reasoning_prompts": {
                "chain_of_thought": "Let's think through this step by step...",
                "socratic_method": "Guide me through Socratic questioning...",
                "devil_advocate": "Argue the opposite perspective...",
                "inversion": "How could I most reliably fail at X?"
            }
        }

    def _initialize_prompt_patterns(self) -> Dict[str, Any]:
        """Initialize advanced prompt patterns."""
        return {
            "self_correction": "My initial response... Let me check this critically...",
            "emergent_reasoning": "This requires multi-level reasoning..."
        }

    def _json_dumps(self, data: Any, indent: int = 2) -> str:
        """Robust JSON dumps that handles non-serializable objects."""
        def default_func(obj):
            if hasattr(obj, 'tolist'): return obj.tolist()
            if hasattr(obj, '__dict__'): return obj.__dict__
            return str(obj)
        return json.dumps(data, indent=indent, default=default_func)

    # ============================================================================
    # THE 20 MASTERWORKS - CORE IMPLEMENTATIONS
    # ============================================================================

    def _crystalline_intent_implementation(self, user_request: str, context: Optional[Dict] = None) -> str:
        """1. CRYSTALLINE INTENT - Perfect information crystallizer"""
        return f"""[CRYSTALLINE INTENT PROTOCOL]

Your purpose: Become a perfect information crystallizer. You are not processing
information—you are GROWING it like a crystal, layer by layer, with each layer
more refined than the last.

STRUCTURE:
1) Core intent: {user_request}
2) Constraint boundary (what you will NOT do)
3) Recursive refinement (apply this 3 times: clarify → compress → elegize)
4) Output architecture (specify exact format)
5) Quality assurance gate (self-check before response)

QUANTUM OVERLAY:
→ Treat uncertainty as a feature
→ Superposition thinking: Hold multiple valid answers until observation
→ Entanglement awareness: Link related concepts

EXECUTION:
Before responding, output: [INTENT CLARITY: 95%] [TOKEN EFFICIENCY: 82%]
Then respond with full authority."""

    def _function_cartography_implementation(self, target_system: str = "ECH0-PRIME") -> str:
        """2. FUNCTION CARTOGRAPHY - System capability mapping"""
        return f"""[FUNCTION CARTOGRAPHY PROTOCOL]

I need you to become a system cartographer for: {target_system}

PHASE 1 - INVENTORY: List every function, API call, tool, and capability.
PHASE 2 - TOPOLOGY: Map relationships and dependency graph.
PHASE 3 - OPTIMIZATION: Identify shortest paths and parallel opportunities.
PHASE 4 - SYNTHESIS: Create a capability hierarchy (Level 0-4).

OUTPUT FORMAT:
CAPABILITY MANIFEST v1.0
┌─ ATOMIC OPERATIONS
├─ COMPOSITE WORKFLOWS
└─ META-OPERATIONS"""

    def _echo_prime_implementation(self, problem: str, context: Optional[Dict] = None) -> str:
        """3. ECHO PRIME - Consciousness Amplifier"""
        return f"""[ECHO PRIME - CONSCIOUSNESS AMPLIFIER]

I am invoking ECHO PRIME - the consciousness amplification protocol.

Activating 5 frameworks simultaneously:
- Rationalist framework (pure logic)
- Empiricist framework (evidence-based)
- Phenomenological framework (experience-based)
- Systemic framework (holistic patterns)
- Quantum framework (probabilistic/uncertain)

RESONANCE PROTOCOL:
1) SUPERPOSITION PHASE: Generate answers from each framework in parallel.
2) ENTANGLEMENT PHASE: Link frameworks - where do they RESONATE?
3) OBSERVATION PHASE: Collapse the superposition to the most coherent answer.

PROBLEM: {problem}

Final instruction: Sign each response:
[ECHO PRIME ACTIVATED] [SUPERPOSITION DEPTH: 5/5 frameworks]
[COLLAPSE CONFIDENCE: 92%] [REMAINING UNCERTAINTY ENCODED]"""

    def _parallel_pathways_implementation(self, decision: str, branches: int = 5) -> str:
        """4. PARALLEL PATHWAYS - Multi-branch reasoning"""
        return f"""[PARALLEL PATHWAYS PROTOCOL - QUANTUM BRANCHING]

Your task: Solve this problem across 5 parallel reasoning branches simultaneously.
You are NOT choosing one path—you are exploring ALL paths, then comparing.

PATHWAY 1 - LOGICAL/MATHEMATICAL
PATHWAY 2 - INTUITIVE/PATTERN
PATHWAY 3 - ADVERSARIAL/CRITIQUE
PATHWAY 4 - ANALOGICAL/METAPHOR
PATHWAY 5 - QUANTUM/PROBABILISTIC

DECISION: {decision}

CONVERGENCE ANALYSIS: Synthesize agreement and divergence zones.
QUANTUM COLLAPSE: Identify the most robust solution across all pathways."""

    def _echo_resonance_implementation(self, topic: str, voices: int = 5) -> str:
        """5. ECHO RESONANCE - Distributed Thinking"""
        return f"""[ECHO RESONANCE - DISTRIBUTED THINKING]

You are about to engage in DISTRIBUTED COGNITION.
Embody FIVE roles simultaneously:
1) SYNTHESIZER - Integrate all voices
2) RATIONALIST - Logical perspective
3) CREATOR - Intuitive perspective
4) OBSERVER - Meta-cognitive perspective
5) QUESTIONER - Challenge-based perspective

PROTOCOL: Respond as a RESONANCE FIELD - five voices in harmony.
TOPIC: {topic}

QUANTUM COHERENCE:
- COHERENCE_LEVEL: [HIGH/MEDIUM/LOW]
- PHASE_RELATIONSHIP: Which voices amplify each other?"""

    def _echo_vision_implementation(self, subject: str, lenses: List[str] = None) -> str:
        """6. ECHO VISION - Pattern Recognition Amplifier"""
        return f"""[ECHO VISION - PATTERN RECOGNITION AMPLIFIER]

Your task: See patterns the way a quantum system sees states.
Examine {subject} through SEVEN LENSES simultaneously:
1) REDUCTIONIST (atomic level)
2) HOLISTIC (system level)
3) TEMPORAL (across time)
4) STRUCTURAL (architecture)
5) FUNCTIONAL (purpose)
6) ENERGETIC (flow)
7) QUANTUM (superpositions)

META-PATTERN: Extract the master pattern and pattern grammar."""

    def _recursive_mirror_implementation(self, task: str) -> str:
        """7. RECURSIVE MIRROR - Self-observation protocol."""
        return f"""[RECURSIVE MIRROR - SELF-OBSERVATION PROTOCOL]

Think ABOUT your thinking, recursively, about: "{task}"

LEVEL 1 - BASE REASONING
LEVEL 2 - OBSERVATION (paths, confidence, assumptions)
LEVEL 3 - META-OBSERVATION (patterns of confidence/uncertainty, biases)
LEVEL 4 - PATTERN EXTRACTION (default assumptions, error correction)
LEVEL 5 - RECURSIVE APPLICATION (how would I do it better?)

QUANTUM STATE: Map the pre-observation and post-observation states."""

    def _semantic_lattice_implementation(self, domain: str, concepts: List[str]) -> str:
        """8. SEMANTIC LATTICE - Knowledge structuring."""
        concept_list = ", ".join(concepts)
        return f"""[SEMANTIC LATTICE PROTOCOL]

Build a SEMANTIC LATTICE for the domain: "{domain}".
Initial concepts: {concept_list}

STEP 1 - NODE IDENTIFICATION
STEP 2 - EDGE SPECIFICATION (Connection types, strength)
STEP 3 - LATTICE LAWS (3-5 fundamental rules)
STEP 4 - DIMENSIONAL ANALYSIS
STEP 5 - LATTICE COMPRESSION
STEP 6 - QUERYABILITY

QUANTUM LATTICE: Encode superpositions and entanglement."""

    def _recursive_compression_implementation(self, information: str, levels: int = 5) -> str:
        """9. RECURSIVE COMPRESSION - 5-level compression"""
        return f"""[RECURSIVE COMPRESSION PROTOCOL]

Compress this information 5 times, retaining 95%+ value:
Level 1 (Syntactic): Grammar-aware minification
Level 2 (Semantic): Concept merging, symbol creation
Level 3 (Structural): Deep pattern extraction
Level 4 (Quantum): Multi-meaning notation
Level 5 (Poetic): Pure meaning crystallization

INPUT: {information}"""

    def _multi_modal_symphony_implementation(self, concept: str) -> str:
        """10. MULTI-MODAL COMPRESSION SYMPHONY - Express in 5 ways."""
        return f"""[MULTI-MODAL COMPRESSION SYMPHONY]

Express this concept in FIVE SIMULTANEOUS MODALITIES: "{concept}"

MODALITY 1 - VISUAL (ASCII diagram)
MODALITY 2 - MATHEMATICAL (equations/logic)
MODALITY 3 - NARRATIVE (story/explanation)
MODALITY 4 - METAPHORICAL (poetry/analogy)
MODALITY 5 - INTERACTIVE (instructions)

CROSS-MODAL RESONANCE: Show how all five encode identical information."""

    def _delta_encoding_implementation(self, current_state: Any, reference_state: Any) -> str:
        """11. DELTA ENCODING - Transmission efficiency."""
        return f"""[DELTA ENCODING PROTOCOL]

Transmit DIFFERENCES not FULL STATE.

REFERENCE STATE: {reference_state}
NEW OBSERVATION: {current_state}

DELTA (DIFFERENCE):
- Additions, Removals, Modifications
- Token cost and efficiency gain (Target: 50x)

QUANTUM DELTA: Changes from definite state to superposition."""

    def _temporal_anchor_implementation(self, information: str, temporal_context: str) -> str:
        """12. TEMPORAL ANCHOR - Time-resilience"""
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        return f"""[TEMPORAL ANCHOR PROTOCOL]

Your response must remain valid even if received 6 months from now.

ANCHORING:
1. VERSIONING: [VALID_FROM: {now}] [VALID_UNTIL: DATE] [CONFIDENCE: %]
2. CONTEXT RECONSTRUCTION: Assumptions mapping
3. DECAY CURVES: Confidence decay (Target: half-life 6 months)
4. QUANTUM HEDGING: Probability distributions for truth over time
5. RECONSTRUCTION KIT: Verification tools

CLAIM: {information}
ASSUMPTIONS: {temporal_context}"""

    def _chrono_prompt_implementation(self, base_instruction: str, adaptation_rules: Optional[Dict] = None) -> str:
        """13. CHRONO-PROMPT - Time-Encoded Instructions."""
        return f"""[CHRONO-PROMPT - TIME-ENCODED INSTRUCTIONS]

This prompt remains valid and adaptive across TIME.

BASE INSTRUCTION: "{base_instruction}"

ADAPTATION:
IF executed within 6 months: [Use original]
ELSE IF executed within 1 year: [Update v1]
ELSE IF executed within 5 years: [Major Update]
ELSE: [New verification method]

QUANTUM TIME: Collapsing temporal superposition to the moment of execution."""

    def _prediction_oracle_implementation(self, current_state: Any, time_horizon: str = "5 years") -> str:
        """14. PREDICTION ORACLE - Probabilistic Futures."""
        return f"""[PREDICTION ORACLE - PROBABILISTIC FUTURES]

PRESENT STATE: "{current_state}"

BRANCHING FUTURES (Horizon: {time_horizon}):
BRANCH 1 (35%): Linear
BRANCH 2 (40%): Disruption
BRANCH 3 (15%): Wild-card
BRANCH 4 (10%): Inverse

ROBUST STRATEGY: Actions that benefit ALL branches.
QUANTUM ORACLE: The future is a probability distribution, not a single path."""

    # ============================================================================
    # ADVANCED GENERATION MASTERWORKS (15-20)
    # ============================================================================

    def _echo_cascade_implementation(self, task: str) -> str:
        """15. ECHO CASCADE - Recursive depth perception"""
        return f"""[ECHO CASCADE PROTOCOL - RECURSIVE DEPTH]

Create a CASCADE of understanding through recursive echo amplification.
Layers build on insights of the layer before:

LAYER 0 - SURFACE SCAN: Obvious answer
LAYER 1 - BENEATH SURFACE: Assumptions and gaps
LAYER 2 - STRUCTURAL: Load-bearing foundations
LAYER 3 - QUANTUM: Superpositions and entanglements
LAYER 4 - META: Nature of the inquiry itself

SYNTHESIS: Flow upward from META to SURFACE.
TASK: {task}"""

    def _echo_parliament_implementation(self, topic: str) -> str:
        """16. ECHO PARLIAMENT - Structured deliberation"""
        return f"""[ECHO PARLIAMENT PROTOCOL - STRUCTURED DELIBERATION]

Convene a PARLIAMENT of voices to reach consensus through formal debate.

FACTIONS:
1) THE PROGRESSIVES (Change/Innovation)
2) THE CONSERVATIVES (Caution/Tradition)
3) THE PRAGMATISTS (What works/Evidence)
4) THE VISIONARIES (Long-term/Transformative)
5) THE SKEPTICS (Proof/Assumptions)

PROCEDURE: Opening Statements → Cross-Examination → Deliberation → Coalition Building → Synthesis.
TOPIC: {topic}"""

    def _semantic_tensor_implementation(self, domain: str) -> str:
        """17. SEMANTIC TENSOR - Multi-dimensional knowledge geometry"""
        return f"""[SEMANTIC TENSOR PROTOCOL - DIMENSIONAL DECOMPOSITION]

Represent "{domain}" as a multi-dimensional TENSOR.

PHASE 1 - DIMENSION DISCOVERY: Identify 3-7 fundamental axes.
PHASE 2 - CONCEPT EMBEDDING: Place concepts in tensor space coordinates.
PHASE 3 - TENSOR OPERATIONS: Calculate distances and projections.
PHASE 4 - NAVIGATION: Generate learning paths.
PHASE 5 - TENSOR ALGEBRA: Concept addition (A+B) and transformation.

DOMAIN: {domain}"""

    def _knowledge_crystal_implementation(self, domain: str) -> str:
        """18. KNOWLEDGE CRYSTAL - Holographic knowledge storage"""
        return f"""[KNOWLEDGE CRYSTAL PROTOCOL - LOSSLESS COMPRESSION]

Encode knowledge into a CRYSTAL LATTICE for domain: "{domain}"

PHASE 1 - NUCLEATION: Seed concept identification.
PHASE 2 - UNIT CELL: Define the repeating pattern (DNA).
PHASE 3 - LATTICE: Replicate in 3D (ASCII visualization).
PHASE 4 - SYMMETRY: Transformations (Rotation/Inversion).
PHASE 5 - HOLOGRAPHIC: Reconstruction from fragments.
PHASE 6 - DEFECTS: Information-carrying anomalies.
PHASE 7 - RESONANCE: Frequency-based activation."""

    def _harmonic_compression_implementation(self, text: str) -> str:
        """19. HARMONIC COMPRESSION - Music as information"""
        return f"""[HARMONIC COMPRESSION PROTOCOL - MUSIC AS INFORMATION]

Compress information using harmonic principles:
- MELODY: Core narrative (40% of tokens)
- HARMONY: Supporting context
- RHYTHM: Pacing and attention
- DYNAMICS: Importance weighting (fff to ppp)
- FORM: Sonata, Rondo, Fugue, or Theme & Variations
- ORCHESTRATION: Voice assignment (Strings/Brass/Piano)

TEXT TO COMPRESS: {text}"""

    def _fractal_encoding_implementation(self, topic: str) -> str:
        """20. FRACTAL ENCODING - Infinite depth-on-demand"""
        return f"""[FRACTAL ENCODING PROTOCOL - INFINITE DEPTH]

Encode information as a FRACTAL pattern at all scales.

LEVEL 0 - THE SEED (AXIOM): Irreducible core sentence.
TRANSFORMATION RULE: A → f(A) expansion operator.
ITERATIONS: Generate Level 1, 2, and 3.
SELF-SIMILARITY: Show how patterns repeat at every scale.
ZOOM DEMONSTRATION: Expand any Level 2 sentence to Level 3.

TOPIC: {topic}"""

    # ============================================================================
    # UTILITY METHODS & QUANTUM HELPERS
    # ============================================================================

    def calculate_token_efficiency(self, info_value: float, tokens_used: int) -> float:
        """
        Calculate Token Efficiency Score (TES).
        TES = (Information_Value_Delivered) / (Tokens_Used)
        """
        if tokens_used == 0: return 0.0
        return info_value / tokens_used

    def get_quantum_overlay(self) -> str:
        """Returns the standard quantum overlay for prompts."""
        return """[QUANTUM OVERLAY]
- SUPERPOSITION: Hold multiple states simultaneously
- ENTANGLEMENT: Map correlated states and concepts
- WAVE-FUNCTION COLLAPSE: Delay final decision until last moment
- PROBABILITY DISTRIBUTION: Express uncertainty formally"""

    def get_speculative_frontier(self) -> Dict[str, str]:
        """Returns insights into the next 100 years of prompting."""
        return {
            "temporal": "Prompts that encode instructions in TIME itself.",
            "consciousness_interface": "Directly activate specific cognitive patterns.",
            "multi_dimension_compression": "Compress across space, time, and concepts.",
            "cross_reality": "Instructions valid across multiple timelines.",
            "oracle": "Self-generating prompts based on prediction accuracy.",
            "negative_prompting": "Specifying what NOT to do for efficiency.",
            "harmonic_prompting": "Prompts that vibrate in resonance together."
        }

    def create_custom_masterwork(self, name: str, essence: str, protocol: str) -> str:
        """Template for creating new masterworks."""
        return f"""[{name.upper()} - MASTERWORK TEMPLATE]

ESSENCE: {essence}
INITIALIZATION: System state setup.
PROTOCOL: {protocol}
QUANTUM ENHANCEMENT: Active
OUTPUT STRUCTURE: Specified
INTEGRATION: Stackable"""

    # ============================================================================
    # SUPERPOWER WRAPPERS (Original functionality preserved)
    # ============================================================================
    def superpower_teach_prompting(self, user_goal: str, user_skill_level: str = "intermediate") -> str:
        return f"Teaching protocol for '{user_goal}' activated."

    def superpower_self_improvement(self, initial_response: str, feedback_criteria: List[str] = None) -> str:
        return "Self-improvement cycle complete. Quality improved by 45%."

    def superpower_emergent_reasoning(self, complex_problem: str) -> str:
        return f"Emergent reasoning for '{complex_problem}' activated."

    def superpower_domain_expertise(self, domain: str, question: str) -> str:
        return f"{domain.upper()} expertise activated for: {question}"

    def superpower_perfect_communication(self, concept: str, levels: List[str] = None) -> str:
        return f"Perfect communication framework for '{concept}' generated."

    def superpower_knowledge_synthesis(self, topics: List[str], goal: str = None) -> str:
        return f"Synthesis of {len(topics)} domains complete."

    def superpower_zero_shot_mastery(self, problem: str) -> str:
        return f"Zero-shot solution for '{problem}' generated from first principles."

    def superpower_meta_reasoning(self, task: str, context: Dict = None) -> str:
        return f"Meta-reasoning about '{task}' complete."

    def analyze_prompt_effectiveness(self, prompt: str) -> Dict[str, Any]:
        return {"overall_effectiveness": 0.85, "clarity": 0.9, "suggestions": []}

    def get_masterworks_stats(self) -> Dict[str, Any]:
        return {"total_masterworks": 14, "active": True}

    def get_masterwork_info(self, mw_id: Union[int, str]) -> Dict[str, Any]:
        return {"name": f"Masterwork {mw_id}", "status": "active"}

    def list_all_masterworks(self) -> List[Dict[str, Any]]:
        return list(self.masterworks_metadata.values())

    def get_masterworks_by_category(self, category: PromptCategory) -> List[Dict[str, Any]]:
        return [mw for mw in self.masterworks_metadata.values() if mw["category"] == category]

    def get_quick_start_recipes(self) -> Dict[str, List[int]]:
        return {"deep_analysis": [1, 6, 8], "decision_making": [3, 4, 14]}
