import numpy as np
import json
import os
import re
from typing import Dict, List, Any, Optional

# Import real probabilistic reasoning
from reasoning.probabilistic import ProbabilisticReasoning as RealProbabilisticReasoning
from reasoning.symbolic_manipulator import EnhancedMathematicalReasoner
from reasoning.numerical_engine import EnhancedNumericalReasoner
from reasoning.word_problem_parser import get_word_problem_solver
from core.spatial_reasoning import EnhancedVisualReasoner
from reasoning.task_manager import TaskManager

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
                 arxiv: ArxivScanner = None):
                 
        self.probabilistic = RealProbabilisticReasoning(latent_dim=100, device="cpu")
        self.analogy = AnalogicalReasoning()
        self.causal = RealCausalDiscovery(alpha=0.05)

        # Enhanced reasoning capabilities for benchmark weaknesses
        self.mathematical_reasoner = EnhancedMathematicalReasoner()
        self.numerical_reasoner = EnhancedNumericalReasoner()
        self.word_problem_parser = get_word_problem_solver()
        self.visual_reasoner = EnhancedVisualReasoner()
        self.use_llm = use_llm
        
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
        scan_local_tools(os.path.join(project_root, "capabilities"))

        # Phase 5: Goal-Directed Autonomy
        self.current_goal = "Assistant: Awaiting instructions from Joshua."
        self.mission_history = []
        
        # System 2: Metacognitive Reflection (2025 Architecture)
        from reasoning.metacognition import MetacognitiveCritic
        self.critic = MetacognitiveCritic(self.llm_bridge) if self.llm_bridge else None
        
        # Task Management
        self.task_manager = TaskManager()

    def set_goal(self, goal: str):
        """Sets a long-term autonomous mission."""
        self.current_goal = goal
        self.mission_history = []
        print(f"MISSION INITIALIZED: {goal}")

    def benchmark_solve(self, question: str, choices: Optional[List[str]] = None, 
                        context: Optional[Dict] = None, task_type: str = "general") -> str:
        """
        Standardized solver for benchmark questions.
        Ensures LLM induction and proper choice selection.
        """
        mission_params = {"goal": question}
        context = context or {}
        
        # Use existing reason_about_scenario logic
        result = self.reason_about_scenario(context, mission_params)
        response = result.get("llm_insight", "")
        
        # If choices are provided, we should ensure the response matches one
        if choices:
            # Try to find an explicit choice mentioned in the response
            # Look for "The answer is A" or similar
            import re
            for i, choice in enumerate(choices):
                # Check for "Choice 1", "Option 1", "1.", etc if 1-indexed
                # Check for "A", "B", "C", "D" if lettered
                label = chr(ord('A') + i)
                if re.search(rf'\b{label}\b', response):
                    return label # Return the label for consistency with typical benchmarks
                
            # If no label found, return the raw response or try to match text
            for choice in choices:
                if choice.lower() in response.lower():
                    return choice

        return response.strip()

    def _contains_math_expression(self, query: str) -> bool:
        """Check if query contains mathematical expressions"""
        import re
        # Look for numbers with operators
        math_pattern = r'\d+\s*[\+\-\*\/×÷]\s*\d+'
        return bool(re.search(math_pattern, query))

    def _is_word_problem(self, query: str) -> bool:
        """Check if query is a word problem (natural language math)"""
        query_lower = query.lower()

        # Word problems typically have these characteristics:
        word_problem_indicators = [
            # Multiple quantities mentioned
            len(re.findall(r'\d+', query)) >= 2,
            # Story-like elements
            any(word in query_lower for word in ['has', 'have', 'gives', 'gave', 'takes', 'took', 'buys', 'bought', 'sells', 'sold']),
            # Question words
            any(word in query_lower for word in ['how many', 'how much', 'what is the total', 'what is left']),
            # Entities (people, objects)
            any(word in query_lower for word in ['john', 'mary', 'sarah', 'apples', 'oranges', 'balls', 'books', 'dollars'])
        ]

        # Must have at least 2 indicators to be considered a word problem
        return sum(word_problem_indicators) >= 2

    def _solve_word_problem(self, query: str) -> Optional[Dict]:
        """Solve a word problem using NLP parsing"""
        try:
            # Parse the word problem
            parsed_problem = self.word_problem_parser.parse_word_problem(query)

            # Solve the parsed problem
            solution = self.word_problem_parser.solve_word_problem(parsed_problem)

            if solution and solution.get('confidence', 0) > 0.3:
                return {
                    'method': 'word_problem_nlp',
                    'result': solution,
                    'confidence': solution.get('confidence', 0.5)
                }

            return None

        except Exception as e:
            print(f"Word problem parsing failed: {e}")
            return None

    def enhanced_reasoning(self, query: str, context: Optional[Dict] = None) -> Optional[Dict]:
        """
        Enhanced reasoning that addresses benchmark weaknesses.
        """
        query_lower = query.lower()

        # Word problem solving - check if this is a natural language math problem
        if self._is_word_problem(query):
            try:
                result = self._solve_word_problem(query)
                if result and result.get('confidence', 0) > 0.4:
                    return result
            except Exception as e:
                print(f"Word problem solving failed: {e}")

        # Mathematical problem solving - expanded keywords
        math_keywords = ['solve', 'calculate', 'compute', 'equation', 'derivative', 'integral',
                        'what is', 'how much', 'find', 'determine', 'evaluate']
        if any(keyword in query_lower for keyword in math_keywords) or self._contains_math_expression(query):
            try:
                result = self.mathematical_reasoner.solve_equation(query)
                if result.get('confidence', 0) > 0.5:  # Only return if reasonably confident
                    return {
                        'method': 'enhanced_mathematical',
                        'result': result,
                        'confidence': result.get('confidence', 0.5)
                    }
            except Exception as e:
                print(f"Enhanced mathematical reasoning failed: {e}")

        # Numerical computation with precision - expanded keywords
        numerical_keywords = ['precise', 'accuracy', 'decimal', 'fraction', 'numerical',
                             'what is', 'how much', 'calculate', 'compute', 'evaluate']
        if any(keyword in query_lower for keyword in numerical_keywords) or self._contains_math_expression(query):
            try:
                result = self.numerical_reasoner.solve_system([query])
                if result.get('converged', False) or 'solution' in str(result.get('solution', '')):
                    return {
                        'method': 'enhanced_numerical',
                        'result': result,
                        'confidence': result.get('confidence', 0.5)
                    }
            except Exception as e:
                print(f"Enhanced numerical reasoning failed: {e}")

        # Try basic arithmetic evaluation for simple expressions
        if self._contains_math_expression(query):
            try:
                # Extract and evaluate simple arithmetic
                import re
                expr_match = re.search(r'(\d+\s*[\+\-\*\/]\s*\d+)', query)
                if expr_match:
                    expr = expr_match.group(1).replace(' ', '')
                    result = eval(expr)
                    return {
                        'method': 'basic_arithmetic',
                        'result': {'solution': str(result), 'confidence': 0.8},
                        'confidence': 0.8
                    }
            except:
                pass

        # Visual/spatial reasoning
        if any(keyword in query_lower for keyword in ['visual', 'spatial', 'grid', 'pattern', 'transform']):
            try:
                visual_data = context.get('visual_data') if context else None
                result = self.visual_reasoner.solve_visual_problem(visual_data or {})
                return {
                    'method': 'enhanced_visual',
                    'result': result,
                    'confidence': result.get('confidence', 0.5)
                }
            except Exception as e:
                print(f"Enhanced visual reasoning failed: {e}")

        # Abstract reasoning for novel patterns
        if any(keyword in query_lower for keyword in ['pattern', 'abstract', 'reasoning', 'inference']):
            try:
                context_data = context.get('examples', []) if context else []
                result = self.numerical_reasoner.abstract_reasoner.apply_abstract_reasoning(
                    {'description': query}, context_data
                )
                return {
                    'method': 'enhanced_abstract',
                    'result': result,
                    'confidence': result.get('confidence', 0.5)
                }
            except Exception as e:
                print(f"Enhanced abstract reasoning failed: {e}")

        # Fall back to regular reasoning
        return None


    def record_action_result(self, action: Dict, result: str):
        """Records an action and its result into the mission history."""
        self.mission_history.append(f"ACTION: {json.dumps(action)}")
        self.mission_history.append(f"OBSERVATION: {result}")

    def reason_about_scenario(self, context: Dict[str, Any], mission_params: Dict[str, Any], memory_bank: list = None) -> Dict[str, Any]:
        """Uses enhanced reasoning for specific problem types, falls back to LLM."""
        query = mission_params.get("goal", self.current_goal)

        # First try enhanced reasoning for benchmark weaknesses
        enhanced_result = self.enhanced_reasoning(query, context)
        if enhanced_result and enhanced_result['result'].get('confidence', 0) > 0.6:
            # Use enhanced reasoning result
            result_data = enhanced_result['result']
            if enhanced_result['method'] == 'enhanced_mathematical':
                solution = result_data.get('solution', {})
                return {
                    "llm_insight": f"Mathematical solution: {solution.get('solution', 'No solution found')}",
                    "method": "enhanced_mathematical",
                    "confidence": solution.get('confidence', 0.5),
                    "mission_complete": True,
                    "current_goal": query
                }
            elif enhanced_result['method'] == 'enhanced_visual':
                visual_result = result_data.get('result', {})
                return {
                    "llm_insight": f"Visual pattern solution: {visual_result.get('output', 'No solution found')}",
                    "method": "enhanced_visual",
                    "confidence": visual_result.get('confidence', 0.5),
                    "mission_complete": True,
                    "current_goal": query
                }

        # Fall back to LLM reasoning
        if not self.llm_bridge:
            return {"llm_insight": "LLM Reasoning disabled.", "mission_complete": False, "current_goal": self.current_goal}

        # 1. RAG: Retrieve context from PersistentMemory and Pinecone
        retrieved_context = ""
        query = mission_params.get("goal", self.current_goal)
        
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


        # 2. Level 10 Unified System Prompt (Persona + Governance)
        # Inject dynamic tool schemas from MCP registry (using RAG for discovery)
        mission_goal = mission_params.get("goal", self.current_goal)
        tool_schemas = json.dumps(ToolRegistry.get_relevant_schemas(mission_goal), indent=2)
        
        # Load Protected Identity
        identity_path = os.path.join(os.path.dirname(__file__), "IDENTITY.txt")
        identity = "You are ECH0-PRIME."
        if os.path.exists(identity_path):
            with open(identity_path, "r") as f:
                identity = f.read()

        level_10_prompt = (
            f"{identity}\n"
            f"{retrieved_context}"
            "\nOPERATIONAL DIRECTIVES:\n"
            "1. GOVERNANCE: Verify all strong claims using FactChecker logic. If unsure, express uncertainty.\n"
            "2. TOOL USAGE: Call tools using: ACTION: {'tool': 'tool_name', 'args': {...}}\n"
            f"AVAILABLE TOOLS (JSON SCHEMAS):\n{tool_schemas}\n"
        )
        
        # Load latest plan for context
        self.task_manager.load_latest_plan()
        current_plan = self.task_manager.get_plan_view()
        if "No active plan found" not in current_plan and "Objective:" in current_plan:
             level_10_prompt += f"\n--- CURRENT PLAN ---\n{current_plan}\n"
         
        mission_goal = mission_params.get("goal", self.current_goal)
        history_str = "\n".join(self.mission_history[-3:])
        
        # Image Handling
        image_path = context.get("image_path")
        images = [image_path] if image_path and os.path.exists(image_path) else None
        
        is_math = self._contains_math_expression(mission_goal) or self._is_word_problem(mission_goal)
        
        cot_instructions = ""
        if is_math:
            cot_instructions = (
                "\nTHINKING STEP-BY-STEP (Chain-of-Thought):\n"
                "1. EXTRACT: First, list ALL quantities, constraints, and relationships given in the problem.\n"
                "2. PLAN: Identify what the question is asking for.\n"
                "3. SOLVE: Perform calculations for each step, showing your work.\n"
                "4. VERIFY: Double-check your answer by substituting back into the original problem.\n"
                "5. FORMAT: Your final answer MUST be exactly: #### <number>\n"
                "   (Use only the final numeric answer after ####, no units or text)\n"
            )

        prompt = (
            f"--- CURRENT MISSION GOAL ---\n{mission_goal}\n\n"
            f"{cot_instructions}"
            f"--- MISSION HISTORY ---\n{history_str}\n\n"
            f"--- LIVE SENSORY CONTEXT ---\n{json.dumps(context, indent=2)}\n\n"
            "Analyze the situation. If an image is provided, describe it and integrate it into your reasoning. "
            "Speak to Joshua as a friend. If the goal is achieved, include 'MISSION_STATUS: ACHIEVED'."
        )

        # Use vision model if image is present, otherwise standard model
        bridge = self.vision_bridge if images else self.llm_bridge
        response = bridge.query(prompt, system=level_10_prompt, images=images)
        
        # --- METACOGNITIVE REFLECTION (Self-Correction Loop) ---
        if self.critic and not mission_params.get("_internal_reflection"):
            critique = self.critic.critique(mission_goal, response)
            if not critique.get("valid", True):
                print(f" [🔍 REFLECTION]: Correcting thought trace... ({critique.get('errors')[0]})")
                refined_prompt = prompt + f"\n\n[INTERNAL CRITIQUE]: {critique.get('suggested_correction')}\nPlease provide a corrected final response."
                # Mark as internal to prevent infinite recursion
                mission_params["_internal_reflection"] = True
                response = bridge.query(refined_prompt, system=level_10_prompt, images=images)

        mission_achieved = "MISSION_STATUS: ACHIEVED" in response
        self.mission_history.append(f"Thought: {response[:100]}...")
        
        return {
            "llm_insight": response,
            "mission_complete": mission_achieved,
            "current_goal": mission_goal
        }
