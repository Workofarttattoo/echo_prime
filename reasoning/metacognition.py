from typing import Dict, Any, List
import json
import re

class MetacognitiveCritic:
    """
    Implements a System 2 'Monitor' that critiques ECH0's reasoning traces.
    Inspired by 'Reflexion' and 'Thought-Correction' architectures (2025).
    """
    def __init__(self, bridge: Any):
        self.bridge = bridge

    def critique(self, goal: str, reasoning_trace: str) -> Dict[str, Any]:
        """
        Analyzes a reasoning trace for logical fallacies, tool-call syntax errors, 
        and factual inconsistencies.
        """
        system_prompt = (
            "You are the INTERNAL CRITIC for ECH0-PRIME. Your job is to check the reasoning of the main agent.\n"
            "CRITIQUE GUIDELINES:\n"
            "1. LOGIC: Does the conclusion follow the premises?\n"
            "2. TOOLS: Is the ACTION syntax correct? (e.g. ACTION: {'tool': 'name', 'args': {}})\n"
            "3. GOAL: Does this actually move us closer to the user's objective?\n"
            "4. CONSISTENCY: Does the response contradict itself or previous context?\n\n"
            "OUTPUT: Return a JSON object with 'valid' (bool), 'errors' (list), and 'suggested_correction' (string)."
        )

        prompt = (
            f"GOAL: {goal}\n\n"
            f"AGENT REASONING TRACE:\n{reasoning_trace}\n\n"
            "Evaluate this trace. If it is 100% correct and ready, set valid=true."
        )

        try:
            # We use a very low temperature for the critic to ensure deterministic rigor
            response = self.bridge.query(prompt, system=system_prompt, temperature=0.0)
            
            # Simple JSON extraction
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # Fallback
                return {"valid": True, "errors": [], "suggested_correction": ""}
        except Exception as e:
            return {"valid": True, "errors": [f"Critic Error: {e}"], "suggested_correction": ""}

class System2Orchestration:
    """
    Manages the slow-thought iteration loop (o1-style).
    """
    def __init__(self, main_orchestrator: Any):
        self.orchestrator = main_orchestrator
        self.critic = MetacognitiveCritic(main_orchestrator.llm_bridge)

    def execute_with_reflection(self, context: Dict[str, Any], mission_params: Dict[str, Any], max_iters: int = 2) -> Dict[str, Any]:
        """
        Executes a reasoning loop with internal critique cycles.
        """
        goal = mission_params.get("goal", self.orchestrator.current_goal)
        
        # Initial Draft
        result = self.orchestrator.reason_about_scenario(context, mission_params)
        
        # Reflection Loop
        for i in range(max_iters):
            critique_results = self.critic.critique(goal, result["llm_insight"])
            
            if critique_results.get("valid", True):
                break
                
            print(f" [🔍 CRITIC LIGHT]: self-correction cycle {i+1} triggered.")
            
            # Re-inject critique as feedback
            feedback_params = mission_params.copy()
            feedback_params["goal"] = (
                f"{goal}\n\n[CRITIC FEEDBACK]: Your previous response had these issues: "
                f"{', '.join(critique_results.get('errors', []))}. "
                f"Suggested direction: {critique_results.get('suggested_correction', '')}. "
                "Please refine your reasoning."
            )
            
            result = self.orchestrator.reason_about_scenario(context, feedback_params)

        return result
