import json
import re
from typing import Dict, Any, Tuple
from reasoning.llm_bridge import OllamaBridge

class IntelligentGrader:
    """
    Uses a secondary LLM logic to evaluate the correctness of a response.
    Supports partial credit and semantic understanding.
    """
    def __init__(self, model: str = "ech0-unified-14b-enhanced"):
        self.bridge = OllamaBridge(model=model)

    def grade(self, question: str, expected_answer: str, model_response: str) -> Tuple[float, str]:
        """
        Grades a response against an expected answer.
        Returns: (score [0-1], justification)
        """
        system_prompt = (
            "You are an impartial academic grader. Your task is to evaluate if a student's response "
            "matches the reference answer for a specialized math or science problem.\n\n"
            "GRADING CRITERIA:\n"
            "1. If the final numerical or symbolic answer is present and correct, give 1.0.\n"
            "2. If the final answer is missing but the reasoning is perfectly sound and leads to the target, give 0.8.\n"
            "3. If the reasoning is partially correct but contains specific errors, give a score between 0.1 and 0.5.\n"
            "4. If the response is a 'Read Timeout' or 'Bridge Error', give 0.0.\n"
            "5. Ignore conversational pleasantries.\n\n"
            "OUTPUT FORMAT: Return ONLY a JSON object with 'score' (float) and 'justification' (string)."
        )

        prompt = (
            f"QUESTION: {question}\n"
            f"EXPECTED REFERENCE ANSWER: {expected_answer}\n"
            f"STUDENT RESPONSE: {model_response}\n\n"
            "Evaluate the student's work."
        )

        try:
            raw_response = self.bridge.query(prompt, system=system_prompt, temperature=0.1)
            
            # Remove markdown code blocks if present
            clean_response = re.sub(r'```json\n?|\n?```', '', raw_response).strip()
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
            if json_match:
                try:
                    # Try direct parse
                    result = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    # Attempt to fix single quotes to double quotes
                    fixed_json = json_match.group(0).replace("'", '"')
                    result = json.loads(fixed_json)

                score = float(result.get("score", 0.0))
                justification = result.get("justification", "No justification provided.")
                return score, justification
            else:
                # Heuristic fallback if no JSON found
                if "1.0" in clean_response or '"score": 1' in clean_response:
                    return 1.0, "Heuristic pass (failed JSON parse)."
                return 0.0, f"Failed to parse grader response: {raw_response[:100]}"
        except Exception as e:
            return 0.0, f"Grader Error: {str(e)}"


if __name__ == "__main__":
    # Test
    grader = IntelligentGrader()
    res = grader.grade("What is 2+2?", "4", "The result of adding two and two is four.")
    print(f"Test Score: {res}")
