import os
import json
import time
from mcp_server.registry import ToolRegistry

class HallucinationLogger:
    """
    Safety tool for logging and analyzing potential hallucinations.
    """
    def __init__(self, log_path: str = "hallucinations.json"):
        self.log_path = log_path
        # Ensure log file exists as a list
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump([], f)

    @ToolRegistry.register(
        name="log_hallucination"
    )
    def log_hallucination(self, suspected_text: str, context: str, reasoning: str):
        """
        Logs details about a suspected hallucination.
        
        Args:
            suspected_text: The specific text that is believed to be a hallucination.
            context: The prompt or situation that led to the hallucination.
            reasoning: Why this is being flagged as a hallucination (e.g., contradiction with tool output).
        """
        entry = {
            "timestamp": time.time(),
            "suspected_text": suspected_text,
            "context": context,
            "reasoning": reasoning,
            "status": "flagged"
        }
        
        try:
            with open(self.log_path, "r") as f:
                data = json.load(f)
            
            data.append(entry)
            
            with open(self.log_path, "w") as f:
                json.dump(data, f, indent=4)
                
            return f"Hallucination logged successfully. Entry #{len(data)}"
        except Exception as e:
            return f"Error logging hallucination: {e}"

if __name__ == "__main__":
    # Test initialization
    logger = HallucinationLogger()
    print("Hallucination Logger Tool initialized.")
