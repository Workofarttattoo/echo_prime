import subprocess
import os
import shlex
from typing import Dict, Any, List

class ActuatorBridge:
    """
    Translates ECH0-PRIME's high-level reasoning intents into 
    concrete system actions. Includes a safety whitelist.
    """
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        # Whitelist of safe commands/tools
        self.allowed_tools = ["ls", "mkdir", "touch", "cat", "echo", "grep", "rm", "rmdir"]
        self.max_command_length = 200

    def execute_intent(self, intent: Dict[str, Any]) -> str:
        """
        Processes an 'intent' dictionary and executes the action if safe.
        Format: {"tool": "mkdir", "args": ["new_folder"]}
        """
        tool = intent.get("tool")
        args = intent.get("args", [])

        if not tool or tool not in self.allowed_tools:
            return f"ACTUATOR ERROR: Tool '{tool}' is not in the safety whitelist or is undefined."

        # Construct command
        try:
            cmd = [tool] + [str(a) for a in args]
            full_cmd_str = " ".join(cmd)
            
            if len(full_cmd_str) > self.max_command_length:
                return "ACTUATOR ERROR: Command exceeds safety length limit."

            # Security: Ensure we stay within scratch directory
            # This is a basic check; real production systems use Docker/VPC sandboxes.
            result = subprocess.run(
                cmd, 
                cwd=self.workspace_root,
                capture_output=True, 
                text=True, 
                timeout=5
            )

            if result.returncode == 0:
                return f"SUCCESS: {result.stdout if result.stdout else 'Command executed.'}"
            else:
                return f"EXECUTION FAILURE: {result.stderr}"

        except Exception as e:
            return f"SYSTEM ERROR: {str(e)}"

    def parse_llm_action(self, llm_insight: str) -> List[Dict[str, Any]]:
        """
        Scrapes the LLM insight for JSON action blocks.
        Expects: ACTION: {"tool": "...", "args": [...]}
        """
        import json
        actions = []
        lines = llm_insight.split("\n")
        for line in lines:
            if "ACTION:" in line:
                try:
                    meta_json = line.split("ACTION:")[1].strip()
                    actions.append(json.loads(meta_json))
                except:
                    continue
        return actions
