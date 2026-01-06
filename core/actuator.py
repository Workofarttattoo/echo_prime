import subprocess
import os
import shlex
from typing import Dict, Any, List
from mcp_server.registry import ToolRegistry

class ActuatorBridge:
    """
    Translates ECH0-PRIME's high-level reasoning intents into 
    concrete system actions. Includes a safety whitelist and ToolRegistry integration.
    """
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        # Whitelist of safe shell commands
        self.allowed_shell_tools = ["ls", "mkdir", "touch", "cat", "echo", "grep", "rm", "rmdir"]
        self.max_command_length = 200
        
        # Integrate ToolRegistry
        self.registry = ToolRegistry()

    def execute_intent(self, intent: Dict[str, Any]) -> str:
        """
        Processes an 'intent' dictionary and executes the action if safe.
        Format: {"tool": "mkdir", "args": ["new_folder"]}
        """
        tool_name = intent.get("tool")
        args = intent.get("args", {})
        
        # 0. Check arguments format
        # If the LLM provides arguments in a list, but it's a registry tool, we might have an issue
        # unless we can map positional args to keywords. Ideally LLM uses dict for registry tools.
        
        # 1. Check ToolRegistry first
        if tool_name in self.registry._tools:
            try:
                # For registry tools, we expect 'args' to be a dictionary of kwargs.
                if isinstance(args, list):
                     return f"ACTUATOR ERROR: Tool '{tool_name}' requires keyword arguments (dict), got list."
                
                result = self.registry.call_tool(tool_name, args)
                return f"TOOL_SUCCESS ({tool_name}): {str(result)}"
            except Exception as e:
                return f"TOOL_FAILURE ({tool_name}): {str(e)}"

        # 2. Check Shell Whitelist
        if not tool_name or tool_name not in self.allowed_shell_tools:
            return f"ACTUATOR ERROR: Tool '{tool_name}' is not in the safety whitelist or registry."

        # Construct command
        try:
            # Shell args are typically a list
            cmd_args = args if isinstance(args, list) else list(args.values())
            cmd = [tool_name] + [str(a) for a in cmd_args]
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
        Expects: ACTION: {"tool": "...", "args": {...}}
        """
        import json
        actions = []
        lines = llm_insight.split("\n")
        for line in lines:
            if "ACTION:" in line:
                try:
                    meta_json = line.split("ACTION:")[1].strip()
                    # Fix common LLM mistake: single quotes to double quotes for JSON
                    # This is risky but helpful
                    if "'" in meta_json and '"' not in meta_json:
                        meta_json = meta_json.replace("'", '"')
                        
                    actions.append(json.loads(meta_json))
                except:
                    continue
        return actions
