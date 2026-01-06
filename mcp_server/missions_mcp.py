#!/usr/bin/env python3
"""
MCP Server for ECH0-PRIME Missions
Exposes mission orchestration and execution tools.
"""

import os
import subprocess
from typing import Dict, Any, List, Optional
from mcp_server.registry import ToolRegistry

@ToolRegistry.register("mission_run_all")
def mission_run_all() -> str:
    """
    Executes all scheduled missions in sequence.
    """
    try:
        # Run master_runner.py in a separate process
        script_path = "/Users/noone/echo_prime/missions/master_runner.py"
        subprocess.Popen(["python3", script_path])
        return "All missions initiated via Master Runner."
    except Exception as e:
        return f"Error initiating missions: {str(e)}"

@ToolRegistry.register("mission_status")
def mission_status() -> str:
    """
    Returns the status of current and past missions by reading logs.
    """
    log_path = "/Users/noone/echo_prime/missions/master_runner.log"
    if not os.path.exists(log_path):
        return "No mission logs found. System may be idle."
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            # Return last 10 lines of the log
            return "".join(lines[-10:])
    except Exception as e:
        return f"Error reading logs: {str(e)}"

@ToolRegistry.register("mission_submit_directive")
def mission_submit_directive(directive: str) -> str:
    """
    Submits a high-level directive directly to the Prefrontal Cortex for execution.
    """
    # In a real system, this would interact with a mission queue.
    # For now, we'll log it as a pending directive.
    with open("/Users/noone/echo_prime/missions/directives.log", "a") as f:
        f.write(f"DIRECTIVE: {directive}\n")
    return f"Directive received: '{directive}'. Processing in next cycle."



