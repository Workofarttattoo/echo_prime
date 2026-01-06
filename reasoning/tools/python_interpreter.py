
"""
Python Code Interpreter Tool.
Allows Echo_Prime to execute Python code safely for calculations, data analysis, and logic verification.
"""

from typing import Dict, Any, Optional
from mcp_server.registry import ToolRegistry
from infrastructure.code_sandbox import CodeSandbox


# Global instance
_sandbox = CodeSandbox(timeout=10.0)

@ToolRegistry.register(name="execute_python")
def execute_python_code(code: str) -> str:

    """
    Executes arbitrary Python code and returns the standard output.
    Useful for:
    - Calculations, math, and number crunching.
    - Data analysis and processing.
    - Verifying logic or algorithms.
    - Running simulations.
    
    The code must print the final result to stdout.
    State is NOT preserved between calls (stateless execution).
    
    Args:
        code: Valid Python code string.
        
    Returns:
        The standard output of the executed code, or error message.
    """
    # Use the global sandbox instance
    result = _sandbox.execute(code, inputs=[])
    
    if result["success"]:
        # Combine all outputs from the (single) execution
        outputs = [r["output"] for r in result["results"]]
        return "\n".join(outputs) if outputs else "No output produced."
    else:
        return f"Execution Error: {result['error']}"
