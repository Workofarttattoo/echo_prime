import os
import sys
import difflib
from mcp_server.registry import ToolRegistry

@ToolRegistry.register()
def audit_source(project_root: str = ".") -> str:
    """
    Scans the ECH0-PRIME codebase for architectural bottlenecks or messy logic.
    Returns a summary of the file structure and potential areas for evolution.
    """
    summary = ["--- CORTEX AUDIT SUMMARY ---"]
    for root, dirs, files in os.walk(project_root):
        if "venv" in root or ".git" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                size = os.path.getsize(path)
                summary.append(f"MODULE: {path} ({size} bytes)")
    return "\n".join(summary)

@ToolRegistry.register()
def propose_evolution(file_path: str, improvement_goal: str) -> str:
    \"\"\"
    ECH0 analyzes a specific file and proposes a 'Level 10' refinement.
    This generates a diff for the user to approve.
    \"\"\"
    if not os.path.exists(file_path):
        return f"ERROR: File {file_path} not found."
    
    with open(file_path, "r") as f:
        content = f.read()
    
    return f"ANALYSIS COMPLETE for {file_path}. Goal: {improvement_goal}. Please provide the intended changes in a standard patch format."

PROTECTED_FILES = [
    "reasoning/orchestrator.py",
    "reasoning/IDENTITY.txt",
    "reasoning/tools/cortex_evolution.py"
]

@ToolRegistry.register()
def apply_evolution(file_path: str, new_code: str) -> str:
    \"\"\"
    Applies an approved code refinement to the ECH0-PRIME core.
    The system prompt and identity modules are LOCKED and cannot be edited by this tool.
    \"\"\"
    # Normalize path for check
    clean_path = os.path.relpath(file_path, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    for protected in PROTECTED_FILES:
        if protected in file_path:
            return f"PERMISSION DENIED: {file_path} is a PROTECTED CORE MODULE. Identity/Prompt edits must be performed manually by Joshua."

    try:
        # Save a backup
        backup_path = file_path + ".bak"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                old_code = f.read()
            
            with open(backup_path, "w") as f:
                f.write(old_code)
            
        with open(file_path, "w") as f:
            f.write(new_code)
            
        return f"SUCCESS: {file_path} has been evolved. Backup created at {backup_path}."
    except Exception as e:
        return f"EVOLUTION FAILED: {str(e)}"
