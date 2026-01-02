import os
import importlib.util
import sys
from mcp_server.registry import ToolRegistry

def scan_local_tools(directory: str):
    """
    Dynamically scans a directory for Python files and imports them to trigger 
    tool registration via decorators.
    """
    if not os.path.exists(directory):
        print(f"Discovery: Directory {directory} does not exist.")
        return

    # Add directory to sys.path if not present
    if directory not in sys.path:
        sys.path.append(directory)

    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            module_path = os.path.join(directory, filename)
            
            try:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"Discovery: Successfully loaded tools from {filename}")
            except Exception as e:
                print(f"Discovery: Error loading {filename}: {e}")
