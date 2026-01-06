import sys
import os

# Add project root to path (parent of scripts/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"Sys Path: {sys.path}")

try:
    import agents
    print(f"Agents package found at: {agents.__file__}")
    
    from agents import multi_agent
    print(f"MultiAgent module found: {multi_agent}")
    
    from agents.multi_agent import MultiAgentSystem
    print(f"MultiAgentSystem class found: {MultiAgentSystem}")
    print("✅ SUCCESS")
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
except Exception as e:
    print(f"❌ UNEXPECTED ERROR: {e}")
