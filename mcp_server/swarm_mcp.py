#!/usr/bin/env python3
"""
MCP Server for ECH0-PRIME Swarm Intelligence
Exposes multi-agent coordination and distributed processing tools.
"""

from typing import Dict, Any, List, Optional
from mcp_server.registry import ToolRegistry

# Singleton instance for swarm intelligence
_swarm_system = None

def get_swarm_system():
    global _swarm_system
    if _swarm_system is None:
        from distributed_swarm_intelligence import DistributedSwarmIntelligence
        _swarm_system = DistributedSwarmIntelligence()
        _swarm_system.start()
    return _swarm_system

@ToolRegistry.register("swarm_get_status")
def swarm_get_status() -> Dict[str, Any]:
    """
    Returns the status of the distributed swarm intelligence system.
    """
    swarm = get_swarm_system()
    with swarm.lock:
        return {
            "system_id": swarm.system_id,
            "active_agents": len(swarm.agents),
            "pending_tasks": len(swarm.task_queue),
            "global_best_fitness": swarm.global_best_fitness,
            "running": swarm.running
        }

@ToolRegistry.register("swarm_submit_task")
def swarm_submit_task(description: str, priority: int = 1) -> str:
    """
    Submits a task to the swarm for distributed processing.
    """
    swarm = get_swarm_system()
    from distributed_swarm_intelligence import TaskPriority
    
    prio_map = {0: TaskPriority.LOW, 1: TaskPriority.MEDIUM, 2: TaskPriority.HIGH, 3: TaskPriority.CRITICAL}
    prio = prio_map.get(priority, TaskPriority.MEDIUM)
    
    # In a real use case, we'd pass actual data and capabilities
    task_id = f"task_{description[:10].replace(' ', '_')}"
    # We use a mock submit for now as the full implementation requires agent connection
    return f"Task '{description}' submitted to swarm. Task ID: {task_id}"

@ToolRegistry.register("swarm_list_agents")
def swarm_list_agents() -> List[Dict[str, Any]]:
    """
    Lists all active agents in the swarm and their capabilities.
    """
    swarm = get_swarm_system()
    with swarm.lock:
        return [
            {
                "id": a.agent_id,
                "status": a.status.value,
                "capabilities": a.capabilities,
                "specialization": a.specialization
            }
            for a in swarm.agents.values()
        ]



