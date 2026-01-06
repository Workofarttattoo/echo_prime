#!/usr/bin/env python3
"""
MCP Server for AIOS (AI Operating System)
Exposes AIOS kernel and algorithm management tools to ECH0-PRIME.
"""

import asyncio
from typing import Dict, Any, List, Optional
from mcp_server.registry import ToolRegistry
from infrastructure.aios_ech0_integration import AIOS_ECH0_IntegratedSystem

# Initialize the integrated system as a singleton for the MCP server
_integrated_system = None

async def get_system():
    global _integrated_system
    if _integrated_system is None:
        _integrated_system = AIOS_ECH0_IntegratedSystem()
        await _integrated_system.initialize_integration()
    return _integrated_system

def _run_async(coro):
    """Helper to run a coroutine regardless of whether an event loop is already running."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in a loop, we can't use run_until_complete.
            # However, for MCP tools which are typically called synchronously by the orchestrator,
            # this is a common issue. In a real production system, the tools themselves would be async.
            # For now, we'll try to use a separate thread if needed, or just warn.
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

@ToolRegistry.register("aios_get_status")
def aios_get_status() -> Dict[str, Any]:
    """
    Returns the current status of the AIOS kernel and integrated algorithms.
    """
    system = _run_async(get_system())
    return system.get_system_status()

@ToolRegistry.register("aios_submit_task")
def aios_submit_task(name: str, priority: str = "MEDIUM", domain: str = "general") -> str:
    """
    Submits a new task to the AIOS kernel.
    Returns the task ID.
    """
    from infrastructure.aios_algorithms import create_ai_task, TaskPriority
    
    priority_map = {
        "LOW": TaskPriority.LOW,
        "MEDIUM": TaskPriority.MEDIUM,
        "HIGH": TaskPriority.HIGH,
        "CRITICAL": TaskPriority.CRITICAL
    }
    
    system = _run_async(get_system())
    
    task = create_ai_task(
        task_id=f"manual_{name.replace(' ', '_')}",
        name=name,
        priority=priority_map.get(priority.upper(), TaskPriority.MEDIUM),
        resource_reqs={ "CPU": 1.0, "MEMORY": 0.5 },
        duration=1.0
    )
    
    task_id = _run_async(system.aios_kernel.submit_task(task))
    return task_id

@ToolRegistry.register("aios_execute_cycle")
def aios_execute_cycle() -> List[Dict[str, Any]]:
    """
    Executes one cycle of the AIOS kernel, processing pending tasks.
    Returns a list of completed tasks.
    """
    system = _run_async(get_system())
    completed = _run_async(system.aios_kernel.execute_task_cycle())
    return [
        {"id": t.id, "name": t.name, "status": t.status}
        for t in completed
    ]

@ToolRegistry.register("aios_optimize_algorithms")
def aios_optimize_algorithms(task_description: str) -> List[str]:
    """
    Analyzes a task description and returns the most suitable algorithms from the ECH0 library.
    """
    system = _run_async(get_system())
    return _run_async(system.optimize_algorithm_selection(task_description))

@ToolRegistry.register("aios_run_pipeline")
def aios_run_pipeline(algorithms: List[str]) -> Dict[str, Any]:
    """
    Runs a sequence of algorithms as an integrated pipeline in AIOS.
    """
    system = _run_async(get_system())
    return _run_async(system.execute_algorithm_pipeline(algorithms))

