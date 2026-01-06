
from mcp_server.registry import ToolRegistry
from reasoning.task_manager import TaskManager, TaskStatus

@ToolRegistry.register()
def create_initial_plan(goal: str):
    """
    Creates an initial plan for a high-level goal using standard decomposition.
    Overwrites any existing plan.
    """
    tm = TaskManager()
    tm.generate_initial_plan(goal)
    return tm.get_plan_view()

@ToolRegistry.register()
def add_task(description: str, parent_id: str = None, depends_on: list = None):
    """
    Adds a new task to the plan.
    Args:
        description: Description of the task.
        parent_id: Optional ID of the parent task.
        depends_on: Optional list of task IDs this task depends on.
    """
    tm = TaskManager()
    tm.load_latest_plan()
    new_id = tm.add_task(description, parent_id, depends_on)
    return f"Task added with ID: {new_id}"

@ToolRegistry.register()
def update_task_status(task_id: str, status: str, result: str = None):
    """
    Updates the status of a task.
    Args:
        task_id: The ID of the task.
        status: The new status ('pending', 'in_progress', 'completed', 'failed', 'blocked').
        result: Optional result description.
    """
    tm = TaskManager()
    tm.load_latest_plan()
    
    try:
        new_status = TaskStatus(status.lower())
    except ValueError:
        return f"Invalid status: {status}. Must be one of {[e.value for e in TaskStatus]}"
        
    try:
        tm.update_task_status(task_id, new_status, result)
        return f"Task {task_id} updated to {status}"
    except Exception as e:
        return f"Error updating task: {e}"

@ToolRegistry.register()
def get_current_plan():
    """Retrieves the current plan status."""
    tm = TaskManager()
    if tm.load_latest_plan():
        return tm.get_plan_view()
    else:
        return "No active plan found."
