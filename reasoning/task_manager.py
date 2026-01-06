
import json
import uuid
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass, field, asdict
from core.artifact_ledger import ArtifactLedger, ArtifactType
from reasoning.planner import HTNPlanner, Task as HTNTask

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class Task:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    subtasks: List['Task'] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list) # List of Task IDs
    result: Optional[str] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "subtasks": [t.to_dict() for t in self.subtasks],
            "dependencies": self.dependencies,
            "result": self.result
        }

    @staticmethod
    def from_dict(data: Dict):
        task = Task(
            id=data["id"],
            description=data["description"],
            status=TaskStatus(data["status"]),
            dependencies=data.get("dependencies", []),
            result=data.get("result")
        )
        task.subtasks = [Task.from_dict(st) for st in data.get("subtasks", [])]
        return task

class TaskManager:
    def __init__(self, ledger: ArtifactLedger = None):
        if ledger is None:
            self.ledger = ArtifactLedger()
        else:
            self.ledger = ledger
            
        self.root_tasks: List[Task] = []
        self.current_plan_id = str(uuid.uuid4())[:8]

    def load_latest_plan(self):
        """Loads the most recent plan from the ledger."""
        # Get recent artifacts and find the last one of type PLAN
        artifacts = self.ledger.get_recent(n=50) # Look back 50 artifacts
        for art in reversed(artifacts):
            if art.type == ArtifactType.PLAN.value:
                try:
                    # art.content is a list of task dicts
                    self.root_tasks = [Task.from_dict(t) for t in art.content]
                    self.current_plan_id = art.correlation_id
                    return True
                except Exception as e:
                    print(f"Failed to load plan artifact {art.id}: {e}")
        return False

    def add_task(self, description: str, parent_id: Optional[str] = None, depends_on: List[str] = None) -> str:
        """Adds a new task to the list."""
        task_id = f"task_{uuid.uuid4().hex[:6]}"
        new_task = Task(id=task_id, description=description, dependencies=depends_on or [])
        
        if parent_id:
            parent = self._find_task(parent_id)
            if parent:
                parent.subtasks.append(new_task)
            else:
                raise ValueError(f"Parent task {parent_id} not found")
        else:
            self.root_tasks.append(new_task)
            
        self._record_plan_update("Task Added")
        return task_id

    def update_task_status(self, task_id: str, status: TaskStatus, result: str = None):
        """Updates the status of a task."""
        task = self._find_task(task_id)
        if task:
            task.status = status
            if result:
                task.result = result
            self._record_plan_update(f"Task {task_id} Update: {status.value}")
        else:
            raise ValueError(f"Task {task_id} not found")

    def get_plan_view(self) -> str:
        """Returns a formatted string of the current plan."""
        output = ["--- CURRENT PLAN ---"]
        for task in self.root_tasks:
            output.extend(self._format_task_tree(task))
        return "\n".join(output)

    def generate_initial_plan(self, goal: str):
        """Uses HTN Planner to generate an initial breakdown."""
        # This connects to the DeepMind-style planning module
        planner = HTNPlanner()
        # In a real implementation, we would need a rich domain knowledge base here.
        # For now, we will add a seed task.
        self.root_tasks = []
        self.add_task(f"Objective: {goal}")
        
    def _find_task(self, task_id: str, tasks: List[Task] = None) -> Optional[Task]:
        if tasks is None:
            tasks = self.root_tasks
        
        for task in tasks:
            if task.id == task_id:
                return task
            found = self._find_task(task_id, task.subtasks)
            if found:
                return found
        return None

    def _format_task_tree(self, task: Task, indent: int = 0) -> List[str]:
        lines = []
        status_icon = {
            TaskStatus.PENDING: "[ ]",
            TaskStatus.IN_PROGRESS: "[COMBINING_DOTS]", # Using text for now
            TaskStatus.COMPLETED: "[x]",
            TaskStatus.FAILED: "[!]",
            TaskStatus.BLOCKED: "[#]"
        }
        icon = status_icon.get(task.status, "[?]")
        prefix = "  " * indent
        lines.append(f"{prefix}{icon} {task.description} ({task.id})")
        
        for subtask in task.subtasks:
            lines.extend(self._format_task_tree(subtask, indent + 1))
            
        return lines

    def _record_plan_update(self, change_desc: str):
        """Saves the current plan state to the ledger."""
        plan_state = [t.to_dict() for t in self.root_tasks]
        self.ledger.record(
            ArtifactType.PLAN,
            content=plan_state,
            correlation_id=self.current_plan_id,
            metadata={"change": change_desc}
        )
