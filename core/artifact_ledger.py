
import time
import uuid
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from enum import Enum

class ArtifactType(Enum):
    PLAN = "plan"
    DIFF = "diff"
    TEST_RESULT = "test_result"
    SCREENSHOT = "screenshot"
    CODE_SNAPSHOT = "code_snapshot"
    THOUGHT_TRACE = "thought_trace"

@dataclass
class Artifact:
    id: str
    type: str # stored as string for flexibility
    content: Any # JSON serializable
    correlation_id: str # Task ID or Mission ID
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self):
        return json.dumps(asdict(self))

class ArtifactLedger:
    """
    Immutable log of plans, actions, diffs, screenshots, and tests.
    Serves as the 'Memory of Work' for the agent.
    """
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            # Default to ech0_prime/artifacts
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            storage_dir = os.path.join(base, "artifacts_ledger")
        
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_file = os.path.join(self.storage_dir, "index.jsonl")

    def record(self, type: ArtifactType, content: Any, correlation_id: str = "global", metadata: Dict = None) -> str:
        """
        Records an artifact to the ledger.
        """
        artifact_id = f"art_{uuid.uuid4().hex[:8]}"
        artifact = Artifact(
            id=artifact_id,
            type=type.value,
            content=content,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )
        
        # 1. Save content to explicit file if large (like screenshots or diffs)
        # For now, we assume content is JSON-serializable string/dict.
        # If it's a binary path (screenshot), we store the path in content.
        
        # 2. Append to index
        with open(self.index_file, "a") as f:
            f.write(artifact.to_json() + "\n")
            
        return artifact_id

    def get_by_correlation_id(self, correlation_id: str) -> List[Artifact]:
        """
        Retrieves all artifacts for a specific task/mission.
        """
        artifacts = []
        if not os.path.exists(self.index_file):
            return []
            
        with open(self.index_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data["correlation_id"] == correlation_id:
                        artifacts.append(Artifact(**data))
                except:
                    continue
        return artifacts

    def get_recent(self, n: int = 10) -> List[Artifact]:
        """Get most recent N artifacts."""
        if not os.path.exists(self.index_file):
            return []
            
        lines = []
        with open(self.index_file, "r") as f:
            # Efficiently read last N lines would be better, but standard readlines is fine for now
            lines = f.readlines()
            
        recent_lines = lines[-n:]
        artifacts = []
        for line in recent_lines:
             try:
                data = json.loads(line)
                artifacts.append(Artifact(**data))
             except:
                pass
        return artifacts
