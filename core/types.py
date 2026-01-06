from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import time

@dataclass
class ToolResult:
    """
    Standardized result from a tool execution.
    Mirrors OpenAI's tool role message.
    """
    tool_call_id: str
    tool_name: str
    content: str
    status: str = "success" # success, error
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.tool_name,
            "content": self.content,
            "status": self.status,
            "timestamp": self.timestamp
        }

@dataclass
class AgentMessage:
    """
    Standardized message in the agent's conversation history.
    """
    role: str # system, user, assistant, tool
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None # If role is tool

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "role": self.role,
            "content": self.content
        }
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d
