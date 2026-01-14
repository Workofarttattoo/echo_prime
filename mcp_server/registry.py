import inspect
import json
from typing import Dict, Any, Callable, List

class ToolRegistry:
    """
    Singleton registry for cognitive tools in the ECH0-PRIME architecture.
    """
    _instance = None
    _tools: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str = None):
        """Decorator to register a function as a tool."""
        def decorator(func: Callable):
            tool_name = name or func.__name__
            cls.register_tool(tool_name, func)
            return func
        return decorator

    @classmethod
    def register_tool(cls, name: str, func: Callable, description: str = None):
        """Manually register a tool (useful for bound methods)."""
        sig = inspect.signature(func)
        doc = description or func.__doc__ or "No description provided."
        
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self': continue
            
            p_type = "string"
            if param.annotation == int: p_type = "integer"
            elif param.annotation == float: p_type = "number"
            elif param.annotation == bool: p_type = "boolean"
            elif param.annotation == dict: p_type = "object"
            elif param.annotation == list: p_type = "array"
            
            parameters["properties"][param_name] = {
                "type": p_type,
                "description": f"Argument {param_name}"
            }
            
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(param_name)

        cls._tools[name] = {
            "name": name,
            "description": doc.strip(),
            "parameters": parameters,
            "func": func
        }

    @classmethod
    def get_schemas(cls) -> List[Dict[str, Any]]:
        """Returns JSON schemas for all registered tools."""
        schemas = []
        for tool in cls._tools.values():
            schemas.append({
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            })
        return schemas

    @classmethod
    def call_tool(cls, name: str, args: Dict[str, Any]) -> Any:
        """Invokes a registered tool by name."""
        if name not in cls._tools:
            return f"Error: Tool '{name}' not found."
        
        try:
            return cls._tools[name]["func"](**args)
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"
