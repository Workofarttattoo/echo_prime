import inspect
import json
from typing import Dict, Any, Callable, List

class ToolRegistry:
    """
    Singleton registry for cognitive tools in the ECH0-PRIME architecture.
    """
    _instance = None
    _tools: Dict[str, Dict[str, Any]] = {}
    _embeddings: Dict[str, Any] = {}
    _embedder = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
            cls._initialize_embedder()
        return cls._instance

    @classmethod
    def _initialize_embedder(cls):
        """Initializes the sentence transformer for semantic search."""
        try:
            from sentence_transformers import SentenceTransformer
            import os
            model_id = os.getenv('ECH0_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            cls._embedder = SentenceTransformer(model_id, use_auth_token=False)
            print(f"✅ ToolRegistry RAG initialized with {model_id}")
        except Exception as e:
            print(f"⚠️ ToolRegistry RAG failed to initialize: {e}. Using keyword fallback.")
            cls._embedder = None

    @classmethod
    def register(cls, name: str = None):
        """Decorator to register a function as a tool."""
        def decorator(func: Callable):
            tool_name = name or func.__name__
            sig = inspect.signature(func)
            
            # Simple docstring parser for description
            doc = func.__doc__ or "No description provided."
            
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self': continue
                
                # Basic type mapping
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

            description = doc.strip()
            cls._tools[tool_name] = {
                "name": tool_name,
                "description": description,
                "parameters": parameters,
                "func": func
            }
            
            # Update embedding cache if embedder is available
            if cls._embedder:
                try:
                    cls._embeddings[tool_name] = cls._embedder.encode(description)
                except Exception:
                    pass

            return func
        return decorator

    @classmethod
    def get_schemas(cls) -> List[Dict[str, Any]]:
        """Returns JSON schemas for all registered tools (OpenAI Format)."""
        schemas = []
        for tool in cls._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })
        return schemas

    @classmethod
    def get_relevant_schemas(cls, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Returns JSON schemas for tools relevant to the query.
        Uses RAG (Semantic Search) if available, otherwise keyword-based ranking.
        """
        if cls._embedder and cls._embeddings:
            try:
                import numpy as np
                query_embedding = cls._embedder.encode(query)
                
                scored_tools = []
                for name, embedding in cls._embeddings.items():
                    # Cosine similarity
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-9
                    )
                    scored_tools.append((similarity, cls._tools[name]))
                
                scored_tools.sort(key=lambda x: x[0], reverse=True)
                return [{
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"]
                    }
                } for score, t in scored_tools[:top_k]]
            except Exception as e:
                print(f"⚠️ Semantic search failed: {e}. Falling back to keywords.")

        # Fallback to keyword-based search
        query_words = set(query.lower().split())
        scored_tools = []
        
        for name, tool in cls._tools.items():
            score = 0
            if any(word in name.lower() for word in query_words):
                score += 5
            desc_words = tool["description"].lower().split()
            score += sum(1 for word in query_words if word in desc_words)
            
            if score > 0:
                scored_tools.append((score, tool))
        
        scored_tools.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_tools:
            all_tools = list(cls._tools.values())
            return [{
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"]
                }
            } for t in all_tools[:top_k]]

        return [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            }
        } for score, t in scored_tools[:top_k]]

    @classmethod
    def call_tool(cls, name: str, args: Dict[str, Any]) -> Any:
        """Invokes a registered tool by name."""
        if name not in cls._tools:
            return f"Error: Tool '{name}' not found."
        
        try:
            return cls._tools[name]["func"](**args)
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"
