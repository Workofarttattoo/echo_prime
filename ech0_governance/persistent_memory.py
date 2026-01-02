import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from memory.manager import MemoryManager
from mcp_server.registry import ToolRegistry

class PersistentMemory:
    """
    Governance wrapper around ECH0's core MemoryManager.
    Provides explicit tool-use interfaces for the LLM and
    state export for the dashboard.
    """
    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager
        self.recent_notes = []

    @ToolRegistry.register(name="store_memory")
    def store(self, note: str) -> str:
        """
        Explicitly stores a text note into episodic memory.
        Returns a confirmation string.
        """
        # Create a dummy embedding for the note (in a real system, use an embedding model)
        # For MVP, we'll use a random vector or a simple hash-based vector if possible,
        # but since the core memory uses numpy vectors, we need something compatible.
        
        # We will use a random vector for now to satisfy the interface, 
        # as we don't have a live embedding model in this class yet.
        # In a full implementation, `orchestrator` would pass the embedding.
        vector = np.random.randn(self.memory.working.chunk_dim) 
        
        # Store in episodic memory
        self.memory.episodic.store_episode(vector)
        
        # Also store in a parallel "text-based" list for retrieval since 
        # the core memory MVP is fully vector-based and doesn't store text payload.
        # We'll extend the functionality here to keep text.
        self.recent_notes.append({"text": note, "timestamp": "now"})
        if len(self.recent_notes) > 50:
            self.recent_notes.pop(0)
            
        return f"Stored note: {note}"

    @ToolRegistry.register(name="search_memory")
    def search(self, query: str) -> str:
        """
        Retrieves relevant notes based on the query.
        For MVP, performs a simple keyword match on recent notes.
        """
        results = [
            n["text"] for n in self.recent_notes 
            if any(term in n["text"].lower() for term in query.lower().split())
        ]
        
        if not results:
            return "No relevant notes found."
        
        return "Found notes:\n" + "\n".join(f"- {r}" for r in results)

    def get_dashboard_state(self) -> List[Dict[str, Any]]:
        """Returns the last 5 notes for the GUI."""
        return self.recent_notes[-5:]
