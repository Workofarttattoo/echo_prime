import json
from typing import List, Dict, Any, Optional
from mcp_server.registry import ToolRegistry

class KnowledgeGraph:
    """
    Structured memory store for Enitity-Relation triples.
    Tools: `lookup`, `add_relation`.
    """
    def __init__(self):
        self.triples = [] # List of {"head": str, "relation": str, "tail": str}
        self.entities = set()

    @ToolRegistry.register(name="add_fact")
    def add_relation(self, head: str, relation: str, tail: str) -> str:
        """Adds a fact to the graph."""
        triple = {"head": head, "relation": relation, "tail": tail}
        if triple not in self.triples:
            self.triples.append(triple)
            self.entities.add(head)
            self.entities.add(tail)
            return f"Added: ({head}) -[{relation}]-> ({tail})"
        return f"Fact already exists: ({head}) -[{relation}]-> ({tail})"

    @ToolRegistry.register(name="lookup_fact")
    def lookup(self, entity: str) -> str:
        """Finds all relations involving the entity."""
        matches = []
        for t in self.triples:
            if t["head"].lower() == entity.lower():
                matches.append(f"-> {t['relation']} -> {t['tail']}")
            elif t["tail"].lower() == entity.lower():
                matches.append(f"<- {t['relation']} <- {t['head']}")
        
        if not matches:
            return f"No known relations for '{entity}'."
        
        return f"Knowledge regarding '{entity}':\n" + "\n".join(matches)

    def get_dashboard_state(self) -> Dict[str, Any]:
        """Returns graph data for visualization."""
        return {
            "nodes": list(self.entities),
            "links": self.triples
        }
