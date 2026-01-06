#!/usr/bin/env python3
import asyncio
import sys
import os
import json
from mcp.server.stdio import stdio_server
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.registry import ToolRegistry
from mcp_server.discovery import scan_local_tools

async def main():
    # Discover tools
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scan_local_tools(os.path.join(project_root, "mcp_server"))
    scan_local_tools(os.path.join(project_root, "reasoning", "tools"))
    
    registry = ToolRegistry()
    server = Server("echo-prime")

    @server.list_tools()
    async def handle_list_tools():
        schemas = registry.get_schemas()
        tools = []
        for s in schemas:
            tools.append(Tool(
                name=s["name"],
                description=s["description"],
                inputSchema=s["parameters"]
            ))
        
        # Add a special 'ask_ech0' tool for reasoning
        tools.append(Tool(
            name="ask_ech0",
            description="Ask ECH0-PRIME for high-level reasoning or to solve a complex problem.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question or task"},
                    "model": {"type": "string", "description": "LLM model to use (default: deepseek-r1:14b)"}
                },
                "required": ["query"]
            }
        ))
        return tools

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict):
        if name == "ask_ech0":
            from reasoning.orchestrator import ReasoningOrchestrator
            from memory.manager import MemoryManager
            from ech0_governance.persistent_memory import PersistentMemory
            from ech0_governance.knowledge_graph import KnowledgeGraph
            
            os.environ["ECH0_SILENT"] = "1"
            memory_manager = MemoryManager()
            gov_mem = PersistentMemory(memory_manager=memory_manager)
            kg = KnowledgeGraph()
            
            reasoner = ReasoningOrchestrator(
                use_llm=True,
                model_name=arguments.get("model", "deepseek-r1:14b"),
                governance_mem=gov_mem,
                knowledge_graph=kg
            )
            
            result = reasoner.reason_about_scenario({"source": "mcp"}, {"goal": arguments["query"]})
            return [TextContent(type="text", text=result.get("llm_insight", "No response."))]
        
        # Call regular tools
        result = registry.call_tool(name, arguments)
        return [TextContent(type="text", text=str(result))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
