#!/usr/bin/env python3
"""
MCP Server for ECH0-PRIME Research Assistant
Exposes scholarly research tools (arXiv, Patent search, etc.).
"""

import asyncio
from typing import Dict, Any, List, Optional
from mcp_server.registry import ToolRegistry
from core.research_assistant import ResearchAssistant

# Initialize ResearchAssistant
research_assistant = ResearchAssistant()

@ToolRegistry.register("research_search_arxiv")
def research_search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search arXiv for research papers matching the query.
    Returns a list of papers with title, summary, authors, and PDF URL.
    """
    try:
        return research_assistant.search_arxiv(query, max_results=max_results)
    except Exception as e:
        return [{"error": str(e)}]

@ToolRegistry.register("research_search_patents")
def research_search_patents(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for patents related to technical engineering questions.
    """
    try:
        return research_assistant.search_patents(query, max_results=max_results)
    except Exception as e:
        return [{"error": str(e)}]

@ToolRegistry.register("research_summarize_paper")
def research_summarize_paper(paper_id: str, summary: str) -> str:
    """
    Produces a scholarly summary and technical synthesis of a research paper.
    """
    try:
        return research_assistant.summarize_paper(paper_id, summary)
    except Exception as e:
        return f"Error: {str(e)}"

@ToolRegistry.register("research_fetch_pdf")
def research_fetch_pdf(pdf_url: str, filename: str) -> str:
    """
    Downloads a research paper PDF for local analysis.
    """
    try:
        return research_assistant.fetch_paper_pdf(pdf_url, filename)
    except Exception as e:
        return f"Error: {str(e)}"
