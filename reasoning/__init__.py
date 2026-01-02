"""
ECH0-PRIME Reasoning Systems
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

High-level reasoning orchestration and LLM integration.
"""

from .orchestrator import ReasoningOrchestrator
from .llm_bridge import OllamaBridge

__all__ = ['ReasoningOrchestrator', 'OllamaBridge']
