"""
ECH0-PRIME Core Cognitive Engine
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Core modules for hierarchical generative models, attention mechanisms, and sensory bridges.
"""

from .engine import HierarchicalGenerativeModel, FreeEnergyEngine, GlobalWorkspace
from .attention import QuantumAttentionHead, CoherenceShaper
from .vision_bridge import VisionBridge
from .audio_bridge import AudioBridge
from .voice_bridge import VoiceBridge
from .actuator import ActuatorBridge

__all__ = [
    'HierarchicalGenerativeModel',
    'FreeEnergyEngine',
    'GlobalWorkspace',
    'QuantumAttentionHead',
    'CoherenceShaper',
    'VisionBridge',
    'AudioBridge',
    'VoiceBridge',
    'ActuatorBridge'
]
