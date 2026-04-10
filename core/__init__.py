"""
ECH0-PRIME Core Cognitive Engine
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Core modules for hierarchical generative models, attention mechanisms, and sensory bridges.
"""

from .engine import HierarchicalGenerativeModel, FreeEnergyEngine, GlobalWorkspace
from .attention import QuantumAttentionHead, CoherenceShaper

__all__ = [
    'HierarchicalGenerativeModel',
    'FreeEnergyEngine',
    'GlobalWorkspace',
    'QuantumAttentionHead',
    'CoherenceShaper',
]

try:
    from .vision_bridge import VisionBridge
    __all__.append('VisionBridge')
except ImportError:
    pass

try:
    from .audio_bridge import AudioBridge
    __all__.append('AudioBridge')
except ImportError:
    pass

try:
    from .voice_bridge import VoiceBridge
    __all__.append('VoiceBridge')
except ImportError:
    pass

try:
    from .actuator import ActuatorBridge
    __all__.append('ActuatorBridge')
except ImportError:
    pass
