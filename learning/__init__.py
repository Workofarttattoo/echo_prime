"""
ECH0-PRIME Learning Systems
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Meta-learning, adaptive learning systems, and ECH0 training data integration.
"""

# Import training data integration (always available)
from .training_data_integration import (
    ECH0TrainingDataManager,
    ECH0TrainingOrchestrator,
    initialize_ech0_training_integration,
    create_training_export
)

# Try to import advanced learning systems (require torch/numpy)
try:
    from .meta import CSALearningSystem
    _advanced_learning_available = True
except ImportError:
    _advanced_learning_available = False
    CSALearningSystem = None

__all__ = [
    'ECH0TrainingDataManager',
    'ECH0TrainingOrchestrator',
    'initialize_ech0_training_integration',
    'create_training_export'
]

if _advanced_learning_available:
    __all__.append('CSALearningSystem')
