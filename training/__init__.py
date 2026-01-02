"""
ECH0-PRIME Training Pipeline
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Training and optimization pipelines.
"""

from .pipeline import TrainingPipeline

__all__ = ['TrainingPipeline', 'TrainingReadinessAnalyzer', 'ECH0TrainingRegimen']


def __getattr__(name):
    if name in {'TrainingReadinessAnalyzer', 'ECH0TrainingRegimen'}:
        from .regimen import TrainingReadinessAnalyzer, ECH0TrainingRegimen

        globals()['TrainingReadinessAnalyzer'] = TrainingReadinessAnalyzer
        globals()['ECH0TrainingRegimen'] = ECH0TrainingRegimen
        return globals()[name]
    raise AttributeError(f"module 'training' has no attribute {name}")
