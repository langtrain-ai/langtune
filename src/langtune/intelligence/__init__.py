"""
Langtune Training Intelligence Engine (TIE)

This module provides intelligent training optimization including:
- Dataset Intelligence: Auto-curation, deduplication, quality scoring
- Adaptive LoRA: Dynamic config based on gradient probing
- Curriculum Learning: Staged sample ordering
- Reinforcement Learning: RLHF, DPO, PPO implementations
- Model-Aware Optimization: Architecture-specific tuning
"""

from .dataset_intelligence import DatasetIntelligence, DatasetQualityReport
from .adaptive_lora import AdaptiveLoRAConfig, probe_model_for_lora
from .curriculum import CurriculumEngine, DifficultyClassifier
from .reinforcement import RLHFTrainer, DPOTrainer, PPOTrainer
from .model_aware import ModelAnalyzer, AutoConfigurator
from .langtrain_score import LangtrainScore, compute_score

__all__ = [
    # Dataset Intelligence
    "DatasetIntelligence",
    "DatasetQualityReport",
    # Adaptive LoRA
    "AdaptiveLoRAConfig",
    "probe_model_for_lora",
    # Curriculum Learning
    "CurriculumEngine",
    "DifficultyClassifier",
    # Reinforcement Learning
    "RLHFTrainer",
    "DPOTrainer",
    "PPOTrainer",
    # Model-Aware
    "ModelAnalyzer",
    "AutoConfigurator",
    # Langtrain Score
    "LangtrainScore",
    "compute_score",
]
