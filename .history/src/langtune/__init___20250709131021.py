"""
Langtune: Efficient LoRA Fine-Tuning for Text LLMs

This package provides tools and modules for efficient fine-tuning of large language models (LLMs) on text data using Low-Rank Adaptation (LoRA).
"""

from .models import (
    LoRALanguageModel, RLHF, CoT, CCoT, GRPO, RLVR, DPO, PPO, LIME, SHAP
)
from .config import default_config
from .utils import encode_text

__all__ = [
    "LoRALanguageModel", "RLHF", "CoT", "CCoT", "GRPO", "RLVR", "DPO", "PPO", "LIME", "SHAP",
    "default_config", "encode_text"
]
