"""
models.py: LoRA-enabled transformer models for Langtune.

This module now re-exports components from `langtune.nn` for backward compatibility.
"""

from .nn.layers import LoRALinear, MultiHeadAttention
from .nn.transformer import TransformerBlock, LoRALanguageModel
from .nn.fast_transformer import FastMultiHeadAttention, FastTransformerBlock, FastLoRALanguageModel

__all__ = [
    "LoRALinear",
    "MultiHeadAttention",
    "TransformerBlock",
    "LoRALanguageModel",
    "FastMultiHeadAttention",
    "FastTransformerBlock",
    "FastLoRALanguageModel",
]