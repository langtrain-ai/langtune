"""
Langtune: Efficient LoRA Fine-Tuning for Text LLMs

This package provides tools and modules for efficient fine-tuning of large language models (LLMs) on text data using Low-Rank Adaptation (LoRA).
"""

# Core models
from .models import (
    LoRALanguageModel, LoRALinear, MultiHeadAttention, TransformerBlock,
    RLHF, CoT, CCoT, GRPO, RLVR, DPO, PPO, LIME, SHAP
)

# Configuration
from .config import (
    Config, ModelConfig, TrainingConfig, DataConfig, LoRAConfig,
    default_config, load_config, save_config, get_preset_config, validate_config
)

# Data handling
from .data import (
    TextDataset, LanguageModelingDataset, DataCollator,
    load_text_file, load_json_file, create_data_loader, split_dataset,
    SimpleTokenizer, create_sample_dataset, load_dataset_from_config
)

# Training
from .trainer import (
    Trainer, EarlyStopping, MetricsTracker, ModelCheckpoint, create_trainer
)

# Utilities
from .utils import (
    set_seed, get_device, count_parameters, count_lora_parameters,
    encode_text, decode_tokens, SimpleTokenizer, create_attention_mask,
    pad_sequences, truncate_sequences, compute_perplexity, compute_bleu_score,
    format_time, format_size, get_model_size, print_model_summary,
    save_model_info, load_model_info, log_gpu_memory, cleanup_gpu_memory
)

# CLI
from .cli import main

__version__ = "0.1.1"

__all__ = [
    # Models
    "LoRALanguageModel", "LoRALinear", "MultiHeadAttention", "TransformerBlock",
    "RLHF", "CoT", "CCoT", "GRPO", "RLVR", "DPO", "PPO", "LIME", "SHAP",
    
    # Configuration
    "Config", "ModelConfig", "TrainingConfig", "DataConfig", "LoRAConfig",
    "default_config", "load_config", "save_config", "get_preset_config", "validate_config",
    
    # Data
    "TextDataset", "LanguageModelingDataset", "DataCollator",
    "load_text_file", "load_json_file", "create_data_loader", "split_dataset",
    "SimpleTokenizer", "create_sample_dataset", "load_dataset_from_config",
    
    # Training
    "Trainer", "EarlyStopping", "MetricsTracker", "ModelCheckpoint", "create_trainer",
    
    # Utilities
    "set_seed", "get_device", "count_parameters", "count_lora_parameters",
    "encode_text", "decode_tokens", "create_attention_mask",
    "pad_sequences", "truncate_sequences", "compute_perplexity", "compute_bleu_score",
    "format_time", "format_size", "get_model_size", "print_model_summary",
    "save_model_info", "load_model_info", "log_gpu_memory", "cleanup_gpu_memory",
    
    # CLI
    "main",
    
    # Version
    "__version__"
]
