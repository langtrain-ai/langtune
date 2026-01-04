"""
Langtune: Efficient LoRA Fine-Tuning for Text LLMs

This package provides tools and modules for efficient fine-tuning of large language models (LLMs) on text data using Low-Rank Adaptation (LoRA).
"""

import os
import sys

__version__ = "0.1.2"

# Banner display control
_BANNER_SHOWN = False
_SHOW_BANNER = os.environ.get("LANGTUNE_NO_BANNER", "0") != "1"


def _check_tpu_available() -> bool:
    """Check if Google TPU is available via torch_xla."""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        # Try to get a TPU device
        device = xm.xla_device()
        return "TPU" in str(device) or "xla" in str(device).lower()
    except (ImportError, RuntimeError, Exception):
        return False


def _get_tpu_version() -> str:
    """Get TPU version if available."""
    try:
        import torch_xla
        # Try to detect TPU version from environment or device info
        import os
        tpu_name = os.environ.get("TPU_NAME", "")
        if "v4" in tpu_name.lower():
            return "4"
        elif "v3" in tpu_name.lower():
            return "3"
        elif "v2" in tpu_name.lower():
            return "2"
        return "4"  # Default to v4 for newer TPUs
    except:
        return "?"


def _show_welcome_banner():
    """Display a beautiful welcome banner on first import."""
    global _BANNER_SHOWN
    
    if _BANNER_SHOWN or not _SHOW_BANNER:
        return
    
    _BANNER_SHOWN = True
    
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich import box
        import torch
        
        console = Console()
        
        # Check GPU/TPU availability with detailed info
        # Check for NVIDIA CUDA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            
            # Get NVIDIA-specific details
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                cuda_version = torch.version.cuda
                
                if gpu_count > 1:
                    gpu_info = f"✓ NVIDIA: {gpu_name} × {gpu_count} ({gpu_memory:.0f}GB each, CUDA {cuda_version})"
                else:
                    gpu_info = f"✓ NVIDIA: {gpu_name} ({gpu_memory:.0f}GB, CUDA {cuda_version})"
            except:
                gpu_info = f"✓ NVIDIA: {gpu_name}"
            
            gpu_style = "green"
        # Check for Google TPU (via torch_xla)
        elif _check_tpu_available():
            try:
                import torch_xla.core.xla_model as xm
                tpu_count = xm.xrt_world_size()
                gpu_info = f"✓ TPU: Google Cloud TPU v{_get_tpu_version()} ({tpu_count} cores)"
            except:
                gpu_info = "✓ TPU: Google Cloud TPU"
            gpu_style = "green"
        # Check for Apple MPS
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info = "✓ Apple: Metal Performance Shaders (MPS)"
            gpu_style = "green"
        else:
            gpu_info = "○ Accelerator: Not available (CPU mode)"
            gpu_style = "yellow"
        
        # Create banner content
        banner_text = Text()
        
        # Logo/Title
        banner_text.append("╔═══════════════════════════════════════════════════════════╗\n", style="cyan bold")
        banner_text.append("║", style="cyan bold")
        banner_text.append("                        ", style="")
        banner_text.append("LANGTUNE", style="bold magenta")
        banner_text.append("                         ", style="")
        banner_text.append("║\n", style="cyan bold")
        banner_text.append("║", style="cyan bold")
        banner_text.append("          Efficient LoRA Fine-Tuning for LLMs          ", style="dim")
        banner_text.append("║\n", style="cyan bold")
        banner_text.append("╚═══════════════════════════════════════════════════════════╝\n", style="cyan bold")
        
        # Info section
        banner_text.append("\n")
        banner_text.append("  📦 Version: ", style="dim")
        banner_text.append(f"v{__version__}\n", style="cyan")
        banner_text.append(f"  🖥️  {gpu_info}\n", style=gpu_style)
        banner_text.append("  📚 Docs: ", style="dim")
        banner_text.append("https://github.com/langtrain-ai/langtune\n", style="blue underline")
        
        # Quick start
        banner_text.append("\n  🚀 ", style="")
        banner_text.append("Quick Start:\n", style="bold")
        banner_text.append("     1. langtune auth login      ", style="cyan")
        banner_text.append("# Get key at langtrain.xyz\n", style="dim")
        banner_text.append("     2. langtune train --preset small --train-file data.txt\n", style="cyan")
        
        # Tips
        banner_text.append("\n  💡 ", style="")
        banner_text.append("Tip: ", style="yellow bold")
        banner_text.append("Set LANGTUNE_NO_BANNER=1 to disable this message\n", style="dim")
        
        console.print(banner_text)
        
    except ImportError:
        # Fallback to simple banner if rich is not available
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║                        LANGTUNE                           ║
║          Efficient LoRA Fine-Tuning for LLMs              ║
╚═══════════════════════════════════════════════════════════╝

  📦 Version: v{__version__}
  📚 Docs: https://langtrain.xyz

  🚀 Quick Start:
     1. langtune auth login                      # Get key at langtrain.xyz
     2. langtune train --preset small --train-file data.txt

  💡 Tip: Set LANGTUNE_NO_BANNER=1 to disable this message
""")


# Show banner on import (unless in non-interactive mode)
if sys.stdout.isatty():
    _show_welcome_banner()


# Core models
from .models import (
    LoRALanguageModel, LoRALinear, MultiHeadAttention, TransformerBlock,
    FastLoRALanguageModel, FastMultiHeadAttention, FastTransformerBlock
)

# Optimizations
from .optimizations import (
    OptimizationConfig, QuantizedLinear, LoRALinear4bit,
    RotaryPositionEmbedding, MemoryEfficientAttention,
    fused_cross_entropy, checkpoint, MixedPrecisionTrainer,
    get_memory_stats, cleanup_memory
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
    Trainer, FastTrainer, EarlyStopping, MetricsTracker, ModelCheckpoint,
    create_trainer, create_fast_trainer
)

# Fine-tuning (best-practice API)
from .finetune import (
    finetune as local_finetune, finetune_from_config, FineTuneConfig
)

# Client SDK
from .client import LangtuneClient, FineTuneJob, JobStatus, Model, APIError, get_client

# High-level API (server + local)
from .api import finetune, generate, chat, list_models, list_jobs, get_job, cancel_job

# Callbacks
from .callbacks import (
    Callback, CallbackList, ProgressCallback, LearningRateMonitorCallback,
    GradientMonitorCallback, ModelSizeCallback, TimerCallback, SaveHistoryCallback,
    MemoryMonitorCallback, WandbCallback, get_default_callbacks, get_verbose_callbacks
)

# Schedulers
from .schedulers import (
    WarmupScheduler, CosineAnnealingWithWarmup, LinearDecayWithWarmup,
    PolynomialDecayWithWarmup, ConstantWithWarmup, OneCycleLRWithWarmup, get_scheduler
)

# Metrics
from .metrics import (
    compute_perplexity, compute_accuracy, compute_top_k_accuracy,
    compute_bleu, compute_rouge_l, compute_diversity, MetricsCalculator
)

# Generation
from .generation import TextGenerator, generate

# Tokenizers
from .tokenizers import CharacterTokenizer, WordTokenizer, BPETokenizer, get_tokenizer

# Distributed
from .distributed import (
    is_distributed, get_rank, get_world_size, is_main_process,
    setup_distributed, cleanup_distributed, wrap_model_ddp, get_distributed_sampler
)

# Logging
from .logging_utils import (
    setup_logging, get_logger, TrainingLogger, ProgressTracker,
    print_banner, print_metrics
)

# Utilities
from .utils import (
    set_seed, get_device, count_parameters, count_lora_parameters,
    encode_text, decode_tokens, SimpleTokenizer, create_attention_mask,
    pad_sequences, truncate_sequences, compute_perplexity, compute_bleu_score,
    format_time, format_size, get_model_size, print_model_summary,
    save_model_info, load_model_info, log_gpu_memory, cleanup_gpu_memory
)

# Authentication
from .auth import (
    get_api_key, set_api_key, verify_api_key, check_usage,
    interactive_login, logout, print_usage_info,
    AuthenticationError, UsageLimitError, require_auth
)

# CLI
from .cli import main

# Facades (Quick Start API)
from .facade import LoRATrainer, QLoRATrainer, ChatModel, deploy

__all__ = [
    # Models
    "LoRALanguageModel", "LoRALinear", "MultiHeadAttention", "TransformerBlock",
    "FastLoRALanguageModel", "FastMultiHeadAttention", "FastTransformerBlock",
    "FastLoRALanguageModel", "FastMultiHeadAttention", "FastTransformerBlock",
    
    # Optimizations
    "OptimizationConfig", "QuantizedLinear", "LoRALinear4bit",
    "RotaryPositionEmbedding", "MemoryEfficientAttention",
    "fused_cross_entropy", "checkpoint", "MixedPrecisionTrainer",
    "get_memory_stats", "cleanup_memory",
    
    # Configuration
    "Config", "ModelConfig", "TrainingConfig", "DataConfig", "LoRAConfig",
    "default_config", "load_config", "save_config", "get_preset_config", "validate_config",
    
    # Data
    "TextDataset", "LanguageModelingDataset", "DataCollator",
    "load_text_file", "load_json_file", "create_data_loader", "split_dataset",
    "SimpleTokenizer", "create_sample_dataset", "load_dataset_from_config",
    
    # Training
    "Trainer", "FastTrainer", "EarlyStopping", "MetricsTracker", "ModelCheckpoint",
    "create_trainer", "create_fast_trainer",
    
    # Fine-tuning
    "finetune", "finetune_from_config", "FineTuneConfig", "train", "fine_tune",
    
    # Utilities
    "set_seed", "get_device", "count_parameters", "count_lora_parameters",
    "encode_text", "decode_tokens", "create_attention_mask",
    "pad_sequences", "truncate_sequences", "compute_perplexity", "compute_bleu_score",
    "format_time", "format_size", "get_model_size", "print_model_summary",
    "save_model_info", "load_model_info", "log_gpu_memory", "cleanup_gpu_memory",
    
    # Authentication
    "get_api_key", "set_api_key", "verify_api_key", "check_usage",
    "interactive_login", "logout", "print_usage_info",
    "AuthenticationError", "UsageLimitError", "require_auth",
    
    # CLI
    "main",
    
    # Facades
    "LoRATrainer", "QLoRATrainer", "ChatModel", "deploy",
    
    # Version
    "__version__"
]


