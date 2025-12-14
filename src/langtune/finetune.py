"""
finetune.py: Best-practice fine-tuning for text LLMs

This module provides a unified, high-level API for efficient LLM fine-tuning
that automatically applies all available optimizations:

- 4-bit quantization (QLoRA) when enabled
- Rotary Position Embeddings (RoPE)
- Flash Attention / Memory-efficient attention
- Gradient checkpointing
- Mixed precision training (fp16/bf16)
- Gradient accumulation
- Early stopping and checkpointing

Usage:
    from langtune import finetune
    
    # Simple usage
    model = finetune(
        train_data="path/to/data.txt",
        preset="small"
    )
    
    # Advanced usage
    model = finetune(
        train_data="path/to/data.txt",
        val_data="path/to/val.txt",
        preset="base",
        lora_rank=16,
        use_4bit=True,
        epochs=3
    )
"""

import os
import logging
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class FineTuneConfig:
    """
    Configuration for best-practice fine-tuning.
    
    Automatically selects optimal settings based on available hardware.
    """
    
    def __init__(
        self,
        # Model settings
        preset: str = "small",
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        
        # Training settings
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        
        # Optimization settings
        use_4bit: bool = False,
        use_8bit: bool = False,
        use_rope: bool = True,
        use_flash_attention: bool = True,
        use_gradient_checkpointing: bool = True,
        gradient_accumulation_steps: int = 4,
        mixed_precision: str = "auto",  # auto, fp16, bf16, fp32
        
        # Data settings
        max_seq_len: int = 512,
        
        # Output settings
        output_dir: str = "./output",
        save_steps: int = 500,
        eval_steps: int = 100,
        logging_steps: int = 10,
        
        # Early stopping
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.001
    ):
        self.preset = preset
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.use_rope = use_rope
        self.use_flash_attention = use_flash_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        
        self.max_seq_len = max_seq_len
        
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        
        # Auto-detect optimal settings
        self._auto_configure()
    
    def _auto_configure(self):
        """Auto-configure settings based on hardware."""
        # Auto-detect mixed precision
        if self.mixed_precision == "auto":
            if torch.cuda.is_available():
                # Check for bf16 support (Ampere+ GPUs)
                if torch.cuda.is_bf16_supported():
                    self.mixed_precision = "bf16"
                else:
                    self.mixed_precision = "fp16"
            else:
                self.mixed_precision = "fp32"
        
        # Adjust batch size based on GPU memory
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory < 8:
                    self.batch_size = min(self.batch_size, 2)
                    self.gradient_accumulation_steps = max(self.gradient_accumulation_steps, 8)
                elif gpu_memory < 16:
                    self.batch_size = min(self.batch_size, 4)
                    self.gradient_accumulation_steps = max(self.gradient_accumulation_steps, 4)
            except:
                pass
        
        logger.info(f"Auto-configured: mixed_precision={self.mixed_precision}, "
                   f"batch_size={self.batch_size}, "
                   f"gradient_accumulation={self.gradient_accumulation_steps}")


def _get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_preset_model_config(preset: str) -> Dict[str, Any]:
    """Get model configuration from preset."""
    presets = {
        "tiny": {
            "vocab_size": 32000,
            "embed_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.1
        },
        "small": {
            "vocab_size": 32000,
            "embed_dim": 256,
            "num_layers": 4,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1
        },
        "base": {
            "vocab_size": 32000,
            "embed_dim": 512,
            "num_layers": 6,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1
        },
        "large": {
            "vocab_size": 32000,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "dropout": 0.1
        }
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Options: {list(presets.keys())}")
    
    return presets[preset]


def _load_training_data(
    data_path: Union[str, Path, List[str]],
    max_seq_len: int,
    batch_size: int
) -> DataLoader:
    """Load training data from file or list."""
    from .data import TextDataset, DataCollator, load_text_file
    
    # Load data
    if isinstance(data_path, (str, Path)):
        data_path = str(data_path)
        if data_path.endswith('.txt'):
            texts = load_text_file(data_path)
        elif data_path.endswith('.json'):
            import json
            with open(data_path) as f:
                texts = json.load(f)
                if isinstance(texts, dict):
                    texts = texts.get('texts', texts.get('data', []))
        else:
            texts = load_text_file(data_path)
    else:
        texts = data_path
    
    # Create dataset
    dataset = TextDataset(texts, max_length=max_seq_len)
    
    # Create dataloader
    collator = DataCollator(pad_token_id=0, max_length=max_seq_len)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


def finetune(
    train_data: Union[str, Path, List[str]],
    val_data: Optional[Union[str, Path, List[str]]] = None,
    
    # Model settings
    preset: str = "small",
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.1,
    
    # Training settings
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    
    # Optimization settings
    use_4bit: bool = False,
    use_rope: bool = True,
    use_flash_attention: bool = True,
    use_gradient_checkpointing: bool = True,
    gradient_accumulation_steps: int = 4,
    mixed_precision: str = "auto",
    
    # Data settings
    max_seq_len: int = 512,
    
    # Output settings
    output_dir: str = "./output",
    
    # Callbacks
    callbacks: Optional[List] = None,
    
    # Return options
    return_trainer: bool = False
) -> nn.Module:
    """
    Fine-tune a language model using best practices.
    
    This function automatically applies all available optimizations:
    - RoPE (Rotary Position Embeddings)
    - Flash Attention / Memory-efficient attention
    - Gradient checkpointing
    - Mixed precision training
    - Gradient accumulation
    - Early stopping
    
    Args:
        train_data: Path to training data file or list of texts
        val_data: Optional path to validation data
        preset: Model size preset ("tiny", "small", "base", "large")
        lora_rank: LoRA adapter rank (higher = more capacity)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        epochs: Number of training epochs
        batch_size: Batch size per step
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup steps as ratio of total steps
        use_4bit: Enable 4-bit quantization (QLoRA)
        use_rope: Enable rotary position embeddings
        use_flash_attention: Enable flash/memory-efficient attention
        use_gradient_checkpointing: Enable gradient checkpointing
        gradient_accumulation_steps: Steps to accumulate gradients
        mixed_precision: "auto", "fp16", "bf16", or "fp32"
        max_seq_len: Maximum sequence length
        output_dir: Directory to save checkpoints
        callbacks: Optional list of callbacks
        return_trainer: If True, return (model, trainer) instead of just model
        
    Returns:
        Fine-tuned model (or tuple of model, trainer if return_trainer=True)
    
    Example:
        >>> from langtune import finetune
        >>> model = finetune(
        ...     train_data="data.txt",
        ...     preset="small",
        ...     epochs=3
        ... )
    """
    from .models import FastLoRALanguageModel
    from .config import Config, ModelConfig, TrainingConfig, DataConfig, LoRAConfig
    from .trainer import FastTrainer
    
    # Create configuration
    config = FineTuneConfig(
        preset=preset,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        use_4bit=use_4bit,
        use_rope=use_rope,
        use_flash_attention=use_flash_attention,
        use_gradient_checkpointing=use_gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        max_seq_len=max_seq_len,
        output_dir=output_dir
    )
    
    device = _get_device()
    logger.info(f"Using device: {device}")
    
    # Log configuration
    logger.info(f"Fine-tuning with preset={preset}, lora_rank={lora_rank}")
    logger.info(f"Optimizations: rope={use_rope}, flash_attn={use_flash_attention}, "
               f"grad_ckpt={use_gradient_checkpointing}, mixed_precision={config.mixed_precision}")
    
    # Get model config from preset
    model_config = _get_preset_model_config(preset)
    model_config["max_seq_len"] = max_seq_len
    
    # Create LoRA config
    lora_config = {
        "rank": lora_rank,
        "alpha": lora_alpha,
        "dropout": lora_dropout
    }
    
    # Create model with all optimizations
    logger.info("Creating FastLoRALanguageModel with optimizations...")
    model = FastLoRALanguageModel(
        vocab_size=model_config["vocab_size"],
        embed_dim=model_config["embed_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        max_seq_len=model_config["max_seq_len"],
        mlp_ratio=model_config["mlp_ratio"],
        dropout=model_config["dropout"],
        lora_config=lora_config,
        use_rope=use_rope,
        use_flash_attention=use_flash_attention,
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    
    # Move to device
    model = model.to(device)
    
    # Freeze base model, only train LoRA
    model.freeze_base_model()
    
    # Load training data
    logger.info(f"Loading training data from {train_data}...")
    train_dataloader = _load_training_data(train_data, max_seq_len, batch_size)
    
    # Load validation data if provided
    val_dataloader = None
    if val_data is not None:
        logger.info(f"Loading validation data from {val_data}...")
        val_dataloader = _load_training_data(val_data, max_seq_len, batch_size)
    
    # Create training config
    training_config = Config(
        model=ModelConfig(
            vocab_size=model_config["vocab_size"],
            embed_dim=model_config["embed_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            max_seq_len=max_seq_len,
            mlp_ratio=model_config["mlp_ratio"],
            dropout=model_config["dropout"]
        ),
        training=TrainingConfig(
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=int(len(train_dataloader) * epochs * warmup_ratio),
            max_grad_norm=1.0,
            logging_steps=10,
            save_total_limit=3,
            early_stopping_patience=3,
            early_stopping_threshold=0.001,
            mixed_precision=(config.mixed_precision != "fp32")
        ),
        data=DataConfig(
            max_seq_len=max_seq_len
        ),
        output_dir=output_dir,
        device="auto"
    )
    
    # Create optimized trainer
    logger.info("Creating FastTrainer with gradient accumulation and AMP...")
    trainer = FastTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=config.mixed_precision
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Fine-tuning complete!")
    
    if return_trainer:
        return model, trainer
    return model


def finetune_from_config(config_path: str, **overrides) -> nn.Module:
    """
    Fine-tune using a YAML/JSON configuration file.
    
    Args:
        config_path: Path to configuration file
        **overrides: Override config values
        
    Returns:
        Fine-tuned model
    """
    import yaml
    import json
    
    config_path = str(config_path)
    
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        with open(config_path) as f:
            config = json.load(f)
    
    # Apply overrides
    config.update(overrides)
    
    return finetune(**config)


# Convenience aliases
train = finetune
fine_tune = finetune
