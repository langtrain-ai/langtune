"""
config.py: Configuration management for Langtune
"""

import yaml
import json
import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    """LoRA configuration parameters."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: list = None
    merge_weights: bool = False
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ['attention.qkv', 'attention.proj', 'mlp.fc1', 'mlp.fc2']

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 32000
    embed_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    max_seq_len: int = 512
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    lora: LoRAConfig = None
    
    def __post_init__(self):
        if self.lora is None:
            self.lora = LoRAConfig()

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    test_file: Optional[str] = None
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None

@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda, mps
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)

# Default configurations
default_model_config = ModelConfig()
default_training_config = TrainingConfig()
default_data_config = DataConfig()
default_config = Config(
    model=default_model_config,
    training=default_training_config,
    data=default_data_config
)

# Preset configurations for different model sizes
PRESET_CONFIGS = {
    "tiny": {
        "model": {
            "vocab_size": 10000,
            "embed_dim": 128,
            "num_layers": 4,
            "num_heads": 4,
            "max_seq_len": 256,
            "lora": {"rank": 4, "alpha": 8}
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 2e-4
        }
    },
    "small": {
        "model": {
            "vocab_size": 32000,
            "embed_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "max_seq_len": 512,
            "lora": {"rank": 8, "alpha": 16}
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4
        }
    },
    "base": {
        "model": {
            "vocab_size": 50257,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "max_seq_len": 1024,
            "lora": {"rank": 16, "alpha": 32}
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 5e-5
        }
    },
    "large": {
        "model": {
            "vocab_size": 50257,
            "embed_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "max_seq_len": 1024,
            "lora": {"rank": 32, "alpha": 64}
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 2e-5
        }
    }
}

def load_config(path: str) -> Config:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        Config object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        elif path.endswith('.json'):
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path}")
    
    return dict_to_config(config_dict)

def save_config(config: Config, path: str) -> None:
    """
    Save configuration to a YAML or JSON file.
    
    Args:
        config: Config object to save
        path: Path to save the configuration file
    """
    config_dict = config_to_dict(config)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif path.endswith('.json'):
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {path}")

def dict_to_config(config_dict: Dict[str, Any]) -> Config:
    """
    Convert a dictionary to a Config object.
    
    Args:
        config_dict: Dictionary containing configuration
        
    Returns:
        Config object
    """
    # Handle nested dictionaries for dataclasses
    if 'model' in config_dict and isinstance(config_dict['model'], dict):
        if 'lora' in config_dict['model'] and isinstance(config_dict['model']['lora'], dict):
            config_dict['model']['lora'] = LoRAConfig(**config_dict['model']['lora'])
        config_dict['model'] = ModelConfig(**config_dict['model'])
    
    if 'training' in config_dict and isinstance(config_dict['training'], dict):
        config_dict['training'] = TrainingConfig(**config_dict['training'])
    
    if 'data' in config_dict and isinstance(config_dict['data'], dict):
        config_dict['data'] = DataConfig(**config_dict['data'])
    
    return Config(**config_dict)

def config_to_dict(config: Config) -> Dict[str, Any]:
    """
    Convert a Config object to a dictionary.
    
    Args:
        config: Config object
        
    Returns:
        Dictionary representation
    """
    return asdict(config)

def get_preset_config(preset_name: str) -> Config:
    """
    Get a preset configuration.
    
    Args:
        preset_name: Name of the preset (tiny, small, base, large)
        
    Returns:
        Config object
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    # Start with default config
    config_dict = config_to_dict(default_config)
    
    # Update with preset values
    preset_dict = PRESET_CONFIGS[preset_name]
    config_dict = deep_update(config_dict, preset_dict)
    
    return dict_to_config(config_dict)

def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep update a dictionary with values from another dictionary.
    
    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    result = base_dict.copy()
    
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    
    return result

def update_config(base_config: Config, updates: Dict[str, Any]) -> Config:
    """
    Update a configuration with new values.
    
    Args:
        base_config: Base configuration
        updates: Dictionary with updates
        
    Returns:
        Updated Config object
    """
    config_dict = config_to_dict(base_config)
    updated_dict = deep_update(config_dict, updates)
    return dict_to_config(updated_dict)

def validate_config(config: Config) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Config object to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Model validation
    if config.model.embed_dim % config.model.num_heads != 0:
        raise ValueError(f"embed_dim ({config.model.embed_dim}) must be divisible by num_heads ({config.model.num_heads})")
    
    if config.model.vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {config.model.vocab_size}")
    
    if config.model.num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {config.model.num_layers}")
    
    # LoRA validation
    if config.model.lora.rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {config.model.lora.rank}")
    
    if config.model.lora.alpha <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {config.model.lora.alpha}")
    
    # Training validation
    if config.training.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.training.batch_size}")
    
    if config.training.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.training.learning_rate}")
    
    if config.training.num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {config.training.num_epochs}")
    
    # Data validation
    if config.data.max_length <= 0:
        raise ValueError(f"max_length must be positive, got {config.data.max_length}")
    
    logger.info("Configuration validation passed")

# Backward compatibility
def load_config_legacy(path):
    """Legacy function for backward compatibility."""
    logger.warning("load_config_legacy is deprecated. Use load_config instead.")
    config_dict = load_config(path)
    return config_to_dict(config_dict)

def update_config_legacy(base_config, updates):
    """Legacy function for backward compatibility."""
    logger.warning("update_config_legacy is deprecated. Use update_config instead.")
    if isinstance(base_config, dict):
        base_config = dict_to_config(base_config)
    updated_config = update_config(base_config, updates)
    return config_to_dict(updated_config) 