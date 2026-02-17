"""
presets.py: Model configuration presets for Langtune
"""
from typing import Dict, Any

def get_preset_model_config(preset: str) -> Dict[str, Any]:
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
