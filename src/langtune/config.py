"""
config.py: Default configuration for Langtune
"""

import yaml

default_config = {
    'vocab_size': 32000,
    'embed_dim': 512,
    'num_layers': 6,
    'num_heads': 8,
    'mlp_ratio': 4,
    'lora': {
        'rank': 8,
        'alpha': 16,
        'dropout': 0.1,
        'target_modules': ['attention.qkv', 'attention.proj', 'mlp.fc1', 'mlp.fc2'],
        'merge_weights': False
    }
}

def load_config(path):
    """
    Loads a YAML config file and returns the config dictionary.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def update_config(base_config, updates):
    """
    Updates the base config dictionary with values from updates.
    """
    config = base_config.copy()
    config.update(updates)
    return config 