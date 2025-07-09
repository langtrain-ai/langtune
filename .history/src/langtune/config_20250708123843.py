"""
config.py: Default configuration for Langtune
"""

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