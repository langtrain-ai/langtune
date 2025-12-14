"""
Pytest configuration for Langtune tests.
"""

import pytest
import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def device():
    """Get available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def small_config():
    """Create a small config for fast testing."""
    return {
        "vocab_size": 1000,
        "embed_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "max_seq_len": 128,
        "mlp_ratio": 4.0,
        "dropout": 0.1
    }


@pytest.fixture
def lora_config():
    """LoRA configuration for testing."""
    return {
        "rank": 4,
        "alpha": 8.0,
        "dropout": 0.1
    }


@pytest.fixture
def sample_batch(small_config, device):
    """Create a sample batch for testing."""
    batch_size = 2
    seq_len = 32
    
    input_ids = torch.randint(0, small_config["vocab_size"], (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :5] = -100  # Mask some tokens
    
    return {
        "input_ids": input_ids,
        "labels": labels
    }
