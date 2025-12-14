"""
utils.py: Utility functions for Langtune
"""

import torch
import numpy as np
import random
import logging
from typing import List, Dict, Any, Optional, Union
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_lora_parameters(model: torch.nn.Module) -> int:
    """Count the number of LoRA parameters in a model."""
    lora_params = 0
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_params += param.numel()
    return lora_params

def save_model_info(model: torch.nn.Module, save_path: str):
    """Save model information to a JSON file."""
    info = {
        "total_parameters": count_parameters(model),
        "lora_parameters": count_lora_parameters(model),
        "model_class": model.__class__.__name__,
        "model_config": getattr(model, 'config', None)
    }
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Model info saved to {save_path}")

def load_model_info(load_path: str) -> Dict[str, Any]:
    """Load model information from a JSON file."""
    with open(load_path, 'r') as f:
        info = json.load(f)
    return info

def encode_text(text: str, tokenizer=None) -> List[int]:
    """
    Encodes text into token IDs using the provided tokenizer.
    If no tokenizer is given, uses character-level encoding as fallback.
    """
    if tokenizer:
        if hasattr(tokenizer, 'encode'):
            return tokenizer.encode(text)
        elif callable(tokenizer):
            return tokenizer(text)
        else:
            raise ValueError("Invalid tokenizer provided")
    
    # Fallback to character-level encoding
    return [ord(c) for c in text]

def decode_tokens(token_ids: List[int], tokenizer=None) -> str:
    """
    Decodes a list of token IDs back into a string.
    """
    if tokenizer:
        if hasattr(tokenizer, 'decode'):
            return tokenizer.decode(token_ids)
        elif callable(tokenizer):
            return tokenizer(token_ids)
        else:
            raise ValueError("Invalid tokenizer provided")
    
    # Fallback to character-level decoding
    return ''.join([chr(i) for i in token_ids if i > 0])

class SimpleTokenizer:
    """
    A simple character-level tokenizer for demonstration purposes.
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Create a simple vocabulary
        self.vocab = {
            "<pad>": self.pad_token_id,
            "<unk>": self.unk_token_id,
            "<bos>": self.bos_token_id,
            "<eos>": self.eos_token_id
        }
        
        # Add common characters
        for i, char in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()[]{}"):
            if len(self.vocab) < vocab_size:
                self.vocab[char] = len(self.vocab)
        
        # Create reverse vocabulary
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.unk_token_id)
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token.startswith("<"):
                    continue
                tokens.append(token)
        
        return "".join(tokens)
    
    def __call__(self, text: str, **kwargs) -> Dict[str, List[int]]:
        """Callable interface for compatibility."""
        token_ids = self.encode(text, **kwargs)
        return {"input_ids": token_ids}

def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """Create attention mask from input IDs."""
    return (input_ids != pad_token_id).long()

def pad_sequences(sequences: List[torch.Tensor], pad_token_id: int = 0, max_length: Optional[int] = None) -> torch.Tensor:
    """Pad sequences to the same length."""
    if max_length is None:
        max_length = max(seq.size(0) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        if seq.size(0) < max_length:
            padding = torch.full((max_length - seq.size(0),), pad_token_id, dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq[:max_length]
        padded_sequences.append(padded_seq)
    
    return torch.stack(padded_sequences)

def truncate_sequences(sequences: List[torch.Tensor], max_length: int) -> List[torch.Tensor]:
    """Truncate sequences to maximum length."""
    return [seq[:max_length] for seq in sequences]

def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return np.exp(loss)

def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score (simplified implementation)."""
    # This is a very simplified BLEU implementation
    # In practice, you'd use a proper BLEU implementation like nltk.translate.bleu_score
    
    def get_ngrams(text: str, n: int) -> set:
        words = text.split()
        return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
    
    total_score = 0.0
    for pred, ref in zip(predictions, references):
        pred_ngrams = get_ngrams(pred, 1)  # Using unigrams for simplicity
        ref_ngrams = get_ngrams(ref, 1)
        
        if len(pred_ngrams) == 0:
            score = 0.0
        else:
            overlap = len(pred_ngrams.intersection(ref_ngrams))
            score = overlap / len(pred_ngrams)
        
        total_score += score
    
    return total_score / len(predictions)

def format_time(seconds: float) -> str:
    """Format time in a human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_size(size_bytes: int) -> str:
    """Format size in a human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f}KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f}MB"
    else:
        return f"{size_bytes/(1024**3):.1f}GB"

def create_directory_structure(base_dir: str, subdirs: List[str]):
    """Create a directory structure."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure: {base_dir}")

def save_json(data: Dict[str, Any], file_path: str):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Data saved to {file_path}")

def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Data loaded from {file_path}")
    return data

def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """Get model size information."""
    total_params = count_parameters(model)
    lora_params = count_lora_parameters(model)
    
    # Estimate memory usage (rough approximation)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size = param_size + buffer_size
    
    return {
        "total_parameters": total_params,
        "lora_parameters": lora_params,
        "regular_parameters": total_params - lora_params,
        "model_size_bytes": model_size,
        "model_size_mb": model_size / (1024**2),
        "lora_ratio": lora_params / total_params if total_params > 0 else 0
    }

def print_model_summary(model: torch.nn.Module):
    """Print a summary of the model."""
    size_info = get_model_size(model)
    
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Total parameters: {size_info['total_parameters']:,}")
    print(f"LoRA parameters: {size_info['lora_parameters']:,}")
    print(f"Regular parameters: {size_info['regular_parameters']:,}")
    print(f"LoRA ratio: {size_info['lora_ratio']:.2%}")
    print(f"Model size: {format_size(size_info['model_size_bytes'])}")
    print("=" * 50)

def warmup_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, base_lr: float, max_lr: float):
    """Create a learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def gradient_accumulation_steps(batch_size: int, effective_batch_size: int) -> int:
    """Calculate gradient accumulation steps."""
    if batch_size >= effective_batch_size:
        return 1
    
    steps = effective_batch_size // batch_size
    if effective_batch_size % batch_size != 0:
        steps += 1
    
    return steps

def log_gpu_memory():
    """Log GPU memory usage if available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def cleanup_gpu_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cleaned up")

# Backward compatibility
def encode_text_legacy(text, tokenizer=None):
    """Legacy function for backward compatibility."""
    logger.warning("encode_text_legacy is deprecated. Use encode_text instead.")
    return encode_text(text, tokenizer)

def decode_tokens_legacy(token_ids):
    """Legacy function for backward compatibility."""
    logger.warning("decode_tokens_legacy is deprecated. Use decode_tokens instead.")
    return decode_tokens(token_ids) 