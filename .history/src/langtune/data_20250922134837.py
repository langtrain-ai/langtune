"""
data.py: Data loading and preprocessing utilities for Langtune
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Union, Iterator
import logging
from pathlib import Path
import random
import numpy as np

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """
    A PyTorch Dataset for text data with tokenization support.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer=None,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        add_special_tokens: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer object (optional)
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
            add_special_tokens: Whether to add special tokens
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens
        
        logger.info(f"Initialized TextDataset with {len(texts)} samples")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        if self.tokenizer:
            # Use provided tokenizer
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                add_special_tokens=self.add_special_tokens,
                return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding.get("attention_mask", torch.ones_like(encoding["input_ids"])).squeeze(0)
            }
        else:
            # Simple character-level tokenization as fallback
            input_ids = torch.tensor([ord(c) for c in text[:self.max_length]], dtype=torch.long)
            
            # Pad or truncate
            if len(input_ids) < self.max_length:
                padding_length = self.max_length - len(input_ids)
                input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
                attention_mask = torch.cat([torch.ones(len(input_ids) - padding_length), torch.zeros(padding_length)])
            else:
                attention_mask = torch.ones(self.max_length)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

class LanguageModelingDataset(Dataset):
    """
    Dataset for language modeling tasks with next-token prediction.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer=None,
        max_length: int = 512,
        stride: int = 128,
        padding: str = "max_length",
        truncation: bool = True
    ):
        """
        Initialize the language modeling dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer object
            max_length: Maximum sequence length
            stride: Stride for sliding window
            padding: Padding strategy
            truncation: Whether to truncate sequences
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.padding = padding
        self.truncation = truncation
        
        # Process texts into sequences
        self.sequences = self._create_sequences()
        
        logger.info(f"Initialized LanguageModelingDataset with {len(self.sequences)} sequences")
    
    def _create_sequences(self) -> List[Dict[str, torch.Tensor]]:
        """Create sequences for language modeling."""
        sequences = []
        
        for text in self.texts:
            if self.tokenizer:
                # Tokenize the text
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                
                # Create sliding window sequences
                for i in range(0, len(tokens), self.stride):
                    sequence = tokens[i:i + self.max_length]
                    
                    if len(sequence) < self.max_length:
                        # Pad sequence
                        sequence = sequence + [self.tokenizer.pad_token_id] * (self.max_length - len(sequence))
                        attention_mask = [1] * (len(sequence) - (self.max_length - len(sequence))) + [0] * (self.max_length - len(sequence))
                    else:
                        attention_mask = [1] * self.max_length
                    
                    # Create labels (shifted by 1 for next token prediction)
                    labels = sequence[1:] + [-100]  # -100 is ignored in loss computation
                    
                    sequences.append({
                        "input_ids": torch.tensor(sequence, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long)
                    })
            else:
                # Simple character-level processing
                chars = [ord(c) for c in text]
                
                for i in range(0, len(chars), self.stride):
                    sequence = chars[i:i + self.max_length]
                    
                    if len(sequence) < self.max_length:
                        sequence = sequence + [0] * (self.max_length - len(sequence))
                        attention_mask = [1] * (len(sequence) - (self.max_length - len(sequence))) + [0] * (self.max_length - len(sequence))
                    else:
                        attention_mask = [1] * self.max_length
                    
                    labels = sequence[1:] + [-100]
                    
                    sequences.append({
                        "input_ids": torch.tensor(sequence, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long)
                    })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.sequences[idx]

class DataCollator:
    """
    Data collator for batching sequences.
    """
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of sequences.
        
        Args:
            batch: List of dictionaries containing sequences
            
        Returns:
            Batched tensors
        """
        # Get the maximum length in the batch
        max_length = max(item["input_ids"].size(0) for item in batch)
        
        # Pad sequences to the same length
        input_ids = []
        attention_masks = []
        labels = []
        
        for item in batch:
            seq_len = item["input_ids"].size(0)
            
            # Pad input_ids
            if seq_len < max_length:
                padding = torch.full((max_length - seq_len,), self.pad_token_id, dtype=torch.long)
                input_ids.append(torch.cat([item["input_ids"], padding]))
            else:
                input_ids.append(item["input_ids"])
            
            # Pad attention_mask
            if seq_len < max_length:
                padding = torch.zeros(max_length - seq_len, dtype=torch.long)
                attention_masks.append(torch.cat([item["attention_mask"], padding]))
            else:
                attention_masks.append(item["attention_mask"])
            
            # Pad labels if present
            if "labels" in item:
                if seq_len < max_length:
                    padding = torch.full((max_length - seq_len,), -100, dtype=torch.long)
                    labels.append(torch.cat([item["labels"], padding]))
                else:
                    labels.append(item["labels"])
        
        result = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks)
        }
        
        if labels:
            result["labels"] = torch.stack(labels)
        
        return result

def load_text_file(file_path: str, encoding: str = "utf-8") -> List[str]:
    """
    Load text from a file.
    
    Args:
        file_path: Path to the text file
        encoding: File encoding
        
    Returns:
        List of text lines
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    
    # Remove empty lines and strip whitespace
    lines = [line.strip() for line in lines if line.strip()]
    
    logger.info(f"Loaded {len(lines)} lines from {file_path}")
    return lines

def load_json_file(file_path: str, text_key: str = "text") -> List[str]:
    """
    Load text from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        text_key: Key containing the text data
        
    Returns:
        List of text strings
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        texts = [item[text_key] for item in data if text_key in item]
    elif isinstance(data, dict):
        if text_key in data:
            texts = [data[text_key]]
        else:
            raise ValueError(f"Key '{text_key}' not found in JSON data")
    else:
        raise ValueError("JSON data must be a list or dictionary")
    
    logger.info(f"Loaded {len(texts)} texts from {file_path}")
    return texts

def create_data_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    collate_fn=None
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        collate_fn: Custom collate function
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

def split_dataset(
    texts: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple[List[str], List[str], List[str]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        texts: List of text strings
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed
        
    Returns:
        Tuple of (train_texts, val_texts, test_texts)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle the data
    indices = list(range(len(texts)))
    random.shuffle(indices)
    
    # Calculate split points
    train_size = int(len(texts) * train_ratio)
    val_size = int(len(texts) * val_ratio)
    
    # Split the data
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    test_texts = [texts[i] for i in test_indices]
    
    logger.info(f"Split dataset: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
    
    return train_texts, val_texts, test_texts

class SimpleTokenizer:
    """
    A simple tokenizer for demonstration purposes.
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

# Example usage and utility functions
def create_sample_dataset(num_samples: int = 1000, text_length: int = 100) -> List[str]:
    """
    Create a sample dataset for testing.
    
    Args:
        num_samples: Number of samples to generate
        text_length: Length of each text sample
        
    Returns:
        List of sample texts
    """
    sample_texts = []
    
    for i in range(num_samples):
        # Generate random text
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()[]{}"
        text = "".join(random.choices(chars, k=text_length))
        sample_texts.append(text)
    
    return sample_texts

def load_dataset_from_config(config) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load datasets based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load training data
    if config.data.train_file:
        if config.data.train_file.endswith('.json'):
            train_texts = load_json_file(config.data.train_file)
        else:
            train_texts = load_text_file(config.data.train_file)
    else:
        logger.warning("No training file specified, creating sample dataset")
        train_texts = create_sample_dataset(1000)
    
    # Load validation data
    if config.data.eval_file:
        if config.data.eval_file.endswith('.json'):
            val_texts = load_json_file(config.data.eval_file)
        else:
            val_texts = load_text_file(config.data.eval_file)
    else:
        # Split training data for validation
        train_texts, val_texts, _ = split_dataset(train_texts, train_ratio=0.9, val_ratio=0.1, test_ratio=0.0)
    
    # Load test data
    if config.data.test_file:
        if config.data.test_file.endswith('.json'):
            test_texts = load_json_file(config.data.test_file)
        else:
            test_texts = load_text_file(config.data.test_file)
    else:
        # Split training data for test
        train_texts, _, test_texts = split_dataset(train_texts, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2)
    
    # Create tokenizer
    tokenizer = None
    if config.data.tokenizer_name:
        # In a real implementation, you would load a proper tokenizer here
        # For now, we'll use the simple tokenizer
        tokenizer = SimpleTokenizer(config.model.vocab_size)
    
    # Create datasets
    train_dataset = LanguageModelingDataset(
        train_texts,
        tokenizer=tokenizer,
        max_length=config.data.max_length
    )
    
    val_dataset = LanguageModelingDataset(
        val_texts,
        tokenizer=tokenizer,
        max_length=config.data.max_length
    )
    
    test_dataset = LanguageModelingDataset(
        test_texts,
        tokenizer=tokenizer,
        max_length=config.data.max_length
    )
    
    return train_dataset, val_dataset, test_dataset
