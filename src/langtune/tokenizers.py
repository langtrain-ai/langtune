"""
tokenizers.py: Tokenization utilities for Langtune

Provides tokenization helpers and wrappers for various tokenizers.
"""

import re
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class CharacterTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to character IDs."""
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        
        for char in text:
            token_id = ord(char) % (self.vocab_size - 4) + 4
            tokens.append(token_id)
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        chars = []
        for tid in token_ids:
            if skip_special_tokens and tid < 4:
                continue
            if tid >= 4:
                chars.append(chr((tid - 4) % 128 + 32))
        return ''.join(chars)
    
    def __call__(self, text: str, **kwargs) -> Dict[str, List[int]]:
        return {"input_ids": self.encode(text, **kwargs)}


class WordTokenizer:
    """Simple word-level tokenizer with vocabulary."""
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None, max_vocab_size: int = 32000):
        self.max_vocab_size = max_vocab_size
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        if not self.vocab:
            self.vocab = {
                self.pad_token: 0,
                self.unk_token: 1,
                self.bos_token: 2,
                self.eos_token: 3
            }
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def fit(self, texts: List[str], min_freq: int = 1):
        """Build vocabulary from texts."""
        word_counts = {}
        for text in texts:
            for word in self._tokenize(text):
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        
        # Add to vocabulary
        for word, count in sorted_words:
            if count < min_freq:
                break
            if len(self.vocab) >= self.max_vocab_size:
                break
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        logger.info(f"Built vocabulary with {len(self.vocab)} tokens")
    
    def _tokenize(self, text: str) -> List[str]:
        """Split text into words."""
        text = text.lower()
        return re.findall(r'\b\w+\b|[^\w\s]', text)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        
        for word in self._tokenize(text):
            tokens.append(self.vocab.get(word, self.unk_token_id))
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        words = []
        for tid in token_ids:
            if skip_special_tokens and tid < 4:
                continue
            word = self.inv_vocab.get(tid, self.unk_token)
            if not skip_special_tokens or not word.startswith("<"):
                words.append(word)
        return ' '.join(words)
    
    def __call__(self, text: str, **kwargs) -> Dict[str, List[int]]:
        return {"input_ids": self.encode(text, **kwargs)}
    
    def save(self, path: str):
        """Save vocabulary to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.vocab, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        """Load vocabulary from file."""
        import json
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab=vocab)


class BPETokenizer:
    """Simple Byte-Pair Encoding tokenizer."""
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
    
    def _get_pairs(self, word: List[str]) -> Dict[tuple, int]:
        """Get pairs of consecutive symbols."""
        pairs = {}
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs
    
    def fit(self, texts: List[str], num_merges: int = None):
        """Learn BPE merges from texts."""
        num_merges = num_merges or (self.vocab_size - 256)
        
        # Initialize vocabulary with characters
        word_freqs = {}
        for text in texts:
            for word in text.split():
                word = ' '.join(list(word)) + ' </w>'
                word_freqs[word] = word_freqs.get(word, 0) + 1
        
        # Build initial vocab
        self.vocab = {chr(i): i for i in range(256)}
        self.vocab['</w>'] = 256
        
        # Learn merges
        for i in range(num_merges):
            pairs = {}
            for word, freq in word_freqs.items():
                word_pairs = self._get_pairs(word.split())
                for pair, count in word_pairs.items():
                    pairs[pair] = pairs.get(pair, 0) + count * freq
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            new_token = ''.join(best_pair)
            
            if len(self.vocab) >= self.vocab_size:
                break
            
            self.merges[best_pair] = len(self.vocab)
            self.vocab[new_token] = len(self.vocab)
            
            # Update word_freqs
            new_word_freqs = {}
            pattern = ' '.join(best_pair)
            replacement = new_token
            for word, freq in word_freqs.items():
                new_word = word.replace(pattern, replacement)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs
        
        logger.info(f"Learned {len(self.merges)} BPE merges")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text using BPE."""
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        
        for word in text.split():
            word = ' '.join(list(word)) + ' </w>'
            while True:
                pairs = self._get_pairs(word.split())
                if not pairs:
                    break
                
                # Find best pair that exists in merges
                best_pair = None
                best_rank = float('inf')
                for pair in pairs:
                    if pair in self.merges and self.merges[pair] < best_rank:
                        best_pair = pair
                        best_rank = self.merges[pair]
                
                if best_pair is None:
                    break
                
                pattern = ' '.join(best_pair)
                replacement = ''.join(best_pair)
                word = word.replace(pattern, replacement)
            
            for token in word.split():
                tokens.append(self.vocab.get(token, self.unk_token_id))
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode BPE tokens."""
        inv_vocab = {v: k for k, v in self.vocab.items()}
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid < 4:
                continue
            token = inv_vocab.get(tid, '')
            tokens.append(token.replace('</w>', ' '))
        return ''.join(tokens).strip()


def get_tokenizer(name: str = "character", **kwargs):
    """Get a tokenizer by name."""
    tokenizers = {
        "character": CharacterTokenizer,
        "char": CharacterTokenizer,
        "word": WordTokenizer,
        "bpe": BPETokenizer
    }
    
    if name not in tokenizers:
        raise ValueError(f"Unknown tokenizer: {name}. Options: {list(tokenizers.keys())}")
    
    return tokenizers[name](**kwargs)
