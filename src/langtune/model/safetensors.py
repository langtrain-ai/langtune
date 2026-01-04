"""
Safetensors Streamer.

Efficiently streams tensors from disk using memory mapping.
"""

import os
import logging
import torch
from typing import Dict, List, Optional, Union, Iterator, Tuple
from pathlib import Path
from safetensors import safe_open
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TensorInfo:
    """Metadata about a tensor in a safetensors file."""
    name: str
    shape: List[int]
    dtype: str
    file_path: str

class TensorStreamer:
    """
    Manages lazy loading of tensors from a directory of safetensors files.
    """
    
    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        self.files = sorted(list(self.model_dir.glob("*.safetensors")))
        
        if not self.files:
            raise FileNotFoundError(f"No .safetensors files found in {self.model_dir}")
            
        self.index: Dict[str, TensorInfo] = {}
        self._build_index()
        
    def _build_index(self):
        """Scan files and build an index of tensor locations."""
        logger.info(f"Indexing {len(self.files)} safetensors files...")
        
        for file_path in self.files:
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        # We don't load the tensor, just inspecting keys
                        # Note: safe_open doesn't give shape/dtype without loading or accessing metadata
                        # Ideally we read the header. safe_open provides proper access.
                        # For pure metadata scan without loading payload, external tools or specialized logic is best.
                        # But standard safe_open is memory efficient (mmap).
                        
                        # We store just the file mapping for now to keep it fast.
                        # Conflict check
                        if key in self.index:
                            logger.warning(f"Duplicate tensor {key} found in {file_path}, ignoring duplicate.")
                            continue
                            
                        self.index[key] = TensorInfo(
                            name=key,
                            shape=[], # Lazy populate if needed, or read tensor for shape cheap
                            dtype="",
                            file_path=str(file_path)
                        )
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                raise
                
    def has_tensor(self, name: str) -> bool:
        return name in self.index
        
    def get_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        """
        Load a single tensor.
        """
        if name not in self.index:
            raise KeyError(f"Tensor {name} not found in model files.")
            
        info = self.index[name]
        
        with safe_open(info.file_path, framework="pt", device=device) as f:
            return f.get_tensor(name)

    def load_state_dict(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Load all tensors into a state dict (WARNING: High Memory Usage).
        Use for small models or debugging only.
        """
        state_dict = {}
        for name in self.index:
            state_dict[name] = self.get_tensor(name, device)
        return state_dict

    def stream(self, device: str = "cpu") -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Yields (name, tensor) pairs one by one to minimize memory usage.
        """
        # Optimize by opening files once
        for file_path in self.files:
            with safe_open(file_path, framework="pt", device=device) as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

# Layer priority order for optimal VRAM management
LAYER_PRIORITY = [
    'self_attn',      # Attention layers first (critical path)
    'q_proj', 'k_proj', 'v_proj', 'o_proj',  # Attention projections
    'mlp',            # MLP layers
    'gate_proj', 'up_proj', 'down_proj',     # MLP projections  
    'input_layernorm', 'post_attention_layernorm',  # Norms last (small)
    'embed_tokens', 'lm_head'  # Embeddings
]

class PriorityTensorStreamer(TensorStreamer):
    """
    Enhanced streamer that loads tensors in priority order to minimize peak VRAM.
    """
    
    def stream_by_priority(self, device: str = "cpu") -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Yields tensors in priority order: attention → MLP → norms.
        This minimizes VRAM spikes during loading.
        """
        yielded = set()
        
        # First pass: yield in priority order
        for priority_key in LAYER_PRIORITY:
            for name in self.index:
                if name in yielded:
                    continue
                if priority_key in name:
                    yield name, self.get_tensor(name, device)
                    yielded.add(name)
        
        # Second pass: yield any remaining tensors
        for name in self.index:
            if name not in yielded:
                yield name, self.get_tensor(name, device)
                yielded.add(name)

    def get_load_order(self) -> List[str]:
        """
        Returns tensor names in priority order (for planning/debugging).
        """
        ordered = []
        remaining = set(self.index.keys())
        
        for priority_key in LAYER_PRIORITY:
            for name in list(remaining):
                if priority_key in name:
                    ordered.append(name)
                    remaining.remove(name)
        
        # Add remaining
        ordered.extend(sorted(remaining))
        return ordered
