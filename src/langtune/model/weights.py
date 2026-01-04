"""
Weight Loader.

Handles reading weights from the streamer and applying:
- Quantize-on-Read (NF4/FP4)
- Precision casting (BF16/FP16)
- Kernel weight format definition
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from .safetensors import TensorStreamer

logger = logging.getLogger(__name__)

class WeightLoader:
    """
    Loads weights into a model structure.
    
    Supports:
    - Direct quantization (NF4) during loading to save memory
    - BF16/FP16 casting
    - Kernel-specific formatting
    """
    
    def __init__(
        self,
        streamer: TensorStreamer,
        quantization_mode: Optional[str] = None, # "nf4", "fp4", "int8"
        compute_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda"
    ):
        self.streamer = streamer
        self.quantization_mode = quantization_mode
        self.compute_dtype = compute_dtype
        self.device = device
        
    def load_into_module(self, module: nn.Module, prefix: str = "", skip_patterns: Optional[list] = None):
        """
        Populate a module's weights from the streamer.
        
        Args:
            module: Target module to load weights into
            prefix: Current prefix for nested modules
            skip_patterns: List of patterns to skip (for planner-aware loading)
        """
        skip_patterns = skip_patterns or []
        
        # Get all parameters/buffers in the module
        # state_dict keys are relative to the module if we use module.named_parameters()
        # But streamer has full keys (e.g. "model.layers.0.self_attn.q_proj.weight")
        
        # We assume the model structure matches the streamer keys for now
        # Ideally, we iterate over the model parameters and fetch corresponding keys
        
        for name, param in module.named_parameters(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this layer should be skipped (planner-aware)
            if any(pattern in full_name for pattern in skip_patterns):
                logger.debug(f"Skipping frozen layer: {full_name}")
                continue
            
            if self.streamer.has_tensor(full_name):
                tensor = self.streamer.get_tensor(full_name, device="cpu")
                
                # Apply Quantize-on-Read
                # Skip quantization for: LoRA adapters, norms, and BIAS terms
                should_quantize = (
                    self.quantization_mode 
                    and "lora" not in full_name 
                    and "norm" not in full_name
                    and "bias" not in name  # Bias terms stay in full precision!
                    and tensor.dim() > 1    # Only quantize 2D+ tensors (weights)
                )
                
                if should_quantize:
                     self._quantize_and_assign(module, name, tensor)
                else:
                    # Standard loading
                    with torch.no_grad():
                        param.data = tensor.to(self.device, dtype=self.compute_dtype)
                        
        # Recursively load children
        for child_name, child in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self.load_into_module(child, prefix=child_prefix, skip_patterns=skip_patterns)

    def _quantize_and_assign(self, module: nn.Module, param_name: str, tensor: torch.Tensor):
        """
        Quantize tensor and replace parameter in module.
        Uses pinned memory for async GPU transfer to avoid staging buffers.
        """
        if self.quantization_mode == "nf4":
            try:
                import bitsandbytes as bnb
                from bitsandbytes.nn import Params4bit
                
                # Use pinned memory for async transfer (Unsloth-killer optimization)
                if not tensor.is_pinned() and self.device != "cpu":
                    tensor = tensor.pin_memory()
                
                # Create CUDA stream for async quantization
                if self.device != "cpu" and torch.cuda.is_available():
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        tensor_gpu = tensor.to(self.device, non_blocking=True)
                        stream.synchronize()
                else:
                    tensor_gpu = tensor.to(self.device)
                
                param_4bit = Params4bit(
                    tensor_gpu, 
                    requires_grad=False, 
                    compress_statistics=True, 
                    quant_type="nf4"
                )
                
                # Replace parameter
                setattr(module, param_name, param_4bit)
                
            except ImportError:
                logger.warning("bitsandbytes not found, falling back to BF16/FP16")
                with torch.no_grad():
                    getattr(module, param_name).data = tensor.to(self.device, dtype=self.compute_dtype)
        else:
            with torch.no_grad():
                getattr(module, param_name).data = tensor.to(self.device, dtype=self.compute_dtype)
