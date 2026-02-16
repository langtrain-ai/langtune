"""
quantization.py: 4-bit/8-bit Quantization Primitives for Langtune.

Implements Block-wise Quantization (similar to bitsandbytes/QLoRA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def quantize_blockwise(x: torch.Tensor, block_size: int = 64, bits: int = 4):
    """
    Quantize tensor block-wise.
    
    Args:
        x: Input tensor
        block_size: Size of quantization block
        bits: Target bits (4 or 8)
        
    Returns:
        quantized_x: Quint8 tensor
        absmax: Absolute max values per block
    """
    # 1. Reshape to blocks
    orig_shape = x.shape
    flattened = x.view(-1)
    
    # Pad if necessary
    numel = flattened.numel()
    padding = (block_size - (numel % block_size)) % block_size
    if padding > 0:
        flattened = F.pad(flattened, (0, padding))
    
    blocks = flattened.view(-1, block_size)
    
    # 2. Find Absmax per block
    absmax = blocks.abs().max(dim=1, keepdim=True)[0]
    absmax = absmax.clamp(min=1e-5) # Avoid zero division
    
    # 3. Quantize
    # Scale to range [-2^(b-1)-1, 2^(b-1)-1]
    # For 4-bit: [-7, 7]
    range_max = 2**(bits-1) - 1
    
    scaled = blocks / absmax * range_max
    quantized = scaled.round().to(torch.int8)
    
    return quantized, absmax, orig_shape, padding

def dequantize_blockwise(quantized: torch.Tensor, absmax: torch.Tensor, orig_shape: tuple, padding: int, bits: int = 4):
    """
    Dequantize block-wise.
    """
    range_max = 2**(bits-1) - 1
    
    # 1. Dequantize
    scaled = quantized.float() / range_max * absmax
    
    # 2. Reshape
    flattened = scaled.view(-1)
    
    # Remove padding
    if padding > 0:
        flattened = flattened[:-padding]
        
    return flattened.view(orig_shape)


class QuantizedLinear(nn.Module):
    """
    Linear layer with on-the-fly dequantization.
    Stores weights in low-bit precision.
    """
    def __init__(self, in_features, out_features, bias=True, bits=4, block_size=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.block_size = block_size
        
        # Placeholder for quantized weights (using int8 as container)
        self.register_buffer('weight_quant', torch.zeros((out_features * in_features // block_size, block_size), dtype=torch.int8)) 
        self.register_buffer('weight_absmax', torch.zeros((out_features * in_features // block_size, 1)))
        
        self.padding = 0
        self.orig_shape = (out_features, in_features)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
    def load_from_linear(self, linear: nn.Linear):
        """Compress a standard Linear layer."""
        with torch.no_grad():
            q, a, s, p = quantize_blockwise(linear.weight.data, self.block_size, self.bits)
            # Basic validation
            # In real impl, we'd handle shapes carefully
            self.weight_quant = q
            self.weight_absmax = a
            self.orig_shape = s
            self.padding = p
            
            if linear.bias is not None:
                self.bias.data = linear.bias.data
                
    def forward(self, x):
        # 1. Dequantize Weights (Just-In-Time)
        # Only unpacks needed blocks if optimized, here we unpack all
        weight = dequantize_blockwise(
            self.weight_quant, 
            self.weight_absmax, 
            self.orig_shape, 
            self.padding, 
            self.bits
        )
        
        # 2. Compute
        return F.linear(x, weight, self.bias)
