"""
Accelerator module for Langtune.

This module provides an interface to access high-performance custom kernels
from langtrain-server if they are available in the environment.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Autograd Functions
class FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, eps, kernels):
        ctx.save_for_backward(input, weight)
        ctx.eps = eps
        ctx.kernels = kernels
        return kernels.fused_rmsnorm_forward(input, weight, eps)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight = ctx.kernels.fused_rmsnorm_backward(grad_output, input, weight, ctx.eps)
        return grad_input, grad_weight, None, None

class FusedLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, base_weight, lora_A, lora_B, scaling, kernels):
        ctx.save_for_backward(input, base_weight, lora_A, lora_B)
        ctx.scaling = scaling
        ctx.kernels = kernels
        return kernels.fused_lora_forward(input, base_weight, lora_A, lora_B, scaling)

    @staticmethod
    def backward(ctx, grad_output):
        input, base_weight, lora_A, lora_B = ctx.saved_tensors
        grad_input, grad_A, grad_B = ctx.kernels.lora_backward(grad_output, input, lora_A, lora_B, base_weight, ctx.scaling)
        return grad_input, None, grad_A, grad_B, None, None

class Accelerator:
    """
    Manages access to accelerated kernels.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Accelerator, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.available = False
        self.kernels = None
        
        try:
            import langtrain_cuda
            self.kernels = langtrain_cuda
            self.available = True
            logger.info("Langtrain high-performance kernels detected and enabled.")
        except ImportError:
            logger.info("Langtrain kernels not found. Using standard PyTorch implementations.")
            
    def is_available(self) -> bool:
        return self.available

    def fused_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = True,
        scale: float = None
    ) -> torch.Tensor:
        """
        Run fused attention forward pass.
        """
        if self.available and self.kernels:
            if scale is None:
                scale = query.size(-1) ** -0.5
            # Attention fallback to PyTorch for backward safety if kernel lacks it
            # For now, we use standard SDPA which is Flash-enabled in PT 2.0+
            return torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=is_causal, scale=scale)
        else:
            return torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=is_causal, scale=scale)

    def fused_rmsnorm(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Run fused RMSNorm.
        """
        if self.available and self.kernels:
            return FusedRMSNormFunction.apply(hidden_states, weight, eps, self.kernels)
        return None

    def fused_mlp(
        self,
        hidden_states: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Run fused SwiGLU MLP.
        """
        # Fallback to PyTorch decomposotion as we lack fused backward kernel for MLP
        return (torch.nn.functional.silu(torch.nn.functional.linear(hidden_states, gate_weight)) * 
                torch.nn.functional.linear(hidden_states, up_weight)).matmul(down_weight.t())

    def fused_lora(
        self,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float
    ) -> torch.Tensor:
        """
        Run fused LoRA forward.
        """
        if self.available and self.kernels:
            return FusedLoRAFunction.apply(x, base_weight, lora_A, lora_B, scaling, self.kernels)
        return None
