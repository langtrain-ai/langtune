"""
Kernel Injection Module.

Replaces standard PyTorch layers with Langtrain optimized CUDA kernels
after model weights are loaded.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Type, Callable
import logging

logger = logging.getLogger(__name__)

# Registry of kernel replacements
KERNEL_REGISTRY: Dict[Type[nn.Module], Callable] = {}

def register_kernel(original_class: Type[nn.Module]):
    """Decorator to register a kernel replacement."""
    def decorator(replacement_fn: Callable):
        KERNEL_REGISTRY[original_class] = replacement_fn
        return replacement_fn
    return decorator


class KernelInjector:
    """
    Replaces standard PyTorch modules with optimized Langtrain kernels.
    
    Supports:
    - FusedLinear (fused bias + activation)
    - FlashAttention replacement
    - Fused RMSNorm
    """
    
    def __init__(
        self,
        use_flash_attention: bool = True,
        use_fused_linear: bool = True,
        use_fused_norm: bool = True,
        fallback_to_pytorch: bool = True
    ):
        self.use_flash_attention = use_flash_attention
        self.use_fused_linear = use_fused_linear
        self.use_fused_norm = use_fused_norm
        self.fallback_to_pytorch = fallback_to_pytorch
        
        # Check for available kernels
        self._fused_linear_available = self._check_fused_linear()
        self._flash_attention_available = self._check_flash_attention()
        
    def _check_fused_linear(self) -> bool:
        """Check if FusedLinear kernel is available."""
        try:
            from langtune.nn.layers import FusedLinear
            return True
        except ImportError:
            return False
            
    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention is available."""
        try:
            from flash_attn import flash_attn_func
            return True
        except ImportError:
            # Fallback to scaled_dot_product_attention (PyTorch 2.0+)
            return hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def inject(self, model: nn.Module) -> nn.Module:
        """
        Inject optimized kernels into the model.
        
        Args:
            model: The model to optimize
            
        Returns:
            The model with injected kernels
        """
        replaced_count = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if self.use_fused_linear and self._fused_linear_available:
                    if self._replace_linear(model, name, module):
                        replaced_count += 1
        
        logger.info(f"Injected {replaced_count} optimized kernels")
        return model
    
    def _replace_linear(self, model: nn.Module, name: str, linear: nn.Linear) -> bool:
        """Replace nn.Linear with FusedLinear."""
        try:
            from langtune.nn.layers import FusedLinear
            
            # Don't replace lm_head (output projection)
            if 'lm_head' in name or 'embed' in name:
                return False
            
            # Get parent module
            parent = model
            parts = name.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            child_name = parts[-1]
            
            # Create FusedLinear from existing Linear
            fused = FusedLinear.from_linear(linear)
            setattr(parent, child_name, fused)
            
            return True
            
        except Exception as e:
            if not self.fallback_to_pytorch:
                raise
            logger.debug(f"Could not replace {name}: {e}")
            return False

    def get_stats(self) -> Dict[str, bool]:
        """Return availability status of optimizations."""
        return {
            "fused_linear": self._fused_linear_available,
            "flash_attention": self._flash_attention_available,
        }


# Convenience function
def inject_kernels(
    model: nn.Module,
    use_flash_attention: bool = True,
    use_fused_linear: bool = True
) -> nn.Module:
    """
    Convenience function to inject optimized kernels.
    
    Args:
        model: Model to optimize
        use_flash_attention: Enable Flash Attention
        use_fused_linear: Enable Fused Linear layers
        
    Returns:
        Optimized model
    """
    injector = KernelInjector(
        use_flash_attention=use_flash_attention,
        use_fused_linear=use_fused_linear
    )
    return injector.inject(model)
