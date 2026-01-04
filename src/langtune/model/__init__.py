"""
Langtune Model Loading Subsystem.

Provides high-performance loading primitives:
- HubResolver: Cached HF downloads
- TensorStreamer: Lazy safetensors loading
- PriorityTensorStreamer: Layer-priority loading for optimal VRAM
- ModelLoader: Orchestration
- KernelInjector: CUDA kernel replacement
"""

from .hub import HubResolver
from .safetensors import TensorStreamer, PriorityTensorStreamer, LAYER_PRIORITY
from .weights import WeightLoader
from .loader import ModelLoader
from .kernels import KernelInjector, inject_kernels

__all__ = [
    "HubResolver",
    "TensorStreamer",
    "PriorityTensorStreamer",
    "LAYER_PRIORITY",
    "WeightLoader",
    "ModelLoader",
    "KernelInjector",
    "inject_kernels"
]
