"""
Model Loader.

Orchestrates the loading pipeline:
1. Hub Resolution
2. Architecture Construction
3. Streaming Weight Loading
4. Kernel Injection
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
import logging

from .hub import HubResolver, get_resolver
from .safetensors import TensorStreamer
from .weights import WeightLoader

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Main entry point for loading models with high performance.
    """
    
    def __init__(
        self, 
        offline: bool = False,
        cache_dir: Optional[str] = None
    ):
        self.resolver = get_resolver(offline=offline, cache_dir=cache_dir)
        
    def load(
        self,
        model_id: str,
        quantization: Optional[str] = None, # "nf4", "fp4"
        dtype: str = "bf16",
        device: str = "cuda",
        planner_config: Optional[Dict[str, Any]] = None,
        inject_kernels: bool = True
    ) -> nn.Module:
        """
        Load a model with the optimized pipeline.
        
        Args:
            model_id: HuggingFace model ID or local path
            quantization: Quantization mode ("nf4", "fp4", None)
            dtype: Compute dtype ("bf16", "fp16")
            device: Target device
            planner_config: Optional config from TrainingPlanner for selective loading
            inject_kernels: Whether to inject optimized CUDA kernels
        """
        # 1. Resolve
        model_path = self.resolver.resolve(model_id)
        
        # 2. Config & Architecture
        # For now, we rely on HF config to build the skeleton
        # In the future, we will use our "architectures" registry
        from transformers import AutoConfig, AutoModelForCausalLM
        
        config = AutoConfig.from_pretrained(model_path)
        
        logger.info(f"Building model skeleton for {model_id}...")
        # Creation on 'meta' device avoids memory allocation!
        with torch.device("meta"):
             model = AutoModelForCausalLM.from_config(config)
             
        # Move empty shell to CPU/device?
        # Meta tensors can't be used directly for loading usually without `to_empty`
        # Using `to_empty` moves to device but allocating memory
        model = model.to_empty(device=device)
        
        # 3. Stream Weights with Priority
        logger.info("Streaming weights with priority ordering...")
        from .safetensors import PriorityTensorStreamer
        streamer = PriorityTensorStreamer(model_path)
        
        # 4. Load & Quantize (with planner-aware skipping)
        compute_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        loader = WeightLoader(streamer, quantization_mode=quantization, compute_dtype=compute_dtype, device=device)
        
        # Determine layers to skip based on planner config
        skip_patterns = []
        if planner_config:
            skip_patterns = planner_config.get("frozen_layer_patterns", [])
            logger.info(f"Planner config: Skipping patterns {skip_patterns}")
        
        loader.load_into_module(model, skip_patterns=skip_patterns)
        
        # 5. Kernel Injection
        if inject_kernels:
            logger.info("Injecting optimized kernels...")
            self._inject_kernels(model)
        
        return model

    def _inject_kernels(self, model: nn.Module):
        """
        Replace layers with Langtrain Custom Kernels.
        """
        from .kernels import inject_kernels
        return inject_kernels(model, use_flash_attention=True, use_fused_linear=True)
