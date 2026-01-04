"""
Adaptive LoRA Configuration Module

Dynamically determines optimal LoRA configuration by probing gradient density
across model layers. Applies LoRA only where learning actually happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class AdaptiveLoRAConfig:
    """
    Adaptive LoRA configuration determined by gradient probing.
    
    Unlike static configs, this is computed based on actual model behavior.
    """
    
    # Target modules to apply LoRA (dynamically selected)
    target_modules: List[str] = field(default_factory=list)
    
    # Rank (dynamically computed)
    lora_r: int = 16
    
    # Alpha (dynamically computed)
    lora_alpha: int = 32
    
    # Dropout
    lora_dropout: float = 0.05
    
    # Gradient density scores per layer
    layer_scores: Dict[str, float] = field(default_factory=dict)
    
    # VRAM estimate (GB)
    estimated_vram: float = 0.0
    
    # Confidence in this config
    confidence: float = 0.0
    
    def to_peft_config(self) -> Dict[str, Any]:
        """Convert to PEFT LoraConfig format."""
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
    
    def __repr__(self) -> str:
        return (
            f"AdaptiveLoRAConfig(r={self.lora_r}, alpha={self.lora_alpha}, "
            f"modules={len(self.target_modules)}, vram={self.estimated_vram:.1f}GB, "
            f"confidence={self.confidence:.2f})"
        )


class GradientProbe:
    """
    Probes model to measure gradient density per layer.
    
    This helps determine which layers are actively learning and
    should receive LoRA adapters.
    """
    
    def __init__(
        self,
        gradient_threshold: float = 0.1,
        min_active_layers: int = 4,
        max_active_layers: int = 32,
    ):
        self.gradient_threshold = gradient_threshold
        self.min_active_layers = min_active_layers
        self.max_active_layers = max_active_layers
        self._layer_gradients: Dict[str, List[float]] = {}
    
    def probe(
        self,
        model: Any,
        sample_batch: Any,
        num_steps: int = 5,
    ) -> Dict[str, float]:
        """
        Probe model with sample batch to measure gradient density.
        
        Args:
            model: PyTorch model
            sample_batch: Small batch of training data
            num_steps: Number of forward-backward passes
            
        Returns:
            Dict mapping layer names to gradient density scores
        """
        import torch
        
        self._layer_gradients = {}
        
        # Register hooks to capture gradients
        hooks = []
        for name, module in model.named_modules():
            if self._is_lora_candidate(name, module):
                hook = module.register_full_backward_hook(
                    lambda m, gi, go, n=name: self._capture_gradient(n, go)
                )
                hooks.append(hook)
        
        # Run probe steps
        model.train()
        for _ in range(num_steps):
            try:
                outputs = model(**sample_batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                loss.backward()
            except Exception:
                pass  # Skip failed batches
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute density scores
        scores = {}
        for name, grads in self._layer_gradients.items():
            if grads:
                # Density = mean of gradient magnitudes
                mean_grad = sum(grads) / len(grads)
                scores[name] = mean_grad
        
        # Normalize scores
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def _capture_gradient(self, name: str, grad_output: Tuple) -> None:
        """Capture gradient magnitude for a layer."""
        import torch
        
        if name not in self._layer_gradients:
            self._layer_gradients[name] = []
        
        for grad in grad_output:
            if grad is not None and isinstance(grad, torch.Tensor):
                magnitude = grad.abs().mean().item()
                self._layer_gradients[name].append(magnitude)
    
    def _is_lora_candidate(self, name: str, module: Any) -> bool:
        """Check if module is a LoRA candidate (Linear, Attention, etc.)."""
        import torch.nn as nn
        
        if isinstance(module, nn.Linear):
            # Filter for attention and MLP layers
            lora_keys = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                        'gate_proj', 'up_proj', 'down_proj',
                        'query', 'key', 'value', 'dense']
            return any(key in name.lower() for key in lora_keys)
        return False


def probe_model_for_lora(
    model: Any,
    sample_batch: Any,
    vram_budget_gb: float = 8.0,
    aggressive: bool = False,
) -> AdaptiveLoRAConfig:
    """
    Probe model and generate optimal LoRA configuration.
    
    Args:
        model: The model to probe
        sample_batch: Small batch of training data
        vram_budget_gb: VRAM budget in GB
        aggressive: If True, be more aggressive with LoRA (lower threshold)
        
    Returns:
        AdaptiveLoRAConfig with optimal settings
    """
    probe = GradientProbe(
        gradient_threshold=0.05 if aggressive else 0.1,
    )
    
    # Probe for gradient density
    layer_scores = probe.probe(model, sample_batch)
    
    # Select active layers (above threshold)
    threshold = 0.05 if aggressive else 0.1
    active_layers = [
        name for name, score in layer_scores.items()
        if score >= threshold
    ]
    
    # Ensure minimum coverage
    if len(active_layers) < probe.min_active_layers:
        # Add top layers by score
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        active_layers = [name for name, _ in sorted_layers[:probe.min_active_layers]]
    
    # Cap at maximum
    if len(active_layers) > probe.max_active_layers:
        sorted_layers = sorted(
            [(name, layer_scores[name]) for name in active_layers],
            key=lambda x: x[1],
            reverse=True
        )
        active_layers = [name for name, _ in sorted_layers[:probe.max_active_layers]]
    
    # Compute optimal rank based on gradient variance
    avg_score = sum(layer_scores.get(l, 0) for l in active_layers) / len(active_layers) if active_layers else 0.5
    
    # Higher gradient activity -> higher rank needed
    if avg_score > 0.7:
        optimal_r = 32
    elif avg_score > 0.4:
        optimal_r = 16
    else:
        optimal_r = 8
    
    # Adjust for VRAM budget
    param_estimate = len(active_layers) * optimal_r * 2  # Rough estimate in millions
    vram_estimate = param_estimate * 0.004  # ~4MB per million params in fp16
    
    if vram_estimate > vram_budget_gb:
        # Reduce rank to fit budget
        scale_factor = vram_budget_gb / vram_estimate
        optimal_r = max(4, int(optimal_r * scale_factor))
    
    # Alpha is typically 2x rank
    optimal_alpha = optimal_r * 2
    
    # Compute confidence
    if layer_scores:
        variance = sum((s - avg_score) ** 2 for s in layer_scores.values()) / len(layer_scores)
        confidence = 1 - min(1, variance)  # Lower variance = higher confidence
    else:
        confidence = 0.5
    
    return AdaptiveLoRAConfig(
        target_modules=active_layers,
        lora_r=optimal_r,
        lora_alpha=optimal_alpha,
        layer_scores=layer_scores,
        estimated_vram=vram_estimate,
        confidence=confidence,
    )
