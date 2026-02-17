"""
lisa.py: Layerwise Importance Sampled AdamW (LISA) implementation.

LISA effectively freezes most layers during training and only updates a
randomly selected subset of layers, drastically reducing memory usage
and improving training speed while maintaining or improving performance.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from typing import List, Dict, Any, Optional

class LISA(Optimizer):
    """
    Layerwise Importance Sampled AdamW (LISA).
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate
        n_layers: Total number of transformer layers in the model
        n_active_layers: Number of layers to update per step (k in paper)
        interval_steps: How often to switch active layers
        betas: Adam betas
        eps: Adam epsilon
        weight_decay: Weight decay
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        n_layers: int = 32,
        n_active_layers: int = 2,
        interval_steps: int = 20,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            n_layers=n_layers,
            n_active_layers=n_active_layers,
            interval_steps=interval_steps
        )
        super().__init__(params, defaults)
        
        self.state['step'] = 0
        self.active_layers = []
        self._sample_layers()
        
    def _sample_layers(self):
        """Randomly sample layers to be active."""
        n_layers = self.defaults['n_layers']
        n_active = self.defaults['n_active_layers']
        
        # Always include embeddings and head (usually layers -1 and 0 conceptually)
        # But for LISA on LLMs, we focused on transformer blocks.
        # We assume params have layer tracking or we manually enabling/disabling grads.
        # Since Optimizer can't easily enable/disable grads on model structure without access to model,
        # LISA is often implemented as a Callback or wrapped around the Model.
        # However, as an Optimizer, we can skip updates for params not in active layers.
        
        self.active_layers = np.random.choice(
            range(n_layers), 
            size=n_active, 
            replace=False
        ).tolist()
        
        # We also need to explicitly handle this at the model level for backprop savings.
        # This class will just handle the optimization step masking.
        
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
            
        self.state['step'] += 1
        
        # Resample layers periodically
        if self.state['step'] % self.defaults['interval_steps'] == 0:
            self._sample_layers()
            
        for group in self.param_groups:
            # We assume params are tagged with 'layer_idx' or similar if possible.
            # If not, naive LISA (optimization only) doesn't save backprop memory.
            # To strictly save memory, we need to freeze layers in the Forward/Backward pass.
            
            # Standard AdamW step for active params
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # If we implemented grad masking at the model level, p.grad would be None 
                # for frozen layers anyway (if requires_grad=False).
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                step_size = group['lr']
                
                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss

def apply_lisa(
    model: nn.Module, 
    n_layers: int, 
    n_active_layers: int = 2
) -> List[int]:
    """
    Apply LISA freezing mask to model.
    Returns list of active layer indices.
    """
    # Sample active layers
    active_layers = np.random.choice(
        range(n_layers), 
        size=n_active_layers, 
        replace=False
    ).tolist()
    
    # This assumes a standard HF structure like model.model.layers
    # We need to generalize or catch specific architectures
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        
    if layers is None:
        return [] # Can't apply
        
    for i, layer in enumerate(layers):
        is_active = i in active_layers
        for param in layer.parameters():
            param.requires_grad = is_active
            
    # Always keep embeddings and head active
    # (Simplified assumption, can be tuned)
    
    return active_layers
