"""
schedulers.py: Learning rate schedulers for Langtune

Provides various learning rate scheduling strategies.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class WarmupScheduler(_LRScheduler):
    """Linear warmup scheduler."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        base_lr: float = None,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / max(self.warmup_steps, 1)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.base_lrs


class CosineAnnealingWithWarmup(_LRScheduler):
    """Cosine annealing with linear warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(self.warmup_steps, 1)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class LinearDecayWithWarmup(_LRScheduler):
    """Linear decay with warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(self.warmup_steps, 1)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            return [
                max(self.min_lr, base_lr * (1 - progress))
                for base_lr in self.base_lrs
            ]


class PolynomialDecayWithWarmup(_LRScheduler):
    """Polynomial decay with warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        power: float = 2.0,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / max(self.warmup_steps, 1)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            decay_factor = (1 - progress) ** self.power
            return [
                max(self.min_lr, base_lr * decay_factor)
                for base_lr in self.base_lrs
            ]


class ConstantWithWarmup(_LRScheduler):
    """Constant learning rate with warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / max(self.warmup_steps, 1)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.base_lrs


class OneCycleLRWithWarmup(_LRScheduler):
    """1cycle LR policy with customizable warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        last_epoch: int = -1
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        # Calculate phase boundaries
        self.warmup_steps = int(total_steps * pct_start)
        self.cooldown_steps = total_steps - self.warmup_steps
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase
            progress = self.last_epoch / max(self.warmup_steps, 1)
            start_lr = self.max_lr / self.div_factor
            return [start_lr + (self.max_lr - start_lr) * progress for _ in self.base_lrs]
        else:
            # Cooldown phase: cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(self.cooldown_steps, 1)
            end_lr = self.max_lr / self.final_div_factor
            return [
                end_lr + 0.5 * (self.max_lr - end_lr) * (1 + math.cos(math.pi * progress))
                for _ in self.base_lrs
            ]


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    **kwargs
) -> _LRScheduler:
    """
    Get a scheduler by name.
    
    Args:
        name: Scheduler name ('cosine', 'linear', 'constant', 'polynomial', 'onecycle')
        optimizer: Optimizer
        total_steps: Total training steps
        warmup_steps: Warmup steps
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler
    """
    schedulers = {
        "cosine": CosineAnnealingWithWarmup,
        "linear": LinearDecayWithWarmup,
        "constant": ConstantWithWarmup,
        "polynomial": PolynomialDecayWithWarmup,
        "warmup": WarmupScheduler,
    }
    
    if name == "onecycle":
        max_lr = kwargs.get("max_lr", optimizer.param_groups[0]['lr'])
        return OneCycleLRWithWarmup(
            optimizer, max_lr, total_steps,
            pct_start=warmup_steps / total_steps if total_steps > 0 else 0.3,
            **{k: v for k, v in kwargs.items() if k != "max_lr"}
        )
    
    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Options: {list(schedulers.keys())}")
    
    scheduler_cls = schedulers[name]
    
    if name == "warmup":
        return scheduler_cls(optimizer, warmup_steps, **kwargs)
    elif name == "constant":
        return scheduler_cls(optimizer, warmup_steps, **kwargs)
    else:
        return scheduler_cls(optimizer, total_steps, warmup_steps, **kwargs)
