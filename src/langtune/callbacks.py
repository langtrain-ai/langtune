"""
callbacks.py: Training callbacks for Langtune

Provides extensible callback system for training hooks.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class. Override methods to customize training behavior."""
    
    def on_train_begin(self, trainer, **kwargs):
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer, batch_idx: int, **kwargs):
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float, **kwargs):
        """Called at the end of each batch."""
        pass
    
    def on_validation_begin(self, trainer, **kwargs):
        """Called at the start of validation."""
        pass
    
    def on_validation_end(self, trainer, metrics: Dict[str, float], **kwargs):
        """Called at the end of validation."""
        pass


class CallbackList:
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback):
        """Add a callback."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer, **kwargs):
        for cb in self.callbacks:
            cb.on_train_begin(trainer, **kwargs)
    
    def on_train_end(self, trainer, **kwargs):
        for cb in self.callbacks:
            cb.on_train_end(trainer, **kwargs)
    
    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, epoch, **kwargs)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics, **kwargs)
    
    def on_batch_begin(self, trainer, batch_idx: int, **kwargs):
        for cb in self.callbacks:
            cb.on_batch_begin(trainer, batch_idx, **kwargs)
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float, **kwargs):
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch_idx, loss, **kwargs)
    
    def on_validation_begin(self, trainer, **kwargs):
        for cb in self.callbacks:
            cb.on_validation_begin(trainer, **kwargs)
    
    def on_validation_end(self, trainer, metrics: Dict[str, float], **kwargs):
        for cb in self.callbacks:
            cb.on_validation_end(trainer, metrics, **kwargs)


class ProgressCallback(Callback):
    """Logs training progress."""
    
    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        self.batch_count = 0
    
    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        self.batch_count = 0
        logger.info(f"Starting epoch {epoch + 1}")
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float, **kwargs):
        self.batch_count += 1
        if self.batch_count % self.log_every == 0:
            logger.info(f"Batch {batch_idx}, Loss: {loss:.4f}")
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"Epoch {epoch + 1} completed - {metrics_str}")


class LearningRateMonitorCallback(Callback):
    """Monitor and log learning rate."""
    
    def __init__(self):
        self.lrs = []
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float, **kwargs):
        if hasattr(trainer, 'optimizer'):
            lr = trainer.optimizer.param_groups[0]['lr']
            self.lrs.append(lr)


class GradientMonitorCallback(Callback):
    """Monitor gradient statistics for debugging."""
    
    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self.step = 0
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float, **kwargs):
        self.step += 1
        if self.step % self.log_every == 0 and hasattr(trainer, 'model'):
            total_norm = 0.0
            param_count = 0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** 0.5
                logger.info(f"Step {self.step}: Gradient norm = {total_norm:.4f}")


class ModelSizeCallback(Callback):
    """Log model size information at training start."""
    
    def on_train_begin(self, trainer, **kwargs):
        if hasattr(trainer, 'model'):
            total = sum(p.numel() for p in trainer.model.parameters())
            trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            
            logger.info(f"Model size: {total:,} total params, {trainable:,} trainable ({100*trainable/total:.1f}%)")


class TimerCallback(Callback):
    """Track training time."""
    
    def __init__(self):
        self.train_start = None
        self.epoch_start = None
        self.epoch_times = []
    
    def on_train_begin(self, trainer, **kwargs):
        self.train_start = time.time()
    
    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        self.epoch_start = time.time()
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        if self.epoch_start:
            elapsed = time.time() - self.epoch_start
            self.epoch_times.append(elapsed)
            logger.info(f"Epoch {epoch + 1} took {elapsed:.1f}s")
    
    def on_train_end(self, trainer, **kwargs):
        if self.train_start:
            total = time.time() - self.train_start
            logger.info(f"Total training time: {total:.1f}s ({total/60:.1f}m)")


class SaveHistoryCallback(Callback):
    """Save training history to JSON."""
    
    def __init__(self, save_path: str = "training_history.json"):
        self.save_path = save_path
        self.history = {"epochs": [], "metrics": []}
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        self.history["epochs"].append(epoch + 1)
        self.history["metrics"].append(metrics)
        
        with open(self.save_path, 'w') as f:
            json.dump(self.history, f, indent=2)


class MemoryMonitorCallback(Callback):
    """Monitor GPU memory usage."""
    
    def __init__(self, log_every: int = 50):
        self.log_every = log_every
        self.step = 0
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float, **kwargs):
        import torch
        self.step += 1
        
        if self.step % self.log_every == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"Step {self.step}: GPU Memory - {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class WandbCallback(Callback):
    """Log to Weights & Biases."""
    
    def __init__(self, project: str = "langtune", run_name: Optional[str] = None):
        self.project = project
        self.run_name = run_name
        self.wandb = None
    
    def on_train_begin(self, trainer, **kwargs):
        try:
            import wandb
            wandb.init(project=self.project, name=self.run_name)
            self.wandb = wandb
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float, **kwargs):
        if self.wandb:
            self.wandb.log({"train_loss": loss})
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        if self.wandb:
            self.wandb.log({"epoch": epoch + 1, **metrics})
    
    def on_train_end(self, trainer, **kwargs):
        if self.wandb:
            self.wandb.finish()


# Default callback presets
def get_default_callbacks() -> CallbackList:
    """Get a list of recommended default callbacks."""
    return CallbackList([
        ModelSizeCallback(),
        ProgressCallback(log_every=10),
        TimerCallback(),
        LearningRateMonitorCallback()
    ])


def get_verbose_callbacks() -> CallbackList:
    """Get verbose callbacks for debugging."""
    return CallbackList([
        ModelSizeCallback(),
        ProgressCallback(log_every=1),
        TimerCallback(),
        LearningRateMonitorCallback(),
        GradientMonitorCallback(log_every=50),
        MemoryMonitorCallback(log_every=50)
    ])
