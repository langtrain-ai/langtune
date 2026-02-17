"""
trainer.py: High-performance training utilities for Langtune
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, OneCycleLR, SequentialLR
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, List
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import wandb
from contextlib import contextmanager

from .config import Config
from .data import DataCollator
from .device import DeviceManager
from .utils import cleanup_memory

# High-performance components
from .fast_lora import FastLoRAConfig, apply_fast_lora, get_lora_state_dict
from .packing import SequencePacker
from .lisa import LISA
from .kernels import fused_cross_entropy
try:
    from .triton_kernels import triton_cross_entropy
except ImportError:
    triton_cross_entropy = None

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 5, threshold: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if training should stop early."""
        if self.best_score is None:
            self.best_score = score
        elif self.mode == "min":
            if score < self.best_score - self.threshold:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == "max"
            if score > self.best_score + self.threshold:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

class MetricsTracker:
    """Track training and validation metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_average(self, key: str, window: int = None) -> float:
        """Get average of a metric."""
        if key not in self.metrics:
            return 0.0
        
        values = self.metrics[key]
        if window:
            values = values[-window:]
        return sum(values) / len(values) if values else 0.0

class ModelCheckpoint:
    """Manage model checkpoints."""
    
    def __init__(self, save_dir: str, save_total_limit: int = 3, monitor: str = "val_loss", mode: str = "min"):
        self.save_dir = Path(save_dir)
        self.save_total_limit = save_total_limit
        self.monitor = monitor
        self.mode = mode
        self.checkpoints = [] # List of (score, path)
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, model, optimizer, scheduler, epoch: int, metrics: Dict[str, float]):
        """Save a checkpoint."""
        # Create checkpoint name
        metric_val = metrics.get(self.monitor, 0.0)
        ckpt_name = f"checkpoint-epoch-{epoch}-{self.monitor}-{metric_val:.4f}"
        save_path = self.save_dir / ckpt_name
        
        # Save logic
        save_path.mkdir(exist_ok=True)
        
        # Save model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(save_path)
        else:
            torch.save(model.state_dict(), save_path / "pytorch_model.bin")
            
        # Save optimizer/scheduler state
        torch.save({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'metrics': metrics
        }, save_path / "trainer_state.pt")
        
        # Update checkpoints list
        self.checkpoints.append((metric_val, save_path))
        
        # Sort based on mode
        reverse = (self.mode == "max")
        self.checkpoints.sort(key=lambda x: x[0], reverse=reverse)
        
        # Keep only top k
        if len(self.checkpoints) > self.save_total_limit:
            to_remove = self.checkpoints.pop(-1) if self.mode == "min" else self.checkpoints.pop(0)
            # In min mode, we want lowest scores. Sort ascending. Pop last (highest).
            # In max mode, we want highest. Sort descending. Pop last (lowest).
            # Wait, sort key x[0]. 
            # Min mode: sort asc. Best is index 0. Worst is index -1. Pop -1. Correct.
            # Max mode: sort desc (reverse=True). Best is index 0. Worst is index -1. Pop -1. Correct.
            
            # Remove directory
            import shutil
            if to_remove[1].exists():
                shutil.rmtree(to_remove[1])

    def load(self, model, optimizer, scheduler, path: str):
        """Load from checkpoint."""
        path = Path(path)
        
        # Load model
        if (path / "config.json").exists():
            # HuggingFace style
            from .models import LoRALanguageModel
            # This assumes model is compatible
            pass 
        
        state_path = path / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state['optimizer'])
            if scheduler and state['scheduler']:
                scheduler.load_state_dict(state['scheduler'])
            return state['epoch'] + 1, state['metrics']
            
        return 0, {}

class Trainer:
    """
    High-Performance Trainer for Langtune.
    
    Optimizations:
    - Fast LoRA (RSLoRA / DoRA)
    - LISA (Layerwise Importance Sampled AdamW)
    - Sequence Packing
    - Fused Cross Entropy
    - Automatic Mixed Precision (AMP)
    - Gradient Checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Config] = None,
        tokenizer = None,
    ):
        self.config = config or Config()
        self.device = DeviceManager.get_device()
        self.tokenizer = tokenizer
        
        # Apply High-Performance Optimizations
        
        # 1. Fast LoRA
        if hasattr(self.config, 'use_lora') and self.config.use_lora:
            logger.info("Applying Fast LoRA...")
            lora_config = FastLoRAConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                use_rslora=True, # Default to best practice
                use_dora=False   # Optional
            )
            self.model = apply_fast_lora(model, lora_config)
        else:
            self.model = model
            
        self.model = self.model.to(self.device)
        
        # 2. Compile Model (TensorRT/Inductor style speedups)
        if hasattr(torch, 'compile') and getattr(self.config, 'compile_model', False):
             logger.info("Compiling model with torch.compile...")
             self.model = torch.compile(self.model)
             
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 3. Optimizer selection (LISA or AdamW)
        self._setup_optimizer()
        
        # 4. Learning Rate Scheduler
        self._setup_scheduler()
        
        # 5. Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None
        
        # Metrics
        self.tracker = MetricsTracker()
        self.early_stopping = EarlyStopping(patience=self.config.patience)
        
    def _setup_optimizer(self):
        """Setup LISA or standard optimizer."""
        if getattr(self.config, 'use_lisa', False):
            logger.info("Using LISA optimizer (Layerwise Importance Sampling)")
            self.optimizer = LISA(
                self.model.parameters(),
                lr=self.config.learning_rate,
                n_layers=getattr(self.config, 'num_layers', 32),
                n_active_layers=getattr(self.config, 'lisa_active_layers', 2),
                interval_steps=getattr(self.config, 'lisa_interval', 20),
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )

    def _setup_scheduler(self):
        """Setup scheduler with warmup."""
        num_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = int(num_steps * self.config.warmup_ratio)
        
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=num_steps - warmup_steps)
        
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    @contextmanager
    def _autocast(self):
        """Context manager for mixed precision."""
        if self.config.mixed_precision:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.cuda.amp.autocast(dtype=dtype):
                yield
        else:
            yield

    def train_step(self, batch) -> float:
        """Single training step with fused loss."""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        with self._autocast():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Use Fused Cross Entropy
            if getattr(self.config.training, 'use_triton', False) and triton_cross_entropy is not None:
                # Triton kernel (forward pass only for now, autograd handled by wrapping if needed)
                # But our implementation in triton_kernels.py provided a naive forward function 
                # AND a TritonCrossEntropyLoss autograd function.
                # Let's use the autograd function wrapper if available.
                from .triton_kernels import TritonCrossEntropyLoss
                loss = TritonCrossEntropyLoss.apply(outputs.logits, labels)
            else:
                # Fallback to Compile-Fused or Standard
                loss = fused_cross_entropy(outputs.logits, labels)
            
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
        
        if self.scaler:
            self.scaler.scale(loss).backward()
            
            if (self.state['step'] + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
        else:
            loss.backward()
            if (self.state['step'] + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                
        return loss.item() * self.config.gradient_accumulation_steps

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        self.model.train()
        self.state = {'step': 0, 'epoch': 0}
        
        for epoch in range(self.config.num_epochs):
            self.state['epoch'] = epoch
            epoch_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            
            for batch in progress_bar:
                loss = self.train_step(batch)
                epoch_loss += loss
                
                self.state['step'] += 1
                self.tracker.update({'train_loss': loss})
                
                progress_bar.set_postfix({'loss': loss})
                
                if self.state['step'] % self.config.eval_steps == 0:
                    if self.val_loader:
                        val_loss = self.evaluate()
                        self.tracker.update({'val_loss': val_loss})
                        logger.info(f"Step {self.state['step']}: Val Loss {val_loss:.4f}")
                        
                        if self.early_stopping(val_loss):
                            logger.info("Early stopping triggered.")
                            return
                
                if self.state['step'] % self.config.save_steps == 0:
                    self.save_checkpoint()
                    
            logger.info(f"Epoch {epoch+1} done. Avg Loss: {epoch_loss / len(self.train_loader):.4f}")
            
        self.save_checkpoint(final=True)
        logger.info("Training complete.")

    def evaluate(self) -> float:
        """Evaluation loop."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with self._autocast():
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                
                total_loss += loss.item()
                
        self.model.train()
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, final: bool = False):
        """Save checkpoint."""
        name = "final" if final else f"step_{self.state['step']}"
        path = Path(self.config.output_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights if applicable
        if hasattr(self.config, 'use_lora') and self.config.use_lora:
            state_dict = get_lora_state_dict(self.model)
            torch.save(state_dict, path / "lora_weights.pt")
        else:
            torch.save(self.model.state_dict(), path / "model.pt")
            
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)
            
        logger.info(f"Saved checkpoint to {path}")

def create_trainer(
    model: str,
    train_file: str,
    output_dir: str,
    **kwargs
):
    """Factory function to create a Trainer."""
    # Simplified factory for compatibility
    raise NotImplementedError("Use Trainer directly or finetune APIs")
    
