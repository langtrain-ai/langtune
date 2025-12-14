"""
trainer.py: Training utilities for Langtune
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, List
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import wandb
from contextlib import contextmanager

from .models import LoRALanguageModel
from .config import Config
from .data import DataCollator

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
        if window is None:
            return np.mean(values)
        else:
            return np.mean(values[-window:])
    
    def get_latest(self, key: str) -> float:
        """Get latest value of a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return self.metrics[key][-1]
    
    def log_epoch(self):
        """Log epoch metrics."""
        epoch_metrics = {}
        for key, values in self.metrics.items():
            epoch_metrics[f"epoch_{key}"] = np.mean(values)
        
        self.history.append(epoch_metrics)
        self.metrics = {}  # Reset for next epoch
        
        return epoch_metrics

class ModelCheckpoint:
    """Model checkpointing utility."""
    
    def __init__(
        self,
        save_dir: str,
        save_best_only: bool = True,
        save_total_limit: int = 3,
        monitor: str = "val_loss",
        mode: str = "min"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.save_total_limit = save_total_limit
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.checkpoints = []
        
    def save(self, model: nn.Module, optimizer, scheduler, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics
        }
        
        # Determine if this is the best checkpoint
        current_score = metrics.get(self.monitor, float('inf') if self.mode == "min" else float('-inf'))
        is_best = False
        
        if self.best_score is None:
            self.best_score = current_score
            is_best = True
        elif self.mode == "min" and current_score < self.best_score:
            self.best_score = current_score
            is_best = True
        elif self.mode == "max" and current_score > self.best_score:
            self.best_score = current_score
            is_best = True
        
        # Save checkpoint
        if not self.save_best_only or is_best:
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            self.checkpoints.append(checkpoint_path)
            
            if is_best:
                best_path = self.save_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                logger.info(f"New best model saved with {self.monitor}={current_score:.4f}")
        
        # Clean up old checkpoints
        if len(self.checkpoints) > self.save_total_limit:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
    
    def load(self, model: nn.Module, optimizer, scheduler, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint["epoch"], checkpoint["metrics"]

class Trainer:
    """
    Main trainer class for fine-tuning language models.
    """
    
    def __init__(
        self,
        model: LoRALanguageModel,
        config: Config,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup utilities
        self.metrics_tracker = MetricsTracker()
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            threshold=config.training.early_stopping_threshold
        )
        self.checkpointer = ModelCheckpoint(
            save_dir=config.output_dir,
            save_total_limit=config.training.save_total_limit,
            monitor="val_loss"
        )
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.training.mixed_precision else None
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(f"LoRA parameters: {self.model.count_lora_parameters():,}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        # Only optimize LoRA parameters if using LoRA
        if hasattr(self.model, 'count_lora_parameters') and self.model.count_lora_parameters() > 0:
            lora_params = []
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower():
                    lora_params.append(param)
            
            logger.info(f"Optimizing {len(lora_params)} LoRA parameter groups")
            return AdamW(lora_params, lr=self.config.training.learning_rate, weight_decay=self.config.training.weight_decay)
        else:
            return AdamW(self.model.parameters(), lr=self.config.training.learning_rate, weight_decay=self.config.training.weight_decay)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        total_steps = len(self.train_dataloader) * self.config.training.num_epochs
        
        if self.config.training.warmup_steps > 0:
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.training.warmup_steps / total_steps
            )
        else:
            return CosineAnnealingLR(self.optimizer, T_max=total_steps)
    
    def _setup_logging(self):
        """Setup logging and experiment tracking."""
        # Setup Weights & Biases if available
        try:
            wandb.init(
                project="langtune",
                config=self.config.__dict__ if hasattr(self.config, '__dict__') else {},
                name=f"run_{int(time.time())}"
            )
            self.use_wandb = True
        except:
            self.use_wandb = False
            logger.warning("Weights & Biases not available, using local logging only")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Logging
            if batch_idx % self.config.training.logging_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                metrics = {
                    "train_loss": loss.item(),
                    "learning_rate": current_lr,
                    "epoch": epoch
                }
                
                self.metrics_tracker.update(metrics)
                
                if self.use_wandb:
                    wandb.log(metrics)
                
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs["loss"]
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop."""
        start_epoch = 0
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            start_epoch, _ = self.checkpointer.load(
                self.model, self.optimizer, self.scheduler, resume_from_checkpoint
            )
            logger.info(f"Resumed training from epoch {start_epoch}")
        
        logger.info("Starting training...")
        
        for epoch in range(start_epoch, self.config.training.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Update metrics tracker
            self.metrics_tracker.update(all_metrics)
            
            # Log epoch metrics
            epoch_metrics = self.metrics_tracker.log_epoch()
            
            # Log to wandb
            if self.use_wandb:
                wandb.log(epoch_metrics)
            
            # Save checkpoint
            self.checkpointer.save(self.model, self.optimizer, self.scheduler, epoch, all_metrics)
            
            # Early stopping
            if val_metrics and "val_loss" in val_metrics:
                if self.early_stopping(val_metrics["val_loss"]):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Log epoch summary
            logger.info(f"Epoch {epoch+1} completed - Train Loss: {train_metrics['train_loss']:.4f}, Val Loss: {val_metrics.get('val_loss', 'N/A')}")
        
        logger.info("Training completed!")
        
        # Final evaluation on test set
        if self.test_dataloader:
            test_metrics = self.evaluate()
            logger.info(f"Final test metrics: {test_metrics}")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test set."""
        if self.test_dataloader is None:
            logger.warning("No test dataloader provided")
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Testing"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs["loss"]
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {"test_loss": avg_loss}
    
    def generate_sample(self, prompt: str, max_length: int = 100) -> str:
        """Generate a sample from the model."""
        self.model.eval()
        
        # Simple tokenization (in practice, you'd use a proper tokenizer)
        input_ids = torch.tensor([ord(c) for c in prompt[:50]], dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
        
    # Simple decoding
        generated_text = "".join([chr(i) for i in generated[0].cpu().tolist()])
        return generated_text


class FastTrainer:
    """
    Optimized trainer with:
    - Gradient accumulation for effective larger batches
    - Enhanced mixed precision training
    - Memory monitoring and optimization
    - Support for FastLoRALanguageModel
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        gradient_accumulation_steps: int = 4,
        mixed_precision: str = "fp16"  # fp16, bf16, or fp32
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Freeze base model if using FastLoRALanguageModel
        if hasattr(self.model, 'freeze_base_model'):
            self.model.freeze_base_model()
        
        # Setup mixed precision
        self.mixed_precision = mixed_precision
        self._setup_amp()
        
        # Setup optimizer (only trainable params)
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Utilities
        self.metrics_tracker = MetricsTracker()
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            threshold=config.training.early_stopping_threshold
        )
        self.checkpointer = ModelCheckpoint(
            save_dir=config.output_dir,
            save_total_limit=config.training.save_total_limit,
            monitor="val_loss"
        )
        
        # Setup logging
        self._setup_logging()
        
        # Log configuration
        self._log_training_info()
    
    def _setup_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)
    
    def _setup_amp(self):
        """Setup automatic mixed precision."""
        try:
            from .optimizations import MixedPrecisionTrainer
            
            if self.mixed_precision == "bf16":
                dtype = torch.bfloat16
            elif self.mixed_precision == "fp16":
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            self.amp_trainer = MixedPrecisionTrainer(
                enabled=(self.mixed_precision != "fp32" and self.device.type == "cuda"),
                dtype=dtype
            )
        except ImportError:
            # Fallback to standard scaler
            self.amp_trainer = None
            self.scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None
    
    def _setup_optimizer(self):
        """Setup optimizer for trainable parameters only."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if len(trainable_params) == 0:
            logger.warning("No trainable parameters found! Check model configuration.")
            trainable_params = list(self.model.parameters())
        
        logger.info(f"Optimizing {len(trainable_params)} parameter groups")
        
        return AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
    
    def _setup_scheduler(self):
        steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.training.num_epochs
        
        if self.config.training.warmup_steps > 0:
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.training.warmup_steps / max(total_steps, 1)
            )
        return CosineAnnealingLR(self.optimizer, T_max=total_steps)
    
    def _setup_logging(self):
        try:
            wandb.init(
                project="langtune-fast",
                config={
                    "gradient_accumulation": self.gradient_accumulation_steps,
                    "mixed_precision": self.mixed_precision,
                    **({k: v for k, v in self.config.__dict__.items() if not k.startswith('_')})
                },
                name=f"fast_run_{int(time.time())}"
            )
            self.use_wandb = True
        except:
            self.use_wandb = False
    
    def _log_training_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"FastTrainer initialized on {self.device}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        logger.info(f"Effective batch size: {self.config.training.batch_size * self.gradient_accumulation_steps}")
    
    def _log_memory(self, prefix: str = ""):
        if self.device.type == "cuda":
            try:
                from .optimizations import log_memory_usage
                log_memory_usage(prefix)
            except ImportError:
                allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(f"{prefix}GPU Memory: {allocated:.2f} GB")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_steps = 0
        accumulated_loss = 0.0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}",
            leave=False
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward with AMP
            if self.amp_trainer:
                with self.amp_trainer.autocast_context:
                    outputs = self.model(**batch)
                    loss = outputs["loss"] / self.gradient_accumulation_steps
                
                # Scale and backward
                scaled_loss = self.amp_trainer.scale_loss(loss)
                scaled_loss.backward()
            else:
                outputs = self.model(**batch)
                loss = outputs["loss"] / self.gradient_accumulation_steps
                loss.backward()
            
            accumulated_loss += loss.item()
            
            # Optimizer step after accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    if self.amp_trainer:
                        self.amp_trainer.unscale_gradients(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                
                # Step
                if self.amp_trainer:
                    self.amp_trainer.step(self.optimizer)
                else:
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                
                # Track
                step_loss = accumulated_loss * self.gradient_accumulation_steps
                total_loss += step_loss
                num_steps += 1
                accumulated_loss = 0.0
                
                progress_bar.set_postfix({"loss": f"{step_loss:.4f}"})
                
                # Log periodically
                if num_steps % (self.config.training.logging_steps // self.gradient_accumulation_steps + 1) == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    metrics = {"train_loss": step_loss, "learning_rate": lr, "epoch": epoch}
                    self.metrics_tracker.update(metrics)
                    if self.use_wandb:
                        wandb.log(metrics)
        
        # Handle remaining batches
        if accumulated_loss > 0:
            if self.amp_trainer:
                self.amp_trainer.step(self.optimizer)
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        self._log_memory(f"Epoch {epoch+1} end - ")
        
        return {"train_loss": total_loss / max(num_steps, 1)}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.amp_trainer:
                    with self.amp_trainer.autocast_context:
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
                
                total_loss += outputs["loss"].item()
                num_batches += 1
        
        return {"val_loss": total_loss / max(num_batches, 1)}
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        start_epoch = 0
        
        if resume_from_checkpoint:
            start_epoch, _ = self.checkpointer.load(
                self.model, self.optimizer, self.scheduler, resume_from_checkpoint
            )
            logger.info(f"Resumed from epoch {start_epoch}")
        
        logger.info("Starting optimized training...")
        self._log_memory("Training start - ")
        
        for epoch in range(start_epoch, self.config.training.num_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            all_metrics = {**train_metrics, **val_metrics}
            self.metrics_tracker.update(all_metrics)
            epoch_metrics = self.metrics_tracker.log_epoch()
            
            if self.use_wandb:
                wandb.log(epoch_metrics)
            
            self.checkpointer.save(self.model, self.optimizer, self.scheduler, epoch, all_metrics)
            
            if val_metrics and "val_loss" in val_metrics:
                if self.early_stopping(val_metrics["val_loss"]):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            logger.info(
                f"Epoch {epoch+1} - Train: {train_metrics['train_loss']:.4f}, "
                f"Val: {val_metrics.get('val_loss', 'N/A')}"
            )
        
        logger.info("Training completed!")
        self._log_memory("Training end - ")
        
        # Cleanup
        try:
            from .optimizations import cleanup_memory
            cleanup_memory()
        except ImportError:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


def create_trainer(
    config: Config,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    test_dataloader: Optional[DataLoader] = None
) -> Trainer:
    """
    Create a trainer instance.
    
    Args:
        config: Training configuration
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        test_dataloader: Test data loader (optional)
        
    Returns:
        Trainer instance
    """
    # Create model
    model = LoRALanguageModel(
        vocab_size=config.model.vocab_size,
        embed_dim=config.model.embed_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        max_seq_len=config.model.max_seq_len,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        lora_config=config.model.lora.__dict__ if config.model.lora else None
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )
    
    return trainer


def create_fast_trainer(
    config: Config,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    test_dataloader: Optional[DataLoader] = None,
    gradient_accumulation_steps: int = 4,
    mixed_precision: str = "fp16"
) -> FastTrainer:
    """
    Create an optimized FastTrainer instance with FastLoRALanguageModel.
    
    Args:
        config: Training configuration
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        test_dataloader: Test data loader (optional)
        gradient_accumulation_steps: Steps to accumulate gradients
        mixed_precision: "fp16", "bf16", or "fp32"
        
    Returns:
        FastTrainer instance
    """
    from .models import FastLoRALanguageModel
    
    # Create optimized model
    model = FastLoRALanguageModel(
        vocab_size=config.model.vocab_size,
        embed_dim=config.model.embed_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        max_seq_len=config.model.max_seq_len,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        lora_config=config.model.lora.__dict__ if config.model.lora else None,
        use_rope=True,
        use_flash_attention=True,
        use_gradient_checkpointing=True
    )
    
    # Create fast trainer
    trainer = FastTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision
    )
    
    return trainer

