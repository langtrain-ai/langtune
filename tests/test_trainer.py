"""
Tests for Langtune trainer classes.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


class TestTrainer:
    """Tests for Trainer class."""
    
    def create_dataloader(self, config, device, num_samples=10):
        """Create a simple dataloader for testing."""
        input_ids = torch.randint(0, config["vocab_size"], (num_samples, 32))
        labels = input_ids.clone()
        
        dataset = TensorDataset(input_ids, labels)
        
        def collate_fn(batch):
            input_ids = torch.stack([b[0] for b in batch])
            labels = torch.stack([b[1] for b in batch])
            return {"input_ids": input_ids, "labels": labels}
        
        return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    def test_trainer_creation(self, small_config, lora_config, device):
        """Test creating a Trainer."""
        from langtune import LoRALanguageModel, Config
        from langtune.trainer import Trainer
        
        model = LoRALanguageModel(
            vocab_size=small_config["vocab_size"],
            embed_dim=small_config["embed_dim"],
            num_layers=small_config["num_layers"],
            num_heads=small_config["num_heads"],
            lora_config=lora_config
        )
        
        dataloader = self.create_dataloader(small_config, device)
        
        config = Config.from_preset("tiny")
        config.training.num_epochs = 1
        
        trainer = Trainer(model, config, dataloader)
        
        assert trainer.model is not None
        assert trainer.optimizer is not None


class TestEarlyStopping:
    """Tests for EarlyStopping utility."""
    
    def test_early_stopping_improvement(self):
        """Test early stopping with improving loss."""
        from langtune import EarlyStopping
        
        es = EarlyStopping(patience=3, mode="min")
        
        assert es(1.0) == False  # First score
        assert es(0.9) == False  # Improvement
        assert es(0.8) == False  # Improvement
    
    def test_early_stopping_triggered(self):
        """Test early stopping triggers after patience."""
        from langtune import EarlyStopping
        
        es = EarlyStopping(patience=2, mode="min")
        
        es(1.0)  # Best
        es(1.1)  # Worse, counter=1
        es(1.2)  # Worse, counter=2
        
        assert es.early_stop == True


class TestMetricsTracker:
    """Tests for MetricsTracker utility."""
    
    def test_metrics_update(self):
        """Test updating metrics."""
        from langtune import MetricsTracker
        
        tracker = MetricsTracker()
        
        tracker.update({"loss": 1.0})
        tracker.update({"loss": 0.9})
        tracker.update({"loss": 0.8})
        
        avg = tracker.get_average("loss")
        assert abs(avg - 0.9) < 0.01
    
    def test_metrics_latest(self):
        """Test getting latest metric."""
        from langtune import MetricsTracker
        
        tracker = MetricsTracker()
        
        tracker.update({"loss": 1.0})
        tracker.update({"loss": 0.5})
        
        assert tracker.get_latest("loss") == 0.5
