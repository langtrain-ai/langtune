"""
Tests for Langtune model classes.
"""

import pytest
import torch


class TestLoRALinear:
    """Tests for LoRALinear layer."""
    
    def test_lora_linear_creation(self):
        """Test creating a LoRALinear layer."""
        from langtune import LoRALinear
        
        layer = LoRALinear(
            in_features=64,
            out_features=64,
            rank=8,
            alpha=16.0,
            dropout=0.1
        )
        
        assert layer.in_features == 64
        assert layer.out_features == 64
        assert layer.rank == 8
    
    def test_lora_linear_forward(self, device):
        """Test LoRALinear forward pass."""
        from langtune import LoRALinear
        
        layer = LoRALinear(64, 64, rank=8).to(device)
        x = torch.randn(2, 32, 64, device=device)
        
        output = layer(x)
        
        assert output.shape == x.shape
    
    def test_lora_linear_parameters(self):
        """Test LoRALinear parameter count."""
        from langtune import LoRALinear
        
        layer = LoRALinear(64, 64, rank=8)
        
        # LoRA A: rank x in_features = 8 x 64 = 512
        # LoRA B: out_features x rank = 64 x 8 = 512
        assert layer.lora_A.numel() == 8 * 64
        assert layer.lora_B.numel() == 64 * 8


class TestLoRALanguageModel:
    """Tests for LoRALanguageModel."""
    
    def test_model_creation(self, small_config, lora_config):
        """Test creating a LoRALanguageModel."""
        from langtune import LoRALanguageModel
        
        model = LoRALanguageModel(
            vocab_size=small_config["vocab_size"],
            embed_dim=small_config["embed_dim"],
            num_layers=small_config["num_layers"],
            num_heads=small_config["num_heads"],
            lora_config=lora_config
        )
        
        assert model.vocab_size == small_config["vocab_size"]
        assert model.count_parameters() > 0
    
    def test_model_forward(self, small_config, lora_config, sample_batch, device):
        """Test LoRALanguageModel forward pass."""
        from langtune import LoRALanguageModel
        
        model = LoRALanguageModel(
            vocab_size=small_config["vocab_size"],
            embed_dim=small_config["embed_dim"],
            num_layers=small_config["num_layers"],
            num_heads=small_config["num_heads"],
            lora_config=lora_config
        ).to(device)
        
        outputs = model(**sample_batch)
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape[-1] == small_config["vocab_size"]
    
    def test_model_generate(self, small_config, device):
        """Test LoRALanguageModel text generation."""
        from langtune import LoRALanguageModel
        
        model = LoRALanguageModel(
            vocab_size=small_config["vocab_size"],
            embed_dim=small_config["embed_dim"],
            num_layers=small_config["num_layers"],
            num_heads=small_config["num_heads"]
        ).to(device)
        
        input_ids = torch.randint(0, small_config["vocab_size"], (1, 10), device=device)
        
        generated = model.generate(input_ids, max_length=20)
        
        assert generated.shape[1] <= 20
        assert generated.shape[1] >= input_ids.shape[1]


class TestFastLoRALanguageModel:
    """Tests for FastLoRALanguageModel."""
    
    def test_fast_model_creation(self, small_config, lora_config):
        """Test creating a FastLoRALanguageModel."""
        from langtune import FastLoRALanguageModel
        
        model = FastLoRALanguageModel(
            vocab_size=small_config["vocab_size"],
            embed_dim=small_config["embed_dim"],
            num_layers=small_config["num_layers"],
            num_heads=small_config["num_heads"],
            lora_config=lora_config,
            use_rope=True,
            use_gradient_checkpointing=True
        )
        
        assert model.vocab_size == small_config["vocab_size"]
        assert model.use_rope == True
    
    def test_fast_model_forward(self, small_config, lora_config, sample_batch, device):
        """Test FastLoRALanguageModel forward pass."""
        from langtune import FastLoRALanguageModel
        
        model = FastLoRALanguageModel(
            vocab_size=small_config["vocab_size"],
            embed_dim=small_config["embed_dim"],
            num_layers=small_config["num_layers"],
            num_heads=small_config["num_heads"],
            lora_config=lora_config
        ).to(device)
        
        outputs = model(**sample_batch)
        
        assert "logits" in outputs
        assert "loss" in outputs
    
    def test_freeze_base_model(self, small_config, lora_config):
        """Test freezing base model parameters."""
        from langtune import FastLoRALanguageModel
        
        model = FastLoRALanguageModel(
            vocab_size=small_config["vocab_size"],
            embed_dim=small_config["embed_dim"],
            num_layers=small_config["num_layers"],
            num_heads=small_config["num_heads"],
            lora_config=lora_config
        )
        
        total_before = sum(p.requires_grad for p in model.parameters())
        model.freeze_base_model()
        trainable_after = sum(p.requires_grad for p in model.parameters())
        
        # Only LoRA parameters should be trainable
        assert trainable_after < total_before
