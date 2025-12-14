"""
Tests for Langtune optimization utilities.
"""

import pytest
import torch


class TestQuantization:
    """Tests for quantization utilities."""
    
    def test_quantize_dequantize(self):
        """Test 4-bit quantization round-trip."""
        from langtune.optimizations import quantize_tensor_4bit, dequantize_tensor_4bit
        
        # Create a tensor
        original = torch.randn(256, 64)
        
        # Quantize
        packed, scales, zeros = quantize_tensor_4bit(original, group_size=64)
        
        # Dequantize
        recovered = dequantize_tensor_4bit(packed, scales, zeros, group_size=64, output_shape=(256, 64))
        
        # Should be approximately equal
        assert recovered.shape == original.shape
        # Check error is reasonable (< 10% of std)
        error = (original - recovered).abs().mean()
        assert error < original.std() * 0.2
    
    def test_quantized_linear(self, device):
        """Test QuantizedLinear layer."""
        from langtune.optimizations import QuantizedLinear
        
        layer = QuantizedLinear(64, 64, group_size=64).to(device)
        
        # Initialize from weight
        weight = torch.randn(64, 64, device=device)
        layer.initialize_from_weight(weight)
        
        # Forward pass
        x = torch.randn(2, 32, 64, device=device)
        output = layer(x)
        
        assert output.shape == x.shape


class TestRoPE:
    """Tests for Rotary Position Embeddings."""
    
    def test_rope_creation(self):
        """Test RoPE creation."""
        from langtune.optimizations import RotaryPositionEmbedding
        
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=512)
        
        assert rope.dim == 64
        assert rope.max_seq_len == 512
    
    def test_rope_forward(self, device):
        """Test RoPE forward pass."""
        from langtune.optimizations import RotaryPositionEmbedding
        
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=512).to(device)
        
        # Create Q and K tensors (batch, heads, seq_len, head_dim)
        q = torch.randn(2, 4, 32, 64, device=device)
        k = torch.randn(2, 4, 32, 64, device=device)
        
        q_rot, k_rot = rope(q, k, seq_len=32)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestMemoryEfficientAttention:
    """Tests for memory-efficient attention."""
    
    def test_attention_creation(self):
        """Test attention creation."""
        from langtune.optimizations import MemoryEfficientAttention
        
        attn = MemoryEfficientAttention(embed_dim=64, num_heads=4)
        
        assert attn.embed_dim == 64
        assert attn.num_heads == 4
    
    def test_attention_forward(self, device):
        """Test attention forward pass."""
        from langtune.optimizations import MemoryEfficientAttention
        
        attn = MemoryEfficientAttention(embed_dim=64, num_heads=4).to(device)
        
        # Create Q, K, V tensors
        q = torch.randn(2, 4, 32, 16, device=device)  # batch, heads, seq, head_dim
        k = torch.randn(2, 4, 32, 16, device=device)
        v = torch.randn(2, 4, 32, 16, device=device)
        
        output = attn(q, k, v, is_causal=True)
        
        assert output.shape == q.shape


class TestFusedCrossEntropy:
    """Tests for fused cross-entropy loss."""
    
    def test_fused_cross_entropy(self, device):
        """Test fused cross-entropy computation."""
        from langtune.optimizations import fused_cross_entropy
        import torch.nn.functional as F
        
        logits = torch.randn(100, 1000, device=device)
        labels = torch.randint(0, 1000, (100,), device=device)
        
        # Fused version
        fused_loss = fused_cross_entropy(logits, labels)
        
        # Standard version
        standard_loss = F.cross_entropy(logits, labels)
        
        # Should be approximately equal
        assert torch.allclose(fused_loss, standard_loss, atol=1e-5)


class TestOptimizationConfig:
    """Tests for OptimizationConfig."""
    
    def test_config_creation(self):
        """Test config creation."""
        from langtune.optimizations import OptimizationConfig
        
        config = OptimizationConfig(
            use_4bit=True,
            use_flash_attention=True,
            mixed_precision="fp16"
        )
        
        assert config.use_4bit == True
        assert config.mixed_precision == "fp16"
    
    def test_invalid_config(self):
        """Test invalid config raises error."""
        from langtune.optimizations import OptimizationConfig
        
        # Can't use both 4-bit and 8-bit
        with pytest.raises(ValueError):
            OptimizationConfig(use_4bit=True, use_8bit=True)
