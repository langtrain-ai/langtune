"""
layers.py: Basic neural network layers with LoRA support for Langtune
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) linear layer.
    
    Implements the LoRA technique for efficient fine-tuning by adding
    low-rank matrices to existing linear layers.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        merge_weights: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.merge_weights = merge_weights
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LoRA weights."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer."""
        if self.merge_weights:
            # Use merged weights for inference
            weight = self.get_merged_weight()
            return F.linear(x, weight)
        else:
            # Use LoRA adaptation
            lora_output = self.dropout_layer(x) @ self.lora_A.T @ self.lora_B.T
            return lora_output * self.scaling
    
    def get_merged_weight(self) -> torch.Tensor:
        """Get the merged weight matrix for inference."""
        return self.lora_B @ self.lora_A * self.scaling
    
    def merge_weights(self):
        """Merge LoRA weights into the base layer."""
        self.merge_weights = True
    
    def unmerge_weights(self):
        """Unmerge LoRA weights for training."""
        self.merge_weights = False

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with LoRA support.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        lora_config: Optional[Dict] = None
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Standard attention projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # LoRA adapters
        self.use_lora = lora_config is not None
        if self.use_lora:
            self.lora_qkv = LoRALinear(
                embed_dim, 3 * embed_dim,
                rank=lora_config.get('rank', 8),
                alpha=lora_config.get('alpha', 16.0),
                dropout=lora_config.get('dropout', 0.1)
            )
            self.lora_proj = LoRALinear(
                embed_dim, embed_dim,
                rank=lora_config.get('rank', 8),
                alpha=lora_config.get('alpha', 16.0),
                dropout=lora_config.get('dropout', 0.1)
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute Q, K, V
        if self.use_lora:
            from langtune.acceleration import Accelerator
            accelerator = Accelerator()
            
            if accelerator.is_available() and x.is_cuda:
                qkv = accelerator.fused_lora(
                    x, 
                    self.qkv.weight, 
                    self.lora_qkv.lora_A, 
                    self.lora_qkv.lora_B, 
                    self.lora_qkv.scaling
                )
                if self.qkv.bias is not None:
                    qkv += self.qkv.bias
            else:
                qkv = self.lora_qkv(x) + self.qkv(x)
        else:
            qkv = self.qkv(x)
            
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        # Output projection
        if self.use_lora:
            if accelerator.is_available() and x.is_cuda:
                out = accelerator.fused_lora(
                    out, 
                    self.proj.weight, 
                    self.lora_proj.lora_A, 
                    self.lora_proj.lora_B, 
                    self.lora_proj.scaling
                )
                if self.proj.bias is not None:
                    out += self.proj.bias
            else:
                out = self.lora_proj(out) + self.proj(out)
        else:
            out = self.proj(out)
            
        return out
