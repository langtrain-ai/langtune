"""
transformer.py: Standard Transformer implementations for Langtune
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict

from .layers import LoRALinear, MultiHeadAttention

logger = logging.getLogger(__name__)

class TransformerBlock(nn.Module):
    """
    Transformer block with LoRA support.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        lora_config: Optional[Dict] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        mlp_dim = int(embed_dim * mlp_ratio)
        
        # Attention
        self.attention = MultiHeadAttention(
            embed_dim, num_heads, dropout, lora_config
        )
        self.attention_norm = nn.LayerNorm(embed_dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # LoRA for MLP
        self.use_lora = lora_config is not None
        if self.use_lora:
            self.lora_mlp_fc1 = LoRALinear(
                embed_dim, mlp_dim,
                rank=lora_config.get('rank', 8),
                alpha=lora_config.get('alpha', 16.0),
                dropout=lora_config.get('dropout', 0.1)
            )
            self.lora_mlp_fc2 = LoRALinear(
                mlp_dim, embed_dim,
                rank=lora_config.get('rank', 8),
                alpha=lora_config.get('alpha', 16.0),
                dropout=lora_config.get('dropout', 0.1)
            )
        
        self.mlp_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(x, mask)
        x = self.attention_norm(x + attn_out)
        
        # MLP with residual connection
        if self.use_lora:
            mlp_out = self.lora_mlp_fc1(x)
            mlp_out = F.gelu(mlp_out)
            mlp_out = self.lora_mlp_fc2(mlp_out)
            # Add original MLP output
            mlp_out = mlp_out + self.mlp(x)
        else:
            mlp_out = self.mlp(x)
            
        x = self.mlp_norm(x + mlp_out)
        return x

class LoRALanguageModel(nn.Module):
    """
    A complete language model with LoRA support for efficient fine-tuning.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 512,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        lora_config: Optional[Dict] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout, lora_config
            )
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized LoRALanguageModel with {self.count_parameters()} parameters")
        if lora_config:
            logger.info(f"LoRA config: {lora_config}")
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def count_lora_parameters(self) -> int:
        """Count LoRA-specific parameters."""
        lora_params = 0
        for module in self.modules():
            if isinstance(module, LoRALinear):
                lora_params += module.lora_A.numel() + module.lora_B.numel()
        return lora_params
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Token and positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to 4D for broadcasting
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            causal_mask = causal_mask * attention_mask
        
        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Final layer norm and output projection
        x = self.norm(x)
        logits = self.head(x)
        
        outputs = {"logits": logits}
        
        # Compute loss if labels are provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            outputs["loss"] = loss
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        pad_token_id: int = 0,
        eos_token_id: int = 1
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass
                outputs = self.forward(input_ids)
                logits = outputs["logits"][:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check for EOS token
                if (next_token == eos_token_id).all():
                    break
        
        return input_ids
