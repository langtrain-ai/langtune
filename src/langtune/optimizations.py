"""
optimizations.py: Efficient fine-tuning optimizations for Langtune

This module implements memory-efficient and speed-optimized techniques
inspired by Unsloth, including:
- 4-bit quantization (QLoRA style)
- Rotary Position Embeddings (RoPE)
- Fused cross-entropy loss
- Memory-efficient attention
- Gradient checkpointing utilities
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Tuple, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

# Check for available optimizations
FLASH_ATTENTION_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    pass

# Check for bitsandbytes (optional 4-bit support)
BITSANDBYTES_AVAILABLE = False
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Quantization Utilities
# =============================================================================

def quantize_tensor_4bit(
    tensor: torch.Tensor,
    group_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to 4-bit representation.
    
    Args:
        tensor: Input tensor to quantize
        group_size: Number of elements per quantization group
        
    Returns:
        Tuple of (quantized_data, scales, zero_points)
    """
    original_shape = tensor.shape
    tensor = tensor.reshape(-1, group_size)
    
    # Compute min/max per group
    min_vals = tensor.min(dim=1, keepdim=True)[0]
    max_vals = tensor.max(dim=1, keepdim=True)[0]
    
    # Compute scale and zero point
    scales = (max_vals - min_vals) / 15.0  # 4-bit = 16 levels
    zero_points = min_vals
    
    # Quantize
    quantized = torch.clamp(
        torch.round((tensor - zero_points) / (scales + 1e-8)),
        0, 15
    ).to(torch.uint8)
    
    # Pack two 4-bit values into one uint8
    packed = quantized[:, ::2] | (quantized[:, 1::2] << 4)
    
    return packed, scales.squeeze(-1), zero_points.squeeze(-1)


def dequantize_tensor_4bit(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    group_size: int = 64,
    output_shape: Tuple[int, ...] = None
) -> torch.Tensor:
    """
    Dequantize a 4-bit tensor back to float.
    
    Args:
        packed: Packed 4-bit tensor
        scales: Quantization scales
        zero_points: Quantization zero points
        group_size: Number of elements per group
        output_shape: Original tensor shape
        
    Returns:
        Dequantized float tensor
    """
    # Unpack 4-bit values
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    
    # Interleave to get original order
    batch_size = packed.shape[0]
    unpacked = torch.zeros(batch_size, group_size, device=packed.device, dtype=torch.float32)
    unpacked[:, ::2] = low.float()
    unpacked[:, 1::2] = high.float()
    
    # Dequantize
    dequantized = unpacked * scales.unsqueeze(-1) + zero_points.unsqueeze(-1)
    
    if output_shape is not None:
        dequantized = dequantized.reshape(output_shape)
    
    return dequantized


class QuantizedLinear(nn.Module):
    """
    4-bit quantized linear layer with efficient on-the-fly dequantization.
    
    This provides significant memory savings (75%) compared to FP16,
    with minimal accuracy loss.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        compute_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.compute_dtype = compute_dtype
        
        # Ensure dimensions are compatible with group size
        assert in_features % group_size == 0, f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        
        # Number of groups
        self.num_groups = in_features // group_size
        
        # Quantized weight storage (packed 4-bit)
        self.register_buffer(
            'weight_packed',
            torch.zeros(out_features * self.num_groups, group_size // 2, dtype=torch.uint8)
        )
        self.register_buffer(
            'weight_scales',
            torch.ones(out_features * self.num_groups, dtype=compute_dtype)
        )
        self.register_buffer(
            'weight_zeros',
            torch.zeros(out_features * self.num_groups, dtype=compute_dtype)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=compute_dtype))
        else:
            self.register_parameter('bias', None)
        
        self._is_initialized = False
    
    def initialize_from_weight(self, weight: torch.Tensor):
        """Initialize quantized weights from a float weight tensor."""
        assert weight.shape == (self.out_features, self.in_features)
        
        # Reshape for group-wise quantization
        weight_grouped = weight.reshape(self.out_features * self.num_groups, self.group_size)
        
        # Quantize
        packed, scales, zeros = quantize_tensor_4bit(weight_grouped, self.group_size)
        
        self.weight_packed.copy_(packed)
        self.weight_scales.copy_(scales.to(self.compute_dtype))
        self.weight_zeros.copy_(zeros.to(self.compute_dtype))
        self._is_initialized = True
    
    def get_weight(self) -> torch.Tensor:
        """Dequantize and return the full weight matrix."""
        weight = dequantize_tensor_4bit(
            self.weight_packed,
            self.weight_scales,
            self.weight_zeros,
            self.group_size,
            (self.out_features, self.in_features)
        )
        return weight.to(self.compute_dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization."""
        weight = self.get_weight()
        output = F.linear(x.to(self.compute_dtype), weight, self.bias)
        return output


class LoRALinear4bit(nn.Module):
    """
    QLoRA-style 4-bit quantized linear with LoRA adapters.
    
    Combines:
    - 4-bit quantized base weights (frozen)
    - Full-precision LoRA adapters (trainable)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        group_size: int = 64,
        compute_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Quantized base weight (frozen)
        self.base = QuantizedLinear(
            in_features, out_features,
            bias=False,
            group_size=group_size,
            compute_dtype=compute_dtype
        )
        
        # LoRA adapters (trainable, full precision)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base output + LoRA adaptation."""
        # Base forward (quantized)
        base_output = self.base(x)
        
        # LoRA forward (full precision)
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling
        
        return base_output + lora_output.to(base_output.dtype)


# =============================================================================
# Rotary Position Embeddings (RoPE)
# =============================================================================

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) for efficient position encoding.
    
    RoPE encodes position information directly into the attention computation,
    providing better extrapolation and computational efficiency.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin for max sequence length
        self._precompute_cos_sin(max_seq_len)
    
    def _precompute_cos_sin(self, seq_len: int):
        """Precompute cos and sin for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape (batch, heads, seq_len, head_dim)
            k: Key tensor of shape (batch, heads, seq_len, head_dim)
            seq_len: Sequence length
            
        Returns:
            Tuple of (q_rotated, k_rotated)
        """
        # Extend cache if needed
        if seq_len > self.max_seq_len:
            self._precompute_cos_sin(seq_len)
            self.max_seq_len = seq_len
        
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary embedding to a single tensor."""
        # Split into two halves
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        
        # Rotate
        rotated = torch.cat((-x2, x1), dim=-1)
        
        # Apply rotation
        return x * cos + rotated * sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Functional interface for applying rotary position embeddings.
    
    More efficient for models that precompute cos/sin.
    """
    # Rotate query
    q1, q2 = q[..., :q.shape[-1]//2], q[..., q.shape[-1]//2:]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    q_out = q * cos + q_rotated * sin
    
    # Rotate key
    k1, k2 = k[..., :k.shape[-1]//2], k[..., k.shape[-1]//2:]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    k_out = k * cos + k_rotated * sin
    
    return q_out, k_out


# =============================================================================
# Memory-Efficient Attention
# =============================================================================

class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention implementation.
    
    Uses chunked computation to reduce peak memory usage,
    with automatic fallback to flash attention when available.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True,
        chunk_size: int = 1024
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.chunk_size = chunk_size
        
        # Use flash attention if available and requested
        self.use_flash = use_flash and FLASH_ATTENTION_AVAILABLE
        
        if self.use_flash:
            logger.info("Using Flash Attention for memory-efficient computation")
        else:
            logger.info("Using chunked attention (Flash Attention not available)")
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Compute attention efficiently.
        
        Args:
            q: Query tensor (batch, heads, seq_len, head_dim)
            k: Key tensor (batch, heads, seq_len, head_dim)
            v: Value tensor (batch, heads, seq_len, head_dim)
            mask: Optional attention mask
            is_causal: Whether to use causal masking
            
        Returns:
            Attention output tensor
        """
        if self.use_flash:
            return self._flash_attention(q, k, v, is_causal)
        else:
            return self._chunked_attention(q, k, v, mask, is_causal)
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool
    ) -> torch.Tensor:
        """Use flash attention for efficient computation."""
        # Flash attention expects (batch, seq_len, heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=is_causal
        )
        
        return out.transpose(1, 2)
    
    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_causal: bool
    ) -> torch.Tensor:
        """
        Compute attention in chunks to reduce memory usage.
        
        This is a fallback when flash attention is not available.
        """
        batch, heads, seq_len, head_dim = q.shape
        
        # For short sequences, use standard attention
        if seq_len <= self.chunk_size:
            return self._standard_attention(q, k, v, mask, is_causal)
        
        # Chunked computation
        output = torch.zeros_like(q)
        
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            q_chunk = q[:, :, start:end, :]
            
            # Compute attention scores for this chunk
            attn = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            
            # Apply causal mask if needed
            if is_causal:
                causal_mask = torch.tril(
                    torch.ones(end - start, seq_len, device=q.device),
                    diagonal=start
                )
                attn = attn.masked_fill(causal_mask == 0, float('-inf'))
            
            if mask is not None:
                attn = attn + mask[:, :, start:end, :]
            
            attn = F.softmax(attn, dim=-1)
            
            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)
            
            output[:, :, start:end, :] = torch.matmul(attn, v)
        
        return output
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_causal: bool
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if is_causal:
            seq_len = q.shape[2]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
            attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        
        if mask is not None:
            attn = attn + mask
        
        attn = F.softmax(attn, dim=-1)
        
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        
        return torch.matmul(attn, v)


# =============================================================================
# Fused Cross-Entropy Loss
# =============================================================================

def fused_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = 'mean',
    label_smoothing: float = 0.0,
    chunk_size: int = 4096
) -> torch.Tensor:
    """
    Memory-efficient cross-entropy loss.
    
    Computes cross-entropy in chunks to avoid materializing the full
    softmax matrix, significantly reducing memory usage for large vocabularies.
    
    Args:
        logits: Model outputs of shape (batch * seq_len, vocab_size)
        labels: Target labels of shape (batch * seq_len,)
        ignore_index: Label index to ignore
        reduction: 'none', 'mean', or 'sum'
        label_smoothing: Label smoothing factor
        chunk_size: Chunk size for processing
        
    Returns:
        Cross-entropy loss
    """
    batch_size = logits.shape[0]
    vocab_size = logits.shape[1]
    
    # For small batches, use standard cross-entropy
    if batch_size <= chunk_size:
        return F.cross_entropy(
            logits, labels,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
    
    # Chunked computation for large batches
    total_loss = 0.0
    total_count = 0
    
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        chunk_logits = logits[start:end]
        chunk_labels = labels[start:end]
        
        # Compute loss for this chunk
        chunk_loss = F.cross_entropy(
            chunk_logits, chunk_labels,
            ignore_index=ignore_index,
            reduction='sum',
            label_smoothing=label_smoothing
        )
        
        # Count valid labels
        valid_mask = chunk_labels != ignore_index
        chunk_count = valid_mask.sum().item()
        
        total_loss += chunk_loss
        total_count += chunk_count
    
    if reduction == 'none':
        raise ValueError("reduction='none' not supported in chunked mode")
    elif reduction == 'sum':
        return total_loss
    else:  # mean
        return total_loss / max(total_count, 1)


# =============================================================================
# Gradient Checkpointing Utilities
# =============================================================================

class GradientCheckpointFunction(torch.autograd.Function):
    """
    Custom gradient checkpointing function for transformer layers.
    
    More efficient than torch.utils.checkpoint by avoiding
    redundant context saves.
    """
    
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        
        # Save RNG state if needed
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_cuda_in_fwd = False
            if torch.cuda.is_available():
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)
        
        # Save input tensors for backward
        ctx.save_for_backward(*args)
        
        with torch.no_grad():
            outputs = run_function(*args)
        
        return outputs
    
    @staticmethod
    def backward(ctx, *output_grads):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad()"
            )
        
        inputs = ctx.saved_tensors
        
        # Restore RNG state
        if ctx.preserve_rng_state:
            rng_devices = []
            if ctx.had_cuda_in_fwd:
                rng_devices = ctx.fwd_gpu_devices
            
        with torch.enable_grad():
            # Recompute forward pass
            detached_inputs = [x.detach().requires_grad_(x.requires_grad) for x in inputs]
            outputs = ctx.run_function(*detached_inputs)
        
        # Compute gradients
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        
        grads = torch.autograd.grad(
            outputs,
            [x for x in detached_inputs if x.requires_grad],
            output_grads,
            allow_unused=True
        )
        
        # Match gradients to inputs
        grad_iter = iter(grads)
        input_grads = []
        for x in detached_inputs:
            if x.requires_grad:
                input_grads.append(next(grad_iter))
            else:
                input_grads.append(None)
        
        return (None, None) + tuple(input_grads)


def get_device_states(*args):
    """Get CUDA RNG state for gradient checkpointing."""
    fwd_gpu_devices = []
    fwd_gpu_states = []
    
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            device = arg.device
            if device not in fwd_gpu_devices:
                fwd_gpu_devices.append(device)
                fwd_gpu_states.append(torch.cuda.get_rng_state(device))
    
    return fwd_gpu_devices, fwd_gpu_states


def checkpoint(function, *args, preserve_rng_state: bool = True):
    """
    Apply gradient checkpointing to a function.
    
    Args:
        function: Function to checkpoint
        *args: Arguments to the function
        preserve_rng_state: Whether to preserve RNG state
        
    Returns:
        Function output with gradient checkpointing applied
    """
    return GradientCheckpointFunction.apply(function, preserve_rng_state, *args)


# =============================================================================
# Mixed Precision Training Utilities
# =============================================================================

class MixedPrecisionTrainer:
    """
    Mixed precision training utilities.
    
    Provides automatic mixed precision (AMP) with gradient scaling
    for stable training.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000
    ):
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        
        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
            logger.info(f"Mixed precision training enabled with {dtype}")
        else:
            self.scaler = None
    
    @property
    def autocast_context(self):
        """Get autocast context manager."""
        return autocast(enabled=self.enabled, dtype=self.dtype)
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient stability."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def unscale_gradients(self, optimizer):
        """Unscale gradients before clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def step(self, optimizer):
        """Take optimizer step with gradient scaling."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def get_scale(self) -> float:
        """Get current gradient scale."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0


# =============================================================================
# Memory Monitoring
# =============================================================================

def get_memory_stats() -> Dict[str, float]:
    """
    Get GPU memory statistics.
    
    Returns:
        Dictionary with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'reserved': torch.cuda.memory_reserved() / 1e9,
        'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
        'max_reserved': torch.cuda.max_memory_reserved() / 1e9
    }


def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cleanup_memory():
    """Free unused GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_memory_usage(prefix: str = ""):
    """Log current GPU memory usage."""
    stats = get_memory_stats()
    if stats:
        logger.info(
            f"{prefix}Memory: {stats['allocated']:.2f}GB allocated, "
            f"{stats['max_allocated']:.2f}GB peak"
        )


# =============================================================================
# Optimization Config
# =============================================================================

class OptimizationConfig:
    """Configuration for optimization settings."""
    
    def __init__(
        self,
        use_4bit: bool = False,
        use_8bit: bool = False,
        use_flash_attention: bool = True,
        use_gradient_checkpointing: bool = True,
        use_fused_ops: bool = True,
        use_rope: bool = True,
        gradient_accumulation_steps: int = 4,
        mixed_precision: str = "fp16",  # fp16, bf16, or fp32
        group_size: int = 64,
        chunk_size: int = 1024
    ):
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.use_flash_attention = use_flash_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_fused_ops = use_fused_ops
        self.use_rope = use_rope
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.group_size = group_size
        self.chunk_size = chunk_size
        
        # Validate
        if use_4bit and use_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization")
        
        if mixed_precision not in ["fp16", "bf16", "fp32"]:
            raise ValueError(f"Invalid mixed_precision: {mixed_precision}")
    
    @property
    def compute_dtype(self) -> torch.dtype:
        """Get compute dtype based on config."""
        if self.mixed_precision == "bf16":
            return torch.bfloat16
        elif self.mixed_precision == "fp16":
            return torch.float16
        else:
            return torch.float32
    
    def __repr__(self):
        return (
            f"OptimizationConfig("
            f"use_4bit={self.use_4bit}, "
            f"use_flash_attention={self.use_flash_attention}, "
            f"use_gradient_checkpointing={self.use_gradient_checkpointing}, "
            f"mixed_precision={self.mixed_precision})"
        )
