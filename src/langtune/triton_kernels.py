"""
triton_kernels.py: High-performance GPU kernels using OpenAI Triton.

Handles fused operations for maximum throughput and reduced memory bandwidth.
"""

import torch
import logging

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    # logger.warning("Triton not installed. Triton kernels will be disabled.")

def is_triton_available():
    return HAS_TRITON and torch.cuda.is_available()

if HAS_TRITON:
    @triton.jit
    def cross_entropy_fwd_kernel(
        logits_ptr, float_loss_ptr, labels_ptr,
        n_rows, n_cols,
        BLOCK_SIZE: tl.constexpr
    ):
        """
        Fused Cross Entropy Forward Kernel.
        Computes log_softmax and nll_loss in a single pass without materializing logits.
        """
        row_idx = tl.program_id(0)
        
        # Pointers to current row
        logits_row_ptr = logits_ptr + row_idx * n_cols
        labels_row_ptr = labels_ptr + row_idx
        
        # Load label
        label = tl.load(labels_row_ptr)
        
        # Ignore index check (usually -100)
        if label == -100:
            tl.store(float_loss_ptr + row_idx, 0.0)
            return

        # Compute max(logits) for numerical stability
        # Loop over columns in blocks
        max_val = -float('inf')
        for i in range(0, n_cols, BLOCK_SIZE):
            col_offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            logits = tl.load(logits_row_ptr + col_offsets, mask=mask, other=-float('inf'))
            max_val = tl.max(max_val, tl.max(logits, 0))
            
        # Compute sum(exp(logits - max_val))
        sum_exp = 0.0
        for i in range(0, n_cols, BLOCK_SIZE):
            col_offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            logits = tl.load(logits_row_ptr + col_offsets, mask=mask, other=-float('inf'))
            sum_exp += tl.sum(tl.exp(logits - max_val))
            
        # Compute loss: -logits[label] + max_val + log(sum_exp)
        # Load logits[label]
        label_logit = tl.load(logits_row_ptr + label)
        loss = -label_logit + max_val + tl.log(sum_exp)
        
        tl.store(float_loss_ptr + row_idx, loss)

    def triton_cross_entropy(logits, labels, ignore_index=-100):
        """
        Triton-accelerated Cross Entropy Loss.
        """
        if not is_triton_available():
            raise RuntimeError("Triton is not available or CUDA is missing.")
            
        n_rows, n_cols = logits.shape
        
        # Ensure contiguous
        logits = logits.contiguous()
        labels = labels.contiguous()
        
        # Output buffer
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        
        # Block size (next power of 2)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK_SIZE >= 2048: num_warps = 8
        if BLOCK_SIZE >= 4096: num_warps = 16
        
        # Grid
        grid = (n_rows,)
        
        cross_entropy_fwd_kernel[grid](
            logits, losses, labels,
            n_rows, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        
        return losses.mean()

    # NOTE: A full training implementation requires the backward kernel as well.
    # For this iteration, we focus on the forward pass logic or a simplified version.
    # Writing a stable backward pass for CE in Triton is complex.
    # Ideally we would use liger-kernel or unsloth's implementation if we could import them.
    # Since we are writing from scratch, we will wrap this in a torch.autograd.Function
    # if we implement the backward pass, OR use this for inference/eval where we only need loss.
    
    # Implementing a basic backward kernel for completeness of the "optimization" request.
    
    @triton.jit
    def cross_entropy_bwd_kernel(
        dloss_ptr, logits_ptr, labels_ptr, dlogits_ptr,
        n_rows, n_cols,
        BLOCK_SIZE: tl.constexpr
    ):
        """
        Fused Cross Entropy Backward Kernel.
        dLoss/dLogits = softmax(logits) - target
        """
        row_idx = tl.program_id(0)
        
        logits_row_ptr = logits_ptr + row_idx * n_cols
        labels_row_ptr = labels_ptr + row_idx
        dlogits_row_ptr = dlogits_ptr + row_idx * n_cols
        
        label = tl.load(labels_row_ptr)
        if label == -100:
            for i in range(0, n_cols, BLOCK_SIZE):
                col_offsets = i + tl.arange(0, BLOCK_SIZE)
                mask = col_offsets < n_cols
                tl.store(dlogits_row_ptr + col_offsets, 0.0, mask=mask)
            return

        # Recompute max and sum_exp for softmax
        max_val = -float('inf')
        for i in range(0, n_cols, BLOCK_SIZE):
            col_offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            logits = tl.load(logits_row_ptr + col_offsets, mask=mask, other=-float('inf'))
            max_val = tl.max(max_val, tl.max(logits, 0))
            
        sum_exp = 0.0
        for i in range(0, n_cols, BLOCK_SIZE):
            col_offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            logits = tl.load(logits_row_ptr + col_offsets, mask=mask, other=-float('inf'))
            sum_exp += tl.sum(tl.exp(logits - max_val))
            
        # Compute gradient: (exp(x - max) / sum_exp) - 1(if target)
        # Scale by dloss (usually 1/batch_size for mean reduction)
        dloss = tl.load(dloss_ptr) # Scalar gradient from mean reduction
        
        for i in range(0, n_cols, BLOCK_SIZE):
            col_offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            
            logits = tl.load(logits_row_ptr + col_offsets, mask=mask, other=-float('inf'))
            softmax = tl.exp(logits - max_val) / sum_exp
            
            # Subtract 1 where index == label
            # This is tricky in vector code. 
            # We can create a mask for where col_offsets == label.
            target_mask = (col_offsets == label)
            
            # Using casting to subtract 1.0 (triton syntax varies, simple way:)
            grad = softmax - target_mask.to(tl.float32)
            
            tl.store(dlogits_row_ptr + col_offsets, grad * dloss, mask=mask)

    class TritonCrossEntropyLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, labels, ignore_index=-100):
            n_rows, n_cols = logits.shape
            logits = logits.contiguous()
            labels = labels.contiguous()
            
            losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
            BLOCK_SIZE = triton.next_power_of_2(n_cols)
            num_warps = 4
            if BLOCK_SIZE >= 2048: num_warps = 8
            if BLOCK_SIZE >= 4096: num_warps = 16
            
            cross_entropy_fwd_kernel[(n_rows,)](
                logits, losses, labels,
                n_rows, n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps
            )
            
            ctx.save_for_backward(logits, labels)
            ctx.ignore_index = ignore_index
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.num_warps = num_warps
            
            return losses.mean()

        @staticmethod
        def backward(ctx, grad_output):
            logits, labels = ctx.saved_tensors
            n_rows, n_cols = logits.shape
            
            # grad_output is a scalar (mean of losses)
            # Make sure it's on device
            if not isinstance(grad_output, torch.Tensor):
                grad_output = torch.tensor(grad_output, device=logits.device)
            
            # Since we reduce by mean, we need to pass that scalar to the kernel
            # BUT efficient kernel usually takes a scalar. 
            # BUT triton kernel takes pointers. 
            # We need to put scalar into a tensor ptr.
            # However, for simplicity here, we can just load the single value in kernel if we pass a 1-element tensor.
            # But `grad_output` from mean() is 0-dim or 1-dim. 
            
            # Adjust grad_output for the mean reduction: d(mean)/dx = 1/N * d(sum)/dx
            # so grad_output holds '1' usually, we need to divide by N inside?
            # actually Pytorch autograd passes the incoming gradient.
            # If the output was losses.mean(), grad_output is 1.0. 
            # The derivative of mean is 1/N. 
            # The kernel implements dLoss/dx. We multiply by grad_output * (1/rows).
            
            grad_scale = grad_output / n_rows
            grad_scale_tensor = torch.full((1,), grad_scale.item(), device=logits.device, dtype=torch.float32)
            
            dlogits = torch.empty_like(logits)
            
            cross_entropy_bwd_kernel[(n_rows,)](
                grad_scale_tensor, logits, labels, dlogits,
                n_rows, n_cols,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps
            )
            
            return dlogits, None, None

else:
    # Fallback
    def triton_cross_entropy(logits, labels, ignore_index=-100):
        # Fallback to standard torch
        return torch.nn.functional.cross_entropy(logits, labels, ignore_index=ignore_index)
        
    class TritonCrossEntropyLoss:
        @staticmethod
        def apply(logits, labels, ignore_index=-100):
            return torch.nn.functional.cross_entropy(logits, labels, ignore_index=ignore_index)

