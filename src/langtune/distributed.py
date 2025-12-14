"""
distributed.py: Distributed training utilities for Langtune

Provides helpers for multi-GPU and distributed training.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_initialized()


def get_rank() -> int:
    """Get current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process."""
    return get_rank() == 0


def setup_distributed(backend: str = "nccl", init_method: str = "env://"):
    """Initialize distributed training."""
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        if world_size > 1:
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank
            )
            logger.info(f"Initialized distributed: rank={rank}, world_size={world_size}")


def cleanup_distributed():
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()


def barrier():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """All-reduce across processes."""
    if is_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source rank."""
    if is_distributed():
        dist.broadcast(tensor, src)
    return tensor


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all processes."""
    if not is_distributed():
        return tensor
    
    world_size = get_world_size()
    tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor)
    return torch.cat(tensors, dim=0)


class DistributedDataParallelWrapper:
    """Simple DDP wrapper for models."""
    
    def __init__(self, model: torch.nn.Module, device_id: Optional[int] = None):
        self.model = model
        self.device_id = device_id
        
        if is_distributed():
            self.model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device_id] if device_id is not None else None,
                output_device=device_id
            )
    
    def __getattr__(self, name):
        return getattr(self.model, name)
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def wrap_model_ddp(model: torch.nn.Module, device_id: Optional[int] = None):
    """Wrap model with DDP if in distributed mode."""
    if is_distributed():
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device_id] if device_id is not None else None,
            output_device=device_id
        )
    return model


def get_distributed_sampler(dataset, shuffle: bool = True):
    """Get distributed sampler for dataset."""
    if is_distributed():
        return torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle
        )
    return None


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce dictionary values across processes."""
    if not is_distributed():
        return input_dict
    
    world_size = get_world_size()
    keys = sorted(input_dict.keys())
    values = torch.tensor([input_dict[k] for k in keys], dtype=torch.float32)
    
    if values.is_cuda:
        values = values.cuda()
    
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    
    if average:
        values /= world_size
    
    return {k: v.item() for k, v in zip(keys, values)}
