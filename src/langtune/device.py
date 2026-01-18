
import torch
import logging
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

class DeviceManager:
    """
    Abstracts device-specific logic for CUDA, MPS (Apple Silicon), and CPU.
    """
    
    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def get_device_type() -> str:
        return DeviceManager.get_device().type

    @staticmethod
    def is_cuda() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def is_mps() -> bool:
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    @staticmethod
    def is_bf16_supported() -> bool:
        if DeviceManager.is_cuda():
            return torch.cuda.is_bf16_supported()
        if DeviceManager.is_mps():
            # MPS supports bf16 on newer hardware/macOS versions
            # but for safety we default to False unless we can verify
            return False 
        return False

    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        stats = {}
        if DeviceManager.is_cuda():
            stats = {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
                'max_reserved': torch.cuda.max_memory_reserved() / 1e9
            }
        elif DeviceManager.is_mps():
            # Try to get MPS memory stats if available (PyTorch 2.0+)
            try:
                if hasattr(torch.mps, 'current_allocated_memory'):
                    stats['allocated'] = torch.mps.current_allocated_memory() / 1e9
                if hasattr(torch.mps, 'driver_allocated_memory'):
                    stats['driver_allocated'] = torch.mps.driver_allocated_memory() / 1e9
            except AttributeError:
                pass
        return stats

    @staticmethod
    def empty_cache():
        if DeviceManager.is_cuda():
            torch.cuda.empty_cache()
        elif DeviceManager.is_mps():
            torch.mps.empty_cache()

    @staticmethod
    def get_rng_state(device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = DeviceManager.get_device()
            
        if device.type == 'cuda':
            return torch.cuda.get_rng_state(device)
        elif device.type == 'mps':
             return torch.mps.get_rng_state()
        return torch.get_rng_state()

    @staticmethod
    def set_rng_state(state: torch.Tensor, device: torch.device = None):
        if device is None:
            device = DeviceManager.get_device()
            
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state, device)
        elif device.type == 'mps':
            torch.mps.set_rng_state(state)
        else:
            torch.set_rng_state(state)

    @staticmethod
    def autocast(enabled: bool = True, dtype: torch.dtype = torch.float16):
        device_type = DeviceManager.get_device_type()
        
        # Use generic torch.amp.autocast which supports cuda, cpu, and possibly mps in newer versions
        if hasattr(torch, 'amp'):
            return torch.amp.autocast(device_type=device_type, enabled=enabled, dtype=dtype)
            
        # Fallback for older PyTorch versions
        if device_type == 'cuda':
            return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)
        
        # MPS autocast fallback (might be no-op or cpu)
        return torch.autocast(device_type=device_type, enabled=enabled, dtype=dtype)
