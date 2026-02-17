"""
verify_triton.py: Verify Triton integration in Langtune.
"""

import sys
import os
import logging
from langtune.config import Config, TrainingConfig
from langtune.trainer import Trainer
try:
    from langtune.triton_kernels import is_triton_available
except ImportError:
    def is_triton_available(): return False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def verify_triton_config():
    print("Verifying Triton Configuration...")
    config = Config(
        model={}, 
        training={'use_triton': True}, 
        data={}
    )
    
    if config.training.use_triton:
        print("✅ Config.training.use_triton is True")
    else:
        print("❌ Config.training.use_triton failed to set")

def verify_triton_availability():
    print("\nVerifying Triton Availability...")
    available = is_triton_available()
    print(f"ℹ️  Triton available: {available}")
    
    if not available:
        print("⚠️  Triton not detected. This is expected on non-CUDA/non-Linux environments.")
        print("    The code should gracefully fallback to standard implementation.")
    else:
        print("✅ Triton is available and ready.")

if __name__ == "__main__":
    verify_triton_config()
    verify_triton_availability()
