"""
verify_optimizations.py: Verify that new optimizations are correctly implemented and importable.
"""

import sys
import os
import torch
from pathlib import Path

# Add paths to sys.path
sys.path.append(str(Path("/Users/priteshraj/Downloads/langtune/src")))
sys.path.append(str(Path("/Users/priteshraj/Downloads/langvision/src")))

def verify_langtune():
    print("Verifying Langtune Optimizations...")
    try:
        from langtune.trainer import Trainer
        from langtune.lisa import LISA
        from langtune.kernels import fused_cross_entropy
        from langtune.fast_lora import FastLoRAConfig
        from langtune.packing import SequencePacker
        
        print("✅ Langtune imports successful")
        
        # Test basic instantiation
        lisa = LISA([torch.nn.Parameter(torch.randn(10))], lr=0.001)
        print("✅ LISA instantiated")
        
        packer = SequencePacker(max_seq_length=1024, pad_token_id=0)
        print("✅ SequencePacker instantiated")
        
    except Exception as e:
        print(f"❌ Langtune verification failed: {e}")
        import traceback
        traceback.print_exc()

def verify_langvision():
    print("\nVerifying Langvision Optimizations...")
    try:
        from langvision.training.fast_trainer import FastTrainer, FastTrainerConfig
        from langvision.training.lisa import LISA
        from langvision.training.kernels import fused_cross_entropy
        
        print("✅ Langvision imports successful")
        
        config = FastTrainerConfig(use_lisa=True)
        print(f"✅ FastTrainerConfig has LISA support: {config.use_lisa}")
        
    except Exception as e:
        print(f"❌ Langvision verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_langtune()
    verify_langvision()
