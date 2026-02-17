"""
verify_triton.py: Verify Triton integration in Langtune.
"""

import sys
import os
import logging
import unittest.mock as mock

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Comprehensive mocking of torch and dependencies
sys.modules['torch'] = mock.MagicMock()
sys.modules['torch.nn'] = mock.MagicMock()
sys.modules['torch.nn.functional'] = mock.MagicMock()
sys.modules['torch.cuda'] = mock.MagicMock()
sys.modules['torch.cuda.amp'] = mock.MagicMock()
sys.modules['torch.optim'] = mock.MagicMock()
sys.modules['torch.optim.lr_scheduler'] = mock.MagicMock()
sys.modules['torch.utils'] = mock.MagicMock()
sys.modules['torch.utils.data'] = mock.MagicMock()
sys.modules['torch.distributed'] = mock.MagicMock()
sys.modules['yaml'] = mock.MagicMock()
sys.modules['wandb'] = mock.MagicMock()
sys.modules['tqdm'] = mock.MagicMock()
sys.modules['numpy'] = mock.MagicMock()

# Now import langtune components
try:
    from langtune.config import Config, TrainingConfig
    from langtune.trainer import Trainer, create_trainer
    from langtune.triton_kernels import is_triton_available
except ImportError as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def verify_triton_config():
    print("Verifying Triton Configuration...")
    try:
        # Create config with Triton enabled
        from langtune.config import ModelConfig, DataConfig, LoRAConfig
        
        model_config = ModelConfig()
        data_config = DataConfig()
        training_config = TrainingConfig(use_triton=True)
        
        config = Config(
            model=model_config,
            training=training_config,
            data=data_config
        )
        
        if config.training.use_triton:
            print("✅ Config.training.use_triton is set to True")
        else:
            print("❌ Config.training.use_triton failed to set")
            
    except Exception as e:
        print(f"❌ Config verification failed: {e}")
        import traceback
        traceback.print_exc()

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
