"""
verify_triton.py: Verify Triton kernel availability for Langtune.

Checks whether the langtrain-server kernel stack is reachable and reports
which acceleration tiers are active. Triton kernels live exclusively in
langtrain-server; langtune delegates to them via kernels.py.
"""

import sys
import os
import logging
import unittest.mock as mock

sys.path.append(os.path.join(os.getcwd(), 'src'))

sys.modules['torch']                  = mock.MagicMock()
sys.modules['torch.nn']               = mock.MagicMock()
sys.modules['torch.nn.functional']    = mock.MagicMock()
sys.modules['torch.cuda']             = mock.MagicMock()
sys.modules['torch.cuda.amp']         = mock.MagicMock()
sys.modules['torch.optim']            = mock.MagicMock()
sys.modules['torch.optim.lr_scheduler'] = mock.MagicMock()
sys.modules['torch.utils']            = mock.MagicMock()
sys.modules['torch.utils.data']       = mock.MagicMock()
sys.modules['torch.distributed']      = mock.MagicMock()
sys.modules['yaml']                   = mock.MagicMock()
sys.modules['wandb']                  = mock.MagicMock()
sys.modules['tqdm']                   = mock.MagicMock()
sys.modules['numpy']                  = mock.MagicMock()

try:
    from langtune.config import Config, TrainingConfig
    from langtune.trainer import Trainer, create_trainer
    from langtune.kernels import kernel_status
except ImportError as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

logging.basicConfig(level=logging.INFO)


def verify_triton_config():
    print("Verifying kernel configuration...")
    try:
        from langtune.config import ModelConfig, DataConfig
        config = Config(model=ModelConfig(), training=TrainingConfig(), data=DataConfig())
        print("✅ Config loaded successfully")
    except Exception as e:
        print(f"❌ Config verification failed: {e}")
        import traceback
        traceback.print_exc()


def verify_triton_availability():
    print("\nVerifying kernel availability via langtrain-server delegation...")
    status = kernel_status()
    server_found = status.get("server_found", False)

    if server_found:
        print("✅ langtrain-server kernel stack reachable")
        for key, val in status.items():
            if key != "server_found":
                tick = "✅" if val else "⚠️ "
                print(f"   {tick}  {key}: {val}")
    else:
        print("⚠️  langtrain-server not found — using PyTorch native ops (fallback)")
        print("    Set LANGTRAIN_SERVER_PATH env var to activate kernels.")


if __name__ == "__main__":
    verify_triton_config()
    verify_triton_availability()
