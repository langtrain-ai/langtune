
import sys
import os

# Add src to path so we can import langtune
sys.path.append(os.path.join(os.getcwd(), 'src'))

import torch
import logging
from langtune.device import DeviceManager
# from langtune import finetune # finetune imports optimizations which we want to test

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_mps")

def verify():
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"MPS Available (torch.backends.mps.is_available()): {torch.backends.mps.is_available()}")
    logger.info(f"DeviceManager Device: {DeviceManager.get_device()}")
    
    if DeviceManager.is_mps():
        logger.info("✅ MPS detected correctly by DeviceManager")
        
        # Test basic tensor creation on MPS
        try:
            x = torch.ones(5, device='mps')
            logger.info(f"✅ Successfully created tensor on MPS: {x}")
        except Exception as e:
            logger.error(f"❌ Failed to create tensor on MPS: {e}")
            
    else:
        logger.warning("⚠️ MPS not detected by DeviceManager (might be on CPU-only setup or older macOS)")

    # Test optimizations import
    try:
        from langtune.optimizations import get_memory_stats
        stats = get_memory_stats()
        logger.info(f"Memory Stats from optimizations: {stats}")
        logger.info("✅ langtune.optimizations imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import langtune.optimizations: {e}")
        import traceback
        traceback.print_exc()

    logger.info("Verification script finished.")

if __name__ == "__main__":
    verify()
