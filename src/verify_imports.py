
import sys
import os

# Add src directories to path
# Assuming we are in /Users/priteshraj/Downloads/langtune/src or executed from there
sys.path.append("/Users/priteshraj/Downloads/langtune/src")
sys.path.append("/Users/priteshraj/Downloads/langvision/src")

print("Checking Langtune imports...")
try:
    import langtune.finetune
    import langtune.trainer
    import langtune.device
    import langtune.presets
    print("Langtune imports successful.")
except ImportError as e:
    print(f"Langtune import failed: {e}")
    sys.exit(1)

print("Checking Langvision imports...")
try:
    from langvision.training import FastTrainer, VisionLLMFineTuner
    print("Langvision imports successful.")
except ImportError as e:
    print(f"Langvision import failed: {e}")
    sys.exit(1)
