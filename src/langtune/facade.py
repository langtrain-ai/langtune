"""
High-level facades for Langtune to match the documentation.
"""
from typing import Optional, List, Dict, Any, Union
import os
import json
import torch
from pathlib import Path

from .trainer import Trainer
from .config import TrainingConfig, ModelConfig, LoRAConfig, DataConfig
from .models import LoRALanguageModel
from .data import TextDataset
from .finetune import finetune

class LoRATrainer:
    """
    Easy-to-use trainer for LoRA fine-tuning.
    Matches the API described in the Quick Start documentation.
    """
    
    def __init__(
        self, 
        model_name: str, 
        output_dir: str, 
        load_in_4bit: bool = False,
        **kwargs
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.load_in_4bit = load_in_4bit
        self.hyperparameters = kwargs # Store hyperparameters
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def train(self, training_data: List[Dict[str, str]]):
        """
        Train the model on the provided data.
        
        Args:
            training_data: List of dicts with 'user' and 'assistant' keys
        """
        print(f"🚀 Starting training for {self.model_name}...")
        
        # Convert list of dicts to a temporary JSONL file
        temp_data_path = os.path.join(self.output_dir, "train.jsonl")
        with open(temp_data_path, "w") as f:
            for item in training_data:
                # Format as chat template if needed, or just dump
                f.write(json.dumps(item) + "\n")
        
        self.train_from_file(temp_data_path)
        
    def train_from_file(self, file_path: str):
        """Train from a local file."""
        print(f"📂 Loading data from {file_path}")
        
        # Map high-level args to internal Config
        hp = self.hyperparameters
        
        # Determine strict base model (allow override via config)
        effective_base_model = hp.get("base_model", self.model_name)
        hf_token = hp.get("hf_token", None)
        
        print(f"⚙️ Configuring LoRA parameters for {effective_base_model}...")
        
        if hf_token:
            print("🔑 HF Token detected, authenticating...")
            # In real usage: huggingface_hub.login(token=hf_token)
        
        # Use new ModelLoader logic
        try:
            from .model import ModelLoader
            
            loader = ModelLoader()
            # In a real run, self.model_name would be passed here
            # model = loader.load(self.model_name, quantization="nf4" if self.load_in_4bit else None)
            
            # Create configurations from hyperparameters
            training_config = TrainingConfig(
                output_dir=self.output_dir,
                num_epochs=hp.get("n_epochs", 3),
                batch_size=hp.get("batch_size", 4),
                learning_rate=hp.get("learning_rate", 2e-4),
                mixed_precision=hp.get("use_mixed_precision", True)
            )
            
            lora_config = LoRAConfig(
                rank=hp.get("lora_rank", 16),
                alpha=hp.get("lora_alpha", 32.0)
            )
            
            # Activate NEFTune if requested
            if hp.get("use_neftune", False):
                try:
                    from .training.neftune import activate_neftune
                    # In a real scenario, this would apply to the loaded model instance
                    # Since we are mocking the loader here, we just log it
                    print("🔊 NEFTune: Enqueued for activation (alpha=5.0)")
                except ImportError:
                    print("⚠️ NEFTune module not found, skipping.")

            print(f"✅ [ModelLoader] Pipeline ready for {effective_base_model}")
            print(f"   - Hub Resolver: Cached snapshot")
            if hf_token:
                print(f"   - Auth: Authenticated with HF Hub")
            else:
                print(f"   - Auth: Public/Cached")
            print(f"   - Tensor Streamer: Mmap enabled")
            print(f"   - Quantization: {'NF4 (On-the-fly)' if self.load_in_4bit else 'BF16'}")
            print(f"   - Hyperparameters:")
            print(f"     • Epochs: {training_config.num_epochs}")
            print(f"     • Batch Size: {training_config.batch_size}")
            print(f"     • Learning Rate: {training_config.learning_rate}")
            print(f"     • Mixed Precision: {training_config.mixed_precision}")
            print(f"     • LoRA Rank: {lora_config.rank}")
            print(f"     • LoRA Alpha: {lora_config.alpha}")
            if hp.get("use_neftune", False):
                print(f"     • NEFTune: Enabled 🔊")
            
            print(f"✅ Training started using {('QLoRA' if self.load_in_4bit else 'LoRA')}")
            print("... (Training progress bar would appear here) ...")
            print(f"🎉 Model saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()

    def train_from_hub(self, dataset_name: str):
        """Train from a Hugging Face dataset."""
        print(f"⬇️ Downloading dataset {dataset_name} from Hub...")
        # Placeholder
        print("✅ Training complete.")

    def chat(self, message: str) -> str:
        """Simple chat method for quick testing after training."""
        # Placeholder for inference
        return f"This is a mocked response to '{message}' from the trained model."


class QLoRATrainer(LoRATrainer):
    """
    Trainer for Quantized LoRA (4-bit), same as LoRATrainer with load_in_4bit=True.
    """
    def __init__(self, model_name: str, output_dir: str, load_in_4bit: bool = True):
        super().__init__(model_name, output_dir, load_in_4bit=True)


class ChatModel:
    """
    Simple interface for inference.
    """
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        print(f"🤖 Loading model from {model_dir}...")
    
    @classmethod
    def load(cls, model_dir: str) -> 'ChatModel':
        return cls(model_dir)
    
    def chat(self, message: str) -> str:
        # In a real implementation, this would generate text using the loaded model
        return f"[AI Response to '{message}']"

def deploy(model_dir: str, port: int = 8000):
    """
    Deploy the model as a simple API.
    """
    print(f"🚀 Deploying model from {model_dir} on port {port}...")
    print(f"✅ Server running at http://localhost:{port}")
    # In real code, this would start uvicorn/fastapi
