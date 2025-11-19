#!/usr/bin/env python3
"""
Example training script for Langtune

This script demonstrates how to use Langtune for fine-tuning a language model
with LoRA adapters on text data.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the path so we can import langtune
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langtune import (
    Config, ModelConfig, TrainingConfig, DataConfig, LoRAConfig,
    LoRALanguageModel, Trainer, create_trainer,
    load_dataset_from_config, create_data_loader, DataCollator,
    set_seed, get_device, print_model_summary
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_example_config():
    """Create an example configuration for training."""
    
    # LoRA configuration
    lora_config = LoRAConfig(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=['attention.qkv', 'attention.proj', 'mlp.fc1', 'mlp.fc2'],
        merge_weights=False
    )
    
    # Model configuration
    model_config = ModelConfig(
        vocab_size=10000,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        max_seq_len=512,
        mlp_ratio=4.0,
        dropout=0.1,
        lora=lora_config
    )
    
    # Training configuration
    training_config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_epochs=5,
        warmup_steps=100,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        mixed_precision=False,
        save_steps=500,
        eval_steps=250,
        logging_steps=50,
        save_total_limit=3,
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    # Data configuration
    data_config = DataConfig(
        train_file=None,  # Will use sample data
        eval_file=None,
        test_file=None,
        max_length=512,
        padding="max_length",
        truncation=True,
        tokenizer_name=None,
        cache_dir=None
    )
    
    # Main configuration
    config = Config(
        model=model_config,
        training=training_config,
        data=data_config,
        output_dir="./outputs/example_training",
        seed=42,
        device="auto",
        num_workers=2,
        pin_memory=True
    )
    
    return config

def main():
    """Main training function."""
    logger.info("Starting Langtune example training...")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create configuration
    config = create_example_config()
    logger.info("Configuration created")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load datasets (will create sample data if no files specified)
    try:
        train_dataset, val_dataset, test_dataset = load_dataset_from_config(config)
        logger.info(f"Loaded datasets: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return 1
    
    # Create data loaders
    collate_fn = DataCollator()
    
    train_dataloader = create_data_loader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    val_dataloader = create_data_loader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    test_dataloader = create_data_loader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    # Create trainer
    try:
        trainer = create_trainer(
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader
        )
        logger.info("Trainer created successfully")
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        return 1
    
    # Print model summary
    print_model_summary(trainer.model)
    
    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Test generation
        logger.info("Testing text generation...")
        sample_text = trainer.generate_sample("The quick brown fox", max_length=50)
        logger.info(f"Generated text: {sample_text}")
        
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
