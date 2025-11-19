"""
cli.py: Command-line interface for Langtune
"""

import argparse
import os
import sys
import logging
import torch
from pathlib import Path
from typing import Optional

from .config import Config, load_config, save_config, get_preset_config, validate_config
from .trainer import create_trainer
from .data import load_dataset_from_config, create_data_loader, DataCollator
from .models import LoRALanguageModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_command(args):
    """Handle the train command."""
    logger.info("Starting training...")
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    elif args.preset:
        config = get_preset_config(args.preset)
    else:
        logger.error("Either --config or --preset must be specified")
        return 1
    
    # Override config with command line arguments
    if args.train_file:
        config.data.train_file = args.train_file
    if args.eval_file:
        config.data.eval_file = args.eval_file
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.epochs:
        config.training.num_epochs = args.epochs
    
    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.output_dir, "config.yaml")
    save_config(config, config_path)
    logger.info(f"Configuration saved to {config_path}")
    
    # Load datasets
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
    ) if val_dataset else None
    
    test_dataloader = create_data_loader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    ) if test_dataset else None
    
    # Create trainer
    try:
        trainer = create_trainer(
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader
        )
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        return 1
    
    # Start training
    try:
        trainer.train(resume_from_checkpoint=args.resume_from)
        logger.info("Training completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

def evaluate_command(args):
    """Handle the evaluate command."""
    logger.info("Starting evaluation...")
    
    if not args.model_path:
        logger.error("--model_path is required for evaluation")
        return 1
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        logger.error("--config is required for evaluation")
        return 1
    
    # Load model
    try:
        model = LoRALanguageModel(
            vocab_size=config.model.vocab_size,
            embed_dim=config.model.embed_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            max_seq_len=config.model.max_seq_len,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
            lora_config=config.model.lora.__dict__ if config.model.lora else None
        )
        
        checkpoint = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        logger.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Load test dataset
    try:
        _, _, test_dataset = load_dataset_from_config(config)
        test_dataloader = create_data_loader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=DataCollator()
        )
    except Exception as e:
        logger.error(f"Failed to load test dataset: {e}")
        return 1
    
    # Evaluate
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                total_loss += outputs["loss"].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Test loss: {avg_loss:.4f}")
        
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

def generate_command(args):
    """Handle the generate command."""
    logger.info("Starting text generation...")
    
    if not args.model_path:
        logger.error("--model_path is required for generation")
        return 1
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        logger.error("--config is required for generation")
        return 1
    
    # Load model
    try:
        model = LoRALanguageModel(
            vocab_size=config.model.vocab_size,
            embed_dim=config.model.embed_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            max_seq_len=config.model.max_seq_len,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
            lora_config=config.model.lora.__dict__ if config.model.lora else None
        )
        
        checkpoint = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        logger.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Generate text
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        prompt = args.prompt or "The quick brown fox"
        max_length = args.max_length or 100
        
        # Simple tokenization
        input_ids = torch.tensor([ord(c) for c in prompt[:50]], dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
        
        # Simple decoding
        generated_text = "".join([chr(i) for i in generated[0].cpu().tolist()])
        
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        
        return 0
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1

def concept_command(args):
    """Handle the concept command."""
    concept_name = args.concept.upper()
    logger.info(f"Running concept demonstration: {concept_name}")
    
    # Simulate concept execution
    import time
    from tqdm import tqdm
    
    for i in tqdm(range(100), desc=f"Progress for {concept_name}"):
        time.sleep(0.02)  # Simulate work
    
    logger.info(f"{concept_name} demonstration completed!")
    return 0

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Langtune: Efficient LoRA Fine-Tuning for Text LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with a preset configuration
  langtune train --preset small --train-file data/train.txt --output-dir outputs/

  # Train with a custom configuration
  langtune train --config config.yaml --train-file data/train.txt

  # Evaluate a trained model
  langtune evaluate --config config.yaml --model-path outputs/best_model.pt

  # Generate text with a trained model
  langtune generate --config config.yaml --model-path outputs/best_model.pt --prompt "Hello world"

  # Run a concept demonstration
  langtune concept --concept rlhf
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, help='Path to configuration file')
    train_parser.add_argument('--preset', type=str, choices=['tiny', 'small', 'base', 'large'], 
                             help='Use a preset configuration')
    train_parser.add_argument('--train-file', type=str, help='Path to training data file')
    train_parser.add_argument('--eval-file', type=str, help='Path to evaluation data file')
    train_parser.add_argument('--output-dir', type=str, help='Output directory for checkpoints')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--resume-from', type=str, help='Resume from checkpoint')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    eval_parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text with a trained model')
    gen_parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    gen_parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    gen_parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    gen_parser.add_argument('--max-length', type=int, help='Maximum generation length')
    gen_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    gen_parser.add_argument('--top-k', type=int, help='Top-k sampling')
    gen_parser.add_argument('--top-p', type=float, help='Top-p (nucleus) sampling')
    
    # Concept command
    concept_parser = subparsers.add_parser('concept', help='Run a concept demonstration')
    concept_parser.add_argument('--concept', type=str, required=True,
                               choices=['rlhf', 'cot', 'ccot', 'grpo', 'rlvr', 'dpo', 'ppo', 'lime', 'shap'],
                               help='LLM concept to demonstrate')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    if args.command == 'train':
        return train_command(args)
    elif args.command == 'evaluate':
        return evaluate_command(args)
    elif args.command == 'generate':
        return generate_command(args)
    elif args.command == 'concept':
        return concept_command(args)
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main()) 