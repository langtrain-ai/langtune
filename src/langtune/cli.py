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
from .auth import (
    get_api_key, verify_api_key, check_usage, interactive_login, logout,
    print_usage_info, AuthenticationError, UsageLimitError, require_auth
)

# Try to import rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich import box
    from rich.text import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Version
__version__ = "0.1.2"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _check_auth():
    """Check authentication before running protected commands."""
    api_key = get_api_key()
    
    if not api_key:
        if RICH_AVAILABLE:
            console.print("\n[bold red]🔐 Authentication Required[/]\n")
            console.print("Langtune requires an API key to run. Get your free key at:")
            console.print("[blue underline]https://app.langtrain.xyz[/]\n")
            console.print("Then authenticate with: [cyan]langtune auth login[/]\n")
        else:
            print("\n🔐 Authentication Required\n")
            print("Get your API key at: https://app.langtrain.xyz")
            print("Then run: langtune auth login\n")
        return False
    
    try:
        usage = check_usage(api_key)
        if RICH_AVAILABLE:
            remaining = f"{usage['tokens_remaining']:,}"
            console.print(f"[dim]Tokens remaining: {remaining}[/]")
        return True
    except AuthenticationError as e:
        if RICH_AVAILABLE:
            console.print(f"[red]❌ {e}[/]")
        else:
            print(f"❌ {e}")
        return False
    except UsageLimitError as e:
        if RICH_AVAILABLE:
            console.print(f"[yellow]⚠️ {e}[/]")
        else:
            print(f"⚠️ {e}")
        return False


def train_command(args):
    """Handle the train command."""
    # Check authentication first
    if not _check_auth():
        return 1
    
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
    # Check authentication first
    if not _check_auth():
        return 1
    
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
    # Check authentication first
    if not _check_auth():
        return 1
    
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
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]🧪 Running concept demonstration:[/] [bold magenta]{concept_name}[/]\n")
    else:
        logger.info(f"Running concept demonstration: {concept_name}")
    
    # Simulate concept execution with rich progress
    import time
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Processing {concept_name}...", total=100)
            for i in range(100):
                time.sleep(0.02)
                progress.update(task, advance=1)
        
        console.print(f"\n[bold green]✓[/] {concept_name} demonstration completed!\n")
    else:
        from tqdm import tqdm
        for i in tqdm(range(100), desc=f"Progress for {concept_name}"):
            time.sleep(0.02)
        logger.info(f"{concept_name} demonstration completed!")
    
    return 0


def _check_tpu() -> bool:
    """Check if Google TPU is available via torch_xla."""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        return "TPU" in str(device) or "xla" in str(device).lower()
    except:
        return False


def _get_tpu_info() -> str:
    """Get TPU information if available."""
    try:
        import os
        import torch_xla.core.xla_model as xm
        tpu_name = os.environ.get("TPU_NAME", "")
        tpu_cores = xm.xrt_world_size()
        
        # Detect version
        if "v4" in tpu_name.lower():
            version = "v4"
        elif "v3" in tpu_name.lower():
            version = "v3"
        elif "v2" in tpu_name.lower():
            version = "v2"
        else:
            version = ""
        
        return f"{version} ({tpu_cores} cores)"
    except:
        return "(available)"


def version_command(args):
    """Handle the version command."""
    if RICH_AVAILABLE:
        # Check accelerator availability with detailed info
        accelerator_type = "None"
        
        # Check for NVIDIA CUDA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                cuda_version = torch.version.cuda
                
                if gpu_count > 1:
                    gpu_info = f"[green]✓ NVIDIA {gpu_name} × {gpu_count} ({gpu_memory:.0f}GB each)[/]"
                else:
                    gpu_info = f"[green]✓ NVIDIA {gpu_name} ({gpu_memory:.0f}GB)[/]"
                
                accelerator_type = f"CUDA {cuda_version}"
            except:
                gpu_info = f"[green]✓ NVIDIA {gpu_name}[/]"
                accelerator_type = "CUDA"
        # Check for Google TPU
        elif _check_tpu():
            tpu_info = _get_tpu_info()
            gpu_info = f"[green]✓ Google TPU {tpu_info}[/]"
            accelerator_type = "TPU (torch_xla)"
        # Check for Apple MPS
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info = "[green]✓ Apple Metal Performance Shaders (MPS)[/]"
            accelerator_type = "Metal"
        else:
            gpu_info = "[yellow]○ Not available (CPU mode)[/]"
            accelerator_type = "CPU"
        
        table = Table(title="Langtune System Info", box=box.ROUNDED, title_style="bold magenta")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        
        table.add_row("Langtune Version", f"v{__version__}")
        table.add_row("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        table.add_row("PyTorch Version", torch.__version__)
        table.add_row("Accelerator", gpu_info)
        table.add_row("Backend", accelerator_type)
        
        console.print()
        console.print(table)
        console.print()
    else:
        print(f"Langtune v{__version__}")
        print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print(f"PyTorch {torch.__version__}")
        print(f"CUDA: {'Available' if torch.cuda.is_available() else 'Not available'}")
        print(f"TPU: {'Available' if _check_tpu() else 'Not available'}")
    
    return 0


def info_command(args):
    """Handle the info command - show quick start guide."""
    if RICH_AVAILABLE:
        console.print()
        
        # Quick start panel
        quick_start = Text()
        quick_start.append("1. Prepare your data\n", style="bold cyan")
        quick_start.append("   Place your training text in a .txt or .json file\n\n")
        quick_start.append("2. Start training\n", style="bold cyan")
        quick_start.append("   langtune train --preset small --train-file data.txt\n\n", style="green")
        quick_start.append("3. Evaluate your model\n", style="bold cyan")
        quick_start.append("   langtune evaluate --config config.yaml --model-path model.pt\n\n", style="green")
        quick_start.append("4. Generate text\n", style="bold cyan")
        quick_start.append("   langtune generate --config config.yaml --model-path model.pt --prompt \"Hello\"\n", style="green")
        
        panel = Panel(
            quick_start,
            title="[bold]🚀 Quick Start Guide[/]",
            border_style="cyan",
            box=box.ROUNDED
        )
        console.print(panel)
        
        # Available presets
        presets_table = Table(title="Available Model Presets", box=box.SIMPLE)
        presets_table.add_column("Preset", style="cyan bold")
        presets_table.add_column("Parameters", style="white")
        presets_table.add_column("Use Case", style="dim")
        
        presets_table.add_row("tiny", "~1M", "Quick experiments, testing")
        presets_table.add_row("small", "~10M", "Small datasets, fast training")
        presets_table.add_row("base", "~50M", "General purpose")
        presets_table.add_row("large", "~100M+", "Large datasets, best quality")
        
        console.print(presets_table)
        console.print()
        
        # Links
        console.print("[dim]📚 Documentation:[/] [blue underline]https://github.com/langtrain-ai/langtune[/]")
        console.print("[dim]🐛 Report issues:[/] [blue underline]https://github.com/langtrain-ai/langtune/issues[/]")
        console.print()
    else:
        print("""
🚀 Quick Start Guide
====================

1. Prepare your data
   Place your training text in a .txt or .json file

2. Start training
   langtune train --preset small --train-file data.txt

3. Evaluate your model
   langtune evaluate --config config.yaml --model-path model.pt

4. Generate text
   langtune generate --config config.yaml --model-path model.pt --prompt "Hello"

Available Presets: tiny, small, base, large

📚 Docs: https://github.com/langtrain-ai/langtune
""")
    
    return 0


def _print_banner():
    """Print the CLI banner."""
    if RICH_AVAILABLE:
        banner = Text()
        banner.append("\n╔═══════════════════════════════════════════════════════════╗\n", style="cyan bold")
        banner.append("║", style="cyan bold")
        banner.append("                        ", style="")
        banner.append("LANGTUNE", style="bold magenta")
        banner.append("                         ", style="")
        banner.append("║\n", style="cyan bold")
        banner.append("║", style="cyan bold")
        banner.append("          Efficient LoRA Fine-Tuning for LLMs          ", style="dim")
        banner.append("║\n", style="cyan bold")
        banner.append("╚═══════════════════════════════════════════════════════════╝\n", style="cyan bold")
        console.print(banner)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Langtune: Efficient LoRA Fine-Tuning for Text LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  langtune info                                              # Show quick start guide
  langtune version                                           # Show version info
  langtune train --preset small --train-file data.txt        # Train with preset
  langtune train --config config.yaml                        # Train with config
  langtune evaluate --config config.yaml --model-path m.pt   # Evaluate model
  langtune generate --config c.yaml --model-path m.pt        # Generate text
  langtune concept --concept rlhf                            # Concept demo

Learn more: https://github.com/langtrain-ai/langtune
        """
    )
    
    parser.add_argument('-v', '--version', action='store_true', help='Show version information')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Auth command
    auth_parser = subparsers.add_parser('auth', help='Manage API key authentication')
    auth_subparsers = auth_parser.add_subparsers(dest='auth_command', help='Auth commands')
    auth_subparsers.add_parser('login', help='Login with your API key')
    auth_subparsers.add_parser('logout', help='Remove stored API key')
    auth_subparsers.add_parser('status', help='Show authentication status and usage')
    
    # Version command
    subparsers.add_parser('version', help='Show version and system information')
    
    # Info command
    subparsers.add_parser('info', help='Show quick start guide and documentation')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model with LoRA')
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
    
    # Optimization flags
    train_parser.add_argument('--fast', action='store_true', 
                             help='Use FastLoRALanguageModel with all optimizations (RoPE, flash attention, grad checkpointing)')
    train_parser.add_argument('--4bit', dest='use_4bit', action='store_true',
                             help='Use 4-bit quantization (QLoRA style)')
    train_parser.add_argument('--gradient-checkpointing', action='store_true',
                             help='Enable gradient checkpointing to reduce memory')
    train_parser.add_argument('--mixed-precision', type=str, choices=['fp16', 'bf16', 'fp32'], 
                             default='fp16', help='Mixed precision training mode')
    train_parser.add_argument('--gradient-accumulation', type=int, default=1,
                             help='Number of gradient accumulation steps')
    
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
    
    # Handle -v/--version flag
    if args.version:
        return version_command(args)
    
    if not args.command:
        _print_banner()
        parser.print_help()
        if RICH_AVAILABLE:
            console.print("\n[dim]💡 Tip: Run[/] [cyan]langtune info[/] [dim]for a quick start guide[/]\n")
        return 1
    
    # Route to appropriate command handler
    if args.command == 'auth':
        if not args.auth_command:
            # Show auth help
            if RICH_AVAILABLE:
                console.print("\n[bold cyan]🔐 Authentication Commands[/]\n")
                console.print("  [cyan]langtune auth login[/]   - Login with your API key")
                console.print("  [cyan]langtune auth logout[/]  - Remove stored API key")
                console.print("  [cyan]langtune auth status[/]  - Show auth status and usage\n")
                console.print("[dim]Get your API key at:[/] [blue underline]https://app.langtrain.xyz[/]\n")
            else:
                print("\nAuthentication Commands:\n")
                print("  langtune auth login   - Login with your API key")
                print("  langtune auth logout  - Remove stored API key")
                print("  langtune auth status  - Show auth status and usage\n")
            return 0
        elif args.auth_command == 'login':
            return 0 if interactive_login() else 1
        elif args.auth_command == 'logout':
            logout()
            return 0
        elif args.auth_command == 'status':
            print_usage_info()
            return 0
    elif args.command == 'version':
        return version_command(args)
    elif args.command == 'info':
        return info_command(args)
    elif args.command == 'train':
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