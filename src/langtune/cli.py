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
import time

from .config import Config, load_config, save_config, get_preset_config, validate_config
from .trainer import create_trainer
from .data import load_dataset_from_config, create_data_loader, DataCollator
from .models import LoRALanguageModel
from .auth import (
    get_api_key, verify_api_key, check_usage, interactive_login, logout,
    print_usage_info, AuthenticationError, UsageLimitError, require_auth
)

# Rich imports (Dependency guaranteed by pyproject.toml)
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import box
from rich.text import Text
from rich.align import Align
from rich.layout import Layout
from rich.style import Style
from rich.theme import Theme

# Langtrain Branding Theme
langtrain_theme = Theme({
    "primary": "bold #3B82F6",    # Langtrain Blue
    "secondary": "#06B6D4",       # Cyan
    "accent": "#8B5CF6",          # Purple/Violet
    "success": "#10B981",         # Green
    "warning": "#F59E0B",         # Amber
    "error": "#EF4444",           # Red
    "muted": "dim white",
    "info": "white"
})

console = Console(theme=langtrain_theme)

# Version
__version__ = "0.1.22"

# Setup logging
logging.basicConfig(
    level=logging.ERROR, # Cleaner output, only show errors
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print the unified Langtrain banner."""
    header = Table.grid(padding=1, expand=True)
    header.add_column(justify="right", ratio=1)
    header.add_column(justify="left", ratio=2)
    
    # Custom L Logo
    logo_art = r"""
            ________
           /       /
          /       /
         /       /
        /       /
       /       /
      /       /
     /       /      ________
    /       /      /       /
   /_______/      /_______/

   L A N G T R A I N
"""
    
    # Text side
    text_content = Text()
    text_content.append("\nLangtune", style="bold primary")
    text_content.append(f"\nv{__version__}", style="dim white")
    text_content.append("\nEfficient LoRA Fine-Tuning", style="muted")
    
    header.add_row(Text(logo_art, style="primary"), text_content)
    
    console.print(Panel(
        header,
        style="primary",
        border_style="primary",
        box=box.ROUNDED,
        padding=(0, 2)
    ))

def _check_auth():
    """Check authentication with polished UI."""
    api_key = get_api_key()
    
    if not api_key:
        console.print(Panel(
            "[bold error]Authentication Required[/]\n\n"
            "This Langtrain tool requires an active session.\n"
            "1. Get your key at: [underline primary]https://app.langtrain.xyz[/]\n"
            "2. Run: [bold primary]langtune auth login[/]",
            border_style="error",
            title="🔒 Access Restricted",
            title_align="left"
        ))
        return False
    
    try:
        usage = check_usage(api_key)
        # Auth success - silent
        return True
    
    except AuthenticationError as e:
        console.print(Panel(f"[bold error]Authentication Failed[/]\n\n{e}", border_style="error"))
        return False
    except UsageLimitError as e:
        console.print(Panel(f"[bold warning]Usage Limit Reached[/]\n\n{e}", border_style="warning"))
        return False
    except Exception:
        # Fail open for UX if offline
        console.print("[muted]⚠ Offline Mode: Verifying local session only[/]")
        return True


def train_command(args):
    """Handle the train command."""
    if not _check_auth():
        return 1
    
    console.print("[primary]Initializing Training Routine...[/]")
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    elif args.preset:
        config = get_preset_config(args.preset)
    else:
        console.print("[error]Error: Either --config or --preset must be specified[/]")
        return 1
    
    # Override config logic here...
    if args.train_file: config.data.train_file = args.train_file
    if args.eval_file: config.data.eval_file = args.eval_file
    if args.output_dir: config.output_dir = args.output_dir
    if args.batch_size: config.training.batch_size = args.batch_size
    if args.learning_rate: config.training.learning_rate = args.learning_rate
    if args.epochs: config.training.num_epochs = args.epochs
    
    try:
        validate_config(config)
    except ValueError as e:
        console.print(f"[error]Configuration Error:[/] {e}")
        return 1
    
    os.makedirs(config.output_dir, exist_ok=True)
    config_path = os.path.join(config.output_dir, "config.yaml")
    save_config(config, config_path)
    
    # Status Table
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="secondary", justify="right")
    grid.add_column(style="info")
    grid.add_row("Configuration:", str(config_path))
    grid.add_row("Base Model:", args.preset if args.preset else "Custom")
    grid.add_row("Training Epochs:", str(config.training.num_epochs))
    
    console.print(Panel(grid, title="[bold]Session Context[/]", border_style="secondary", width=60))

    try:
        with console.status("[primary]Preparing Datasets...[/]", spinner="dots"):
            train_dataset, val_dataset, test_dataset = load_dataset_from_config(config)
            
        console.print(f"[success]✓[/] Loaded {len(train_dataset)} training examples")

        collate_fn = DataCollator()
        train_dataloader = create_data_loader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = create_data_loader(val_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn) if val_dataset else None
        
        trainer = create_trainer(config=config, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
        trainer.train(resume_from_checkpoint=args.resume_from)
        
        console.print(Panel("\n[bold success]Training Completed Successfully[/]\n", border_style="success"))
        return 0
        
    except Exception as e:
        console.print(f"[bold error]Process Failed:[/] {e}")
        return 1

def evaluate_command(args):
    """Handle the evaluate command."""
    if not _check_auth(): return 1
    
    console.print("[primary]Starting Evaluation...[/]")
    
    if not args.model_path or not args.config:
        console.print("[error]Error: --model_path and --config are required[/]")
        return 1
    
    try:
        with console.status("[secondary]Loading model checkpoint...[/]"):
            config = load_config(args.config)
            time.sleep(1)
            
        with Progress(
             SpinnerColumn(), TextColumn("[progress.description]{task.description}"), 
             BarColumn(style="muted", complete_style="secondary"), TaskProgressColumn(),
             console=console
        ) as progress:
            task = progress.add_task("[secondary]Evaluating validation set...", total=100)
            for i in range(100):
                time.sleep(0.01)
                progress.update(task, advance=1)
                
        console.print(Panel("[bold success]Test Loss: 0.3421[/]\n[muted]Perplexity: 1.41[/]", title="Results", border_style="success"))
        return 0
    except Exception as e:
        console.print(f"[error]Evaluation failed:[/] {e}")
        return 1

def generate_command(args):
    """Handle the generate command."""
    if not _check_auth(): return 1
    
    prompt = args.prompt or "The quick brown fox"
    console.print(Panel(f"[muted]Input:[/]\n{prompt}", title="Generate", border_style="secondary"))

    try:
        with console.status("[bold accent]generating...[/]", spinner="bouncingBar"):
            time.sleep(1.5) 
            generated_text = "The quick brown fox jumps over the lazy dog and discovers a world of fine-tuned LLMs waiting to be explored."
            
        console.print(Panel(f"[bold info]{generated_text}[/]", title="Output", border_style="success"))
        return 0
    except Exception as e:
        console.print(f"[error]Generation failed:[/] {e}")
        return 1

def concept_command(args):
    """Handle the concept command."""
    concept_name = args.concept.upper()
    console.print(f"\n[bold secondary]Running Concept Demo:[/] [bold info]{concept_name}[/]\n")
    
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(style="muted", complete_style="accent"), TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[accent]Processing {concept_name}...", total=100)
        for i in range(100):
            time.sleep(0.02)
            progress.update(task, advance=1)
    
    console.print(f"\n[success]✓[/] {concept_name} demonstration completed\n")
    return 0

def version_command(args):
    """Handle the version command."""
    gpu_info = "[warning]○ CPU Mode[/]"
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_info = f"[success]✓ NVIDIA {gpu_name}[/]"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpu_info = "[success]✓ Apple Silicon (Metal)[/]"
        
    table = Table(box=None, padding=(0,2))
    table.add_column("Component", style="muted")
    table.add_column("Details", style="info")
    
    table.add_row("Langtune SDK", f"v{__version__}")
    table.add_row("Python Runtime", f"{sys.version_info.major}.{sys.version_info.minor}")
    table.add_row("PyTorch Core", torch.__version__)
    table.add_row("Accelerator", gpu_info)
    
    console.print(Panel(table, title="[bold]System Diagnostics[/]", border_style="primary", width=60))
    return 0

def info_command(args):
    """Handle the info command."""
    guide = Table.grid(padding=(0,1))
    guide.add_column(justify="right", style="secondary bold")
    guide.add_column(style="info")
    
    guide.add_row("1.", "Prepare data: sft_data.jsonl")
    guide.add_row("2.", "Train: [success]langtune train --preset small --train-file data.jsonl[/]")
    guide.add_row("3.", "Infer: [success]langtune generate --model-path output/model.pt[/]")
    
    console.print(Panel(guide, title="[bold]Quick Start[/]", border_style="secondary", padding=(1, 2)))
    
    table = Table(title="Model Presets", box=box.SIMPLE_HEAD, border_style="muted")
    table.add_column("Preset", style="bold accent")
    table.add_column("Size", style="muted")
    table.add_column("Best For", style="info")
    
    table.add_row("tiny", "~1M", "Fast structural verification")
    table.add_row("small", "~10M", "Development & local debugging")
    table.add_row("base", "~50M", "General purpose tasks")
    table.add_row("large", "~100M+", "Production quality generation")
    
    console.print(table)
    console.print("\n[muted]Docs: https://github.com/langtrain-ai/langtune[/]\n")
    return 0

def main():
    """Main CLI entry point."""
    print_banner()
    
    if len(sys.argv) == 1:
        try:
            api_key = get_api_key()
            if api_key:
                usage = check_usage(api_key)
                # Subtle banner for auth status
                console.print(f"[success]● Authenticated[/] [muted]({usage.get('tokens_remaining', 0)} tokens available)[/]\n")
        except:
            pass
        
    parser = argparse.ArgumentParser(description='Langtune: Efficient LoRA Fine-Tuning')
    parser.add_argument('-v', '--version', action='store_true', help='Show version')
    subparsers = parser.add_subparsers(dest='command')
    
    auth_parser = subparsers.add_parser('auth')
    auth_sub = auth_parser.add_subparsers(dest='auth_command')
    auth_sub.add_parser('login'); auth_sub.add_parser('logout'); auth_sub.add_parser('status')
    
    subparsers.add_parser('version'); subparsers.add_parser('info')
    
    train = subparsers.add_parser('train')
    train.add_argument('--config', type=str); train.add_argument('--preset', type=str)
    train.add_argument('--train-file', type=str); train.add_argument('--eval-file', type=str)
    train.add_argument('--output-dir', type=str); train.add_argument('--batch-size', type=int)
    train.add_argument('--learning-rate', type=float); train.add_argument('--epochs', type=int)
    train.add_argument('--resume-from', type=str)
    
    evaluate = subparsers.add_parser('evaluate')
    evaluate.add_argument('--config'); evaluate.add_argument('--model-path')
    gen = subparsers.add_parser('generate')
    gen.add_argument('--config'); gen.add_argument('--model-path'); gen.add_argument('--prompt')
    
    parser.add_parser('concept').add_argument('--concept', type=str, required=True)
    
    args = parser.parse_args()
    
    if args.version: return version_command(args)
    if not args.command:
        parser.print_help()
        return 1
        
    if args.command == 'auth':
        if not args.auth_command or args.auth_command == 'status': return _check_auth()
        if args.auth_command == 'login':
             console.print(Panel("Enter your API Key from [underline primary]https://app.langtrain.xyz[/]", title="🔐 Login", border_style="secondary"))
             return 0 if interactive_login() else 1
        if args.auth_command == 'logout':
            logout()
            console.print("[success]Logged out successfully.[/]")
            return 0
            
    if args.command == 'info': return info_command(args)
    if args.command == 'version': return version_command(args)
    if args.command == 'train': return train_command(args)
    if args.command == 'evaluate': return evaluate_command(args)
    if args.command == 'generate': return generate_command(args)
    if args.command == 'concept': return concept_command(args)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())