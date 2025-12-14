"""
logging_utils.py: Logging utilities for Langtune

Provides colorful logging and progress tracking.
"""

import logging
import sys
from typing import Optional
from datetime import datetime

# Rich console for pretty output
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True
):
    """Setup logging with optional rich formatting."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = []
    
    if use_rich and RICH_AVAILABLE:
        handlers.append(RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            rich_tracebacks=True
        ))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(file_handler)
    
    logging.basicConfig(level=log_level, handlers=handlers, force=True)
    
    # Set langtune logger
    logger = logging.getLogger("langtune")
    logger.setLevel(log_level)


def get_logger(name: str = "langtune") -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


class TrainingLogger:
    """Structured logger for training progress."""
    
    def __init__(self, name: str = "training", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.log_file = log_file
        self.start_time = None
        self.metrics_history = []
    
    def start(self, message: str = "Starting training"):
        """Log training start."""
        self.start_time = datetime.now()
        self.logger.info(f"🚀 {message}")
    
    def log_epoch(self, epoch: int, total_epochs: int, metrics: dict):
        """Log epoch completion."""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"📊 Epoch {epoch+1}/{total_epochs}: {metrics_str}")
        self.metrics_history.append({"epoch": epoch + 1, **metrics})
    
    def log_step(self, step: int, loss: float, lr: Optional[float] = None):
        """Log training step."""
        msg = f"Step {step}: loss={loss:.4f}"
        if lr is not None:
            msg += f", lr={lr:.2e}"
        self.logger.debug(msg)
    
    def log_validation(self, metrics: dict):
        """Log validation results."""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"✓ Validation: {metrics_str}")
    
    def end(self, message: str = "Training complete"):
        """Log training end."""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            self.logger.info(f"🎉 {message} (took {elapsed})")
        else:
            self.logger.info(f"🎉 {message}")
    
    def error(self, message: str):
        """Log error."""
        self.logger.error(f"❌ {message}")
    
    def warning(self, message: str):
        """Log warning."""
        self.logger.warning(f"⚠️ {message}")


class ProgressTracker:
    """Progress bar for training."""
    
    def __init__(self, total: int, description: str = "Training"):
        self.total = total
        self.description = description
        self.current = 0
        self.progress = None
        self.task = None
    
    def __enter__(self):
        if RICH_AVAILABLE:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
            )
            self.progress.start()
            self.task = self.progress.add_task(self.description, total=self.total)
        return self
    
    def __exit__(self, *args):
        if self.progress:
            self.progress.stop()
    
    def update(self, n: int = 1, description: Optional[str] = None):
        """Update progress."""
        self.current += n
        if self.progress and self.task is not None:
            self.progress.update(self.task, advance=n)
            if description:
                self.progress.update(self.task, description=description)
    
    def set_description(self, description: str):
        """Update description."""
        if self.progress and self.task is not None:
            self.progress.update(self.task, description=description)


def print_banner(text: str, style: str = "cyan"):
    """Print a styled banner."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(f"\n[bold {style}]{'='*60}[/]")
        console.print(f"[bold {style}]  {text}[/]")
        console.print(f"[bold {style}]{'='*60}[/]\n")
    else:
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}\n")


def print_metrics(metrics: dict, title: str = "Metrics"):
    """Print metrics in a nice format."""
    if RICH_AVAILABLE:
        from rich.table import Table
        console = Console()
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in metrics.items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(table)
    else:
        print(f"\n{title}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
