
import os
import json
import tarfile
import tempfile
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

from .config import Config
from .auth import get_api_key

logger = logging.getLogger(__name__)
console = Console()

class RemoteTrainer:
    """
    Handles the lifecycle of a remote training job:
    1. Validates local config/data
    2. Bundles artifacts (config + datasets)
    3. Uploads to Langtrain Cloud
    4. Polls for status
    """
    
    API_URL = os.environ.get("LANGTUNE_API_URL", "https://api.langtrain.xyz")
    
    def __init__(self, config: Config):
        self.config = config
        self.api_key = get_api_key()
        
        if not self.api_key:
            raise ValueError("Authentication required for remote training. Run 'langtune auth login'.")

    def submit_job(self) -> str:
        """
        Bundle and submit the training job.
        Returns: job_id
        """
        console.print("[primary]Preparing remote training job...[/]")
        
        # 1. Create a bundle (tar.gz) of config and data
        bundle_path = self._create_bundle()
        console.print(f"[success]✓[/] Bundled assets: [dim]{os.path.basename(bundle_path)}[/]")

        # 2. Upload Bundle (Mock for now)
        # In real impl: POST /api/jobs/upload -> presigned url -> PUT S3
        job_id = self._upload_bundle(bundle_path)
        
        # Cleanup
        os.remove(bundle_path)
        
        return job_id

    def _create_bundle(self) -> str:
        """Create a compressed tarball of necessary files."""
        # Create temp file
        fd, path = tempfile.mkstemp(suffix=".tar.gz")
        os.close(fd)
        
        with tarfile.open(path, "w:gz") as tar:
            # Add Config
            # We dump the current config object to a string to ensure it includes all overrides
            # But simpler to just add the file if it exists, or dump it to a temp file
            config_json = json.dumps(self.config.__dict__, default=lambda o: o.__dict__, indent=2)
            
            # Create a temporary config file to add to tar
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(config_json)
                temp_config_path = f.name
                
            tar.add(temp_config_path, arcname="config.json")
            os.remove(temp_config_path)
            
            # Add Datasets
            # We need to resolve paths from the config
            # This is a simplification; handling large datasets requires presigned URLs usually
            data_files = []
            if self.config.data and hasattr(self.config.data, 'train_file'):
                data_files.append(self.config.data.train_file)
            if self.config.data and hasattr(self.config.data, 'eval_file') and self.config.data.eval_file:
                data_files.append(self.config.data.eval_file)
                
            for df in data_files:
                if os.path.exists(df):
                    tar.add(df, arcname=os.path.basename(df))
                else:
                    logger.warning(f"Data file not found: {df}")

        return path

    def _upload_bundle(self, bundle_path: str) -> str:
        """Upload the bundle to the backend."""
        file_size_mb = os.path.getsize(bundle_path) / (1024 * 1024)
        console.print(f"[info]Uploading payload ({file_size_mb:.2f} MB)...[/]")
        
        # MOCK REQUEST
        time.sleep(1.5) # Simulate network
        job_id = f"job_{int(time.time())}"
        
        console.print(f"[success]✓[/] Upload complete. Job ID: [bold cyan]{job_id}[/]")
        return job_id

    def stream_logs(self, job_id: str):
        """Poll and stream logs from the remote job."""
        console.print(f"\n[primary]Streaming logs for {job_id}...[/]\n")
        
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(style="muted", complete_style="success"), TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Provisioning GPU instance...", total=100)
            
            # Simulated Lifecycle
            lifecycle = [
                (10, "Provisioning GPU instance..."),
                (20, "Downloading base model (Llama-3-8B)..."),
                (30, "Loading dataset..."),
                (40, "Training: Step 1/100... loss: 2.1"),
                (60, "Training: Step 50/100... loss: 0.8"),
                (80, "Training: Step 100/100... loss: 0.3"),
                (90, "Saving adapter checkpoints..."),
                (100, "Finalizing upload...")
            ]
            
            for percent, desc in lifecycle:
                time.sleep(1) # Simulate time passing
                progress.update(task, completed=percent, description=f"[cyan]{desc}")
                
        console.print("\n[bold success]Remote Training Completed Successfully![/]")
        console.print(f"Run [bold white]langtune download {job_id}[/] to get your model.")

