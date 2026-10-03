
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
        Remote training from a langtune config file isn't available: the
        Langtrain API trains from an uploaded dataset, not from a local bundle.
        This used to print a simulated run; it now says what to do instead.
        """
        raise NotImplementedError(
            "Remote training from the langtune CLI isn't available yet.\n"
            "Start a cloud run in the Langtrain dashboard, or from Python with\n"
            "FastLanguageModel.from_pretrained(model, api_key=...) and\n"
            "FastLanguageModel.train(model, tokenizer, None, dataset_id=...)."
        )

    def stream_logs(self, job_id: str) -> None:
        raise NotImplementedError("Remote training from the langtune CLI isn't available yet.")
