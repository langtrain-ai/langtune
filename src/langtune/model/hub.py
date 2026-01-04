"""
Hub Resolver for efficient model downloads.

Handles interactions with the Hugging Face Hub, including:
- Snapshot downloads with caching
- Authentication
- Offline mode support
- File resolution
"""

import os
import logging
from typing import Optional, List, Dict, Union, Path
from pathlib import Path as PathLib
from huggingface_hub import snapshot_download, get_token, HfFolder
from huggingface_hub.utils import LocalEntryNotFoundError, EntryNotFoundError, RevisionNotFoundError, RepositoryNotFoundError

logger = logging.getLogger(__name__)

class HubResolver:
    """
    Resolves and downloads model files from the Hugging Face Hub.
    
    Optimized for:
    - Speed (concurrent downloads)
    - Caching (avoid re-downloading)
    - Offline usage (finding local files)
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        token: Optional[str] = None,
    ):
        self.cache_dir = cache_dir or os.environ.get("LANGTUNE_CACHE_DIR")
        self.local_files_only = local_files_only
        self.token = token or get_token()
        
    def resolve(
        self,
        model_id: str,
        revision: str = "main",
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ) -> PathLib:
        """
        Download or find model snapshot.
        
        Args:
            model_id: The model ID on HF Hub (e.g. 'meta-llama/Llama-2-7b-hf')
            revision: Branch or commit hash
            allow_patterns: Files to include (default: ['*.safetensors', '*.json', '*.model'])
            ignore_patterns: Files to exclude (default: ['*.bin', '*.pth'])
            
        Returns:
            Path to the directory containing the model files.
        """
        # Default patterns to prioritize safetensors and configs
        if allow_patterns is None:
            allow_patterns = ["*.safetensors", "*.json", "*.model", "tokenizer*"]
            
        if ignore_patterns is None:
            # Explicitly ignore pytorch/pickle weights if we want to enforce safetensors
            ignore_patterns = ["*.bin", "*.pth", "*.pt"]

        logger.info(f"Resolving model {model_id} (revision={revision})...")
        
        try:
            download_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
                token=self.token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                resume_download=True,
                max_workers=8, # Parallel downloads
                tqdm_class=None, # We might want to hook a custom progress bar later
            )
            
            logger.info(f"Model resolved to: {download_path}")
            return PathLib(download_path)
            
        except (LocalEntryNotFoundError, EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError) as e:
            if self.local_files_only:
                logger.error(f"Model {model_id} not found locally and offline mode is enabled.")
                raise RuntimeError(f"Model {model_id} not found locally. Disable offline mode to download.") from e
            else:
                logger.error(f"Failed to download model {model_id}: {e}")
                raise RuntimeError(f"Failed to resolve model {model_id}: {e}") from e
                
        except Exception as e:
            logger.error(f"Unexpected error resolving model {model_id}: {e}")
            raise

    @property
    def is_offline(self) -> bool:
        return self.local_files_only

# Global instance capabilities
_default_resolver = None

def get_resolver(offline: bool = False, cache_dir: Optional[str] = None) -> HubResolver:
    global _default_resolver
    if _default_resolver is None or _default_resolver.local_files_only != offline:
        _default_resolver = HubResolver(local_files_only=offline, cache_dir=cache_dir)
    return _default_resolver
