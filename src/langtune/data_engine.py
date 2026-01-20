"""
data_engine.py: Intelligent Data Curation Engine

Implements "Loss-Driven Pruning" - a method to intelligently select
the most valuable training data by analyzing model perplexity.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import torch
from tqdm import tqdm

from .device import DeviceManager

logger = logging.getLogger(__name__)


@dataclass
class RowProfile:
    """Profile for a single data row."""
    index: int
    loss: float
    text_length: int
    tokens: int = 0
    
    @property
    def difficulty(self) -> str:
        """Categorize difficulty based on loss score."""
        if self.loss < 1.0:
            return "trivial"  # Model already knows this
        elif self.loss < 2.5:
            return "easy"
        elif self.loss < 4.5:
            return "medium"  # Sweet spot for learning
        elif self.loss < 7.0:
            return "hard"
        else:
            return "noise"  # Likely garbage or outlier


@dataclass
class DatasetProfile:
    """Complete profile for a dataset after Scout Pass."""
    source_file: str
    total_rows: int
    profiles: List[RowProfile] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def compute_stats(self):
        """Compute aggregate statistics."""
        if not self.profiles:
            return
        
        losses = [p.loss for p in self.profiles]
        self.stats = {
            "mean_loss": sum(losses) / len(losses),
            "min_loss": min(losses),
            "max_loss": max(losses),
            "trivial_count": sum(1 for p in self.profiles if p.difficulty == "trivial"),
            "easy_count": sum(1 for p in self.profiles if p.difficulty == "easy"),
            "medium_count": sum(1 for p in self.profiles if p.difficulty == "medium"),
            "hard_count": sum(1 for p in self.profiles if p.difficulty == "hard"),
            "noise_count": sum(1 for p in self.profiles if p.difficulty == "noise"),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_file": self.source_file,
            "total_rows": self.total_rows,
            "stats": self.stats,
            "profiles": [asdict(p) for p in self.profiles]
        }
    
    def save(self, output_path: Path):
        """Save profile to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "DatasetProfile":
        """Load profile from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        profile = cls(
            source_file=data["source_file"],
            total_rows=data["total_rows"],
            stats=data.get("stats", {})
        )
        profile.profiles = [RowProfile(**p) for p in data.get("profiles", [])]
        return profile


class ScoutPass:
    """
    Run a forward pass over the dataset to score each row's "difficulty".
    
    This is the core of Loss-Driven Pruning. We use the *base model* (before
    fine-tuning) to calculate perplexity for each training example.
    
    - Low loss = "Model already knows this" -> Skip (waste of compute)
    - Medium loss = "Challenging but learnable" -> Keep (high value)
    - Very high loss = "Noise or outlier" -> Skip (harmful to model)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        max_length: int = 512,
        batch_size: int = 4
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = DeviceManager.get_device()
        
        # Ensure model is in eval mode
        self.model.eval()
        self.model.to(self.device)
    
    def _compute_loss(self, text: str) -> float:
        """Compute loss (perplexity proxy) for a single text."""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            
        return loss
    
    def run(self, data_path: Path, text_key: str = "text") -> DatasetProfile:
        """
        Run the scout pass over a JSONL file.
        
        Args:
            data_path: Path to JSONL file.
            text_key: Key in JSON object containing the text (e.g., "text" or "output").
        
        Returns:
            DatasetProfile with loss scores for each row.
        """
        logger.info(f"Starting Scout Pass on {data_path}")
        
        profiles: List[RowProfile] = []
        total_rows = 0
        
        with open(data_path, 'r') as f:
            lines = f.readlines()
            total_rows = len(lines)
        
        for idx, line in enumerate(tqdm(lines, desc="Scout Pass")):
            try:
                row = json.loads(line.strip())
                text = row.get(text_key, "")
                
                if not text:
                    # Try common alternatives
                    text = row.get("output", row.get("response", row.get("content", "")))
                
                if not text:
                    logger.warning(f"Row {idx} has no text content, skipping")
                    continue
                
                loss = self._compute_loss(text)
                
                profiles.append(RowProfile(
                    index=idx,
                    loss=loss,
                    text_length=len(text),
                    tokens=len(self.tokenizer.encode(text))
                ))
                
            except json.JSONDecodeError:
                logger.warning(f"Row {idx} is not valid JSON, skipping")
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
        
        dataset_profile = DatasetProfile(
            source_file=str(data_path),
            total_rows=total_rows,
            profiles=profiles
        )
        dataset_profile.compute_stats()
        
        logger.info(f"Scout Pass complete. Stats: {dataset_profile.stats}")
        return dataset_profile


class PruningStrategy:
    """
    Select optimal training subset based on loss thresholds.
    """
    
    def __init__(
        self,
        min_loss: float = 1.0,
        max_loss: float = 5.0,
        max_rows: Optional[int] = None
    ):
        """
        Args:
            min_loss: Minimum loss threshold (exclude "trivial" data).
            max_loss: Maximum loss threshold (exclude "noise").
            max_rows: Optional cap on number of rows to keep.
        """
        self.min_loss = min_loss
        self.max_loss = max_loss
        self.max_rows = max_rows
    
    def apply(self, profile: DatasetProfile) -> List[int]:
        """
        Apply pruning strategy to get indices of rows to keep.
        
        Returns:
            List of row indices that should be kept for training.
        """
        candidates = [
            p for p in profile.profiles
            if self.min_loss <= p.loss <= self.max_loss
        ]
        
        # Sort by loss (prefer medium difficulty)
        # Optimal order: medium > hard > easy
        candidates.sort(key=lambda p: abs(p.loss - 3.0))
        
        if self.max_rows:
            candidates = candidates[:self.max_rows]
        
        kept_indices = [p.index for p in candidates]
        
        logger.info(
            f"Pruning: Kept {len(kept_indices)}/{profile.total_rows} rows "
            f"({len(kept_indices)/profile.total_rows*100:.1f}%)"
        )
        
        return kept_indices
    
    def extract_subset(
        self,
        source_path: Path,
        indices: List[int],
        output_path: Path
    ):
        """
        Extract the selected rows and write to a new file.
        """
        indices_set = set(indices)
        
        with open(source_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for idx, line in enumerate(f_in):
                if idx in indices_set:
                    f_out.write(line)
        
        logger.info(f"Wrote curated subset to {output_path}")


def quick_scout(data_path: str, output_path: Optional[str] = None) -> DatasetProfile:
    """
    Quick scout without a real model - uses heuristics for testing.
    
    This is useful for UI development when no model is loaded.
    Uses text length and complexity as a proxy for "difficulty".
    """
    import random
    
    profiles: List[RowProfile] = []
    
    with open(data_path, 'r') as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        try:
            row = json.loads(line.strip())
            text = row.get("text", row.get("output", row.get("response", "")))
            
            # Heuristic: longer and more complex = higher "loss"
            length = len(text)
            complexity = len(set(text.split())) / max(1, len(text.split()))
            
            # Simulate a loss score (1.0 - 8.0 range)
            simulated_loss = 1.0 + (length / 500) * 3 + complexity * 2 + random.uniform(-0.5, 0.5)
            simulated_loss = max(0.5, min(9.0, simulated_loss))
            
            profiles.append(RowProfile(
                index=idx,
                loss=simulated_loss,
                text_length=length,
                tokens=len(text.split())  # Rough approximation
            ))
        except:
            pass
    
    dataset_profile = DatasetProfile(
        source_file=data_path,
        total_rows=len(lines),
        profiles=profiles
    )
    dataset_profile.compute_stats()
    
    if output_path:
        dataset_profile.save(Path(output_path))
    
    return dataset_profile
