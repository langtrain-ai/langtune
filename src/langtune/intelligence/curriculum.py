"""
Curriculum Learning Engine

Orders training samples from easy to hard for more stable training.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import math


class Difficulty(Enum):
    """Sample difficulty levels."""
    EASY = 1
    MEDIUM = 2
    HARD = 3


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    
    # Phases as (start_pct, end_pct, difficulty)
    phases: List[tuple] = None
    
    # Dynamic adjustment
    enable_dynamic_adjustment: bool = True
    
    # Loss plateau threshold for adjustment
    plateau_threshold: float = 0.01
    
    # Window size for loss monitoring
    loss_window: int = 50
    
    def __post_init__(self):
        if self.phases is None:
            # Default: 30% easy, 40% medium, 30% hard
            self.phases = [
                (0.0, 0.3, Difficulty.EASY),
                (0.3, 0.7, Difficulty.MEDIUM),
                (0.7, 1.0, Difficulty.HARD),
            ]


class DifficultyClassifier:
    """
    Classifies samples by difficulty.
    
    Uses multiple signals:
    - Token length
    - Entropy
    - Vocabulary diversity
    - Structure complexity
    """
    
    def __init__(
        self,
        easy_max_tokens: int = 100,
        hard_min_tokens: int = 400,
        low_entropy_threshold: float = 3.5,
        high_entropy_threshold: float = 5.0,
    ):
        self.easy_max_tokens = easy_max_tokens
        self.hard_min_tokens = hard_min_tokens
        self.low_entropy = low_entropy_threshold
        self.high_entropy = high_entropy_threshold
    
    def classify(self, text: str, tokenizer: Optional[Any] = None) -> Difficulty:
        """Classify a single sample's difficulty."""
        # Token count
        if tokenizer:
            tokens = len(tokenizer.encode(text))
        else:
            tokens = len(text.split())
        
        # Entropy
        entropy = self._compute_entropy(text)
        
        # Vocabulary diversity
        words = text.lower().split()
        vocab_diversity = len(set(words)) / len(words) if words else 0
        
        # Score calculation
        score = 0.0
        
        # Length factor (0-1)
        if tokens < self.easy_max_tokens:
            score += 0.0
        elif tokens > self.hard_min_tokens:
            score += 1.0
        else:
            score += (tokens - self.easy_max_tokens) / (self.hard_min_tokens - self.easy_max_tokens)
        
        # Entropy factor (0-1)
        if entropy < self.low_entropy:
            score += 0.0
        elif entropy > self.high_entropy:
            score += 1.0
        else:
            score += (entropy - self.low_entropy) / (self.high_entropy - self.low_entropy)
        
        # Vocab diversity factor
        score += vocab_diversity
        
        # Normalize (0-3 range / 3)
        normalized = score / 3.0
        
        if normalized < 0.33:
            return Difficulty.EASY
        elif normalized < 0.66:
            return Difficulty.MEDIUM
        return Difficulty.HARD
    
    def classify_batch(
        self, 
        texts: List[str], 
        tokenizer: Optional[Any] = None
    ) -> Dict[Difficulty, List[int]]:
        """Classify a batch of samples, return indices by difficulty."""
        result = {Difficulty.EASY: [], Difficulty.MEDIUM: [], Difficulty.HARD: []}
        
        for i, text in enumerate(texts):
            difficulty = self.classify(text, tokenizer)
            result[difficulty].append(i)
        
        return result
    
    def _compute_entropy(self, text: str) -> float:
        """Compute Shannon entropy."""
        from collections import Counter
        
        if not text:
            return 0.0
        
        counter = Counter(text.lower())
        length = len(text)
        
        return -sum(
            (count / length) * math.log2(count / length)
            for count in counter.values()
            if count > 0
        )


class CurriculumEngine:
    """
    Manages curriculum-based training ordering.
    
    Features:
    - Phase-based training (easy -> medium -> hard)
    - Dynamic adjustment based on loss
    - Smooth difficulty transitions
    """
    
    def __init__(
        self,
        config: CurriculumConfig = None,
        classifier: DifficultyClassifier = None,
    ):
        self.config = config or CurriculumConfig()
        self.classifier = classifier or DifficultyClassifier()
        self.loss_history: List[float] = []
        self.current_phase = 0
    
    def order_dataset(
        self,
        dataset: List[Dict[str, Any]],
        text_field: str = "text",
        tokenizer: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Order dataset samples by curriculum.
        
        Returns samples ordered: easy -> medium -> hard
        """
        # Classify all samples
        texts = [sample.get(text_field, "") for sample in dataset]
        classifications = self.classifier.classify_batch(texts, tokenizer)
        
        # Order by difficulty
        ordered = []
        for difficulty in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
            for idx in classifications[difficulty]:
                ordered.append(dataset[idx])
        
        return ordered
    
    def get_phase_samples(
        self,
        dataset: List[Dict[str, Any]],
        progress: float,
        text_field: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Get samples appropriate for current training progress.
        
        Args:
            dataset: Full ordered dataset
            progress: Training progress (0.0 to 1.0)
            text_field: Key for text content
            
        Returns:
            Filtered samples for current phase
        """
        # Find current phase
        current_difficulty = Difficulty.EASY
        for start, end, difficulty in self.config.phases:
            if start <= progress < end:
                current_difficulty = difficulty
                break
        
        # Filter samples
        texts = [sample.get(text_field, "") for sample in dataset]
        filtered = []
        
        for i, sample in enumerate(dataset):
            sample_difficulty = self.classifier.classify(texts[i])
            
            # Include samples up to current difficulty
            if sample_difficulty.value <= current_difficulty.value:
                filtered.append(sample)
        
        return filtered
    
    def should_advance_difficulty(self, current_loss: float) -> bool:
        """
        Check if training should advance to harder samples.
        
        Based on loss plateau detection.
        """
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) < self.config.loss_window:
            return False
        
        # Check for plateau
        recent = self.loss_history[-self.config.loss_window:]
        avg_recent = sum(recent) / len(recent)
        older = self.loss_history[-2 * self.config.loss_window:-self.config.loss_window]
        
        if older:
            avg_older = sum(older) / len(older)
            improvement = (avg_older - avg_recent) / avg_older if avg_older > 0 else 0
            
            # If improvement is below threshold, advance
            if improvement < self.config.plateau_threshold:
                return True
        
        return False
    
    def inject_harder_samples(
        self,
        current_batch: List[Dict[str, Any]],
        hard_samples: List[Dict[str, Any]],
        ratio: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Inject harder samples into current batch."""
        num_hard = int(len(current_batch) * ratio)
        if num_hard > 0 and hard_samples:
            # Replace some samples with harder ones
            import random
            hard_selection = random.sample(hard_samples, min(num_hard, len(hard_samples)))
            return current_batch[:-num_hard] + hard_selection
        return current_batch
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get curriculum metrics."""
        return {
            "current_phase": self.current_phase,
            "loss_history_length": len(self.loss_history),
            "recent_avg_loss": (
                sum(self.loss_history[-10:]) / 10
                if len(self.loss_history) >= 10
                else None
            ),
        }
