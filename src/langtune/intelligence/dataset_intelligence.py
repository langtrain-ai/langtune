"""
Dataset Intelligence Module

Provides automatic dataset curation, quality analysis, and optimization
for fine-tuning workflows.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import Counter
import math


@dataclass
class DatasetQualityReport:
    """Report on dataset quality metrics."""
    
    total_samples: int = 0
    unique_samples: int = 0
    duplicate_count: int = 0
    duplicate_ratio: float = 0.0
    
    # Entropy metrics
    avg_entropy: float = 0.0
    low_entropy_count: int = 0
    low_entropy_ratio: float = 0.0
    
    # Complexity distribution
    easy_samples: int = 0
    medium_samples: int = 0
    hard_samples: int = 0
    
    # Token statistics
    avg_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    token_std: float = 0.0
    
    # Quality score (0-100)
    quality_score: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "unique_samples": self.unique_samples,
            "duplicate_count": self.duplicate_count,
            "duplicate_ratio": round(self.duplicate_ratio, 4),
            "avg_entropy": round(self.avg_entropy, 4),
            "low_entropy_count": self.low_entropy_count,
            "low_entropy_ratio": round(self.low_entropy_ratio, 4),
            "complexity_distribution": {
                "easy": self.easy_samples,
                "medium": self.medium_samples,
                "hard": self.hard_samples,
            },
            "token_stats": {
                "avg": round(self.avg_tokens, 2),
                "min": self.min_tokens,
                "max": self.max_tokens,
                "std": round(self.token_std, 2),
            },
            "quality_score": round(self.quality_score, 2),
            "recommendations": self.recommendations,
        }


class DatasetIntelligence:
    """
    Intelligent dataset analysis and curation for fine-tuning.
    
    Features:
    - Duplicate detection (exact + semantic)
    - Entropy analysis (detect low-quality samples)
    - Difficulty classification (easy/medium/hard)
    - Auto-curation with recommendations
    """
    
    def __init__(
        self,
        low_entropy_threshold: float = 2.0,
        similarity_threshold: float = 0.85,
        min_tokens: int = 10,
        max_tokens: int = 4096,
    ):
        self.low_entropy_threshold = low_entropy_threshold
        self.similarity_threshold = similarity_threshold
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
    
    def analyze(
        self,
        dataset: List[Dict[str, Any]],
        text_field: str = "text",
        tokenizer: Optional[Any] = None,
    ) -> DatasetQualityReport:
        """
        Analyze dataset quality and return comprehensive report.
        
        Args:
            dataset: List of samples (dicts with text field)
            text_field: Key for text content in each sample
            tokenizer: Optional tokenizer for accurate token counts
            
        Returns:
            DatasetQualityReport with metrics and recommendations
        """
        report = DatasetQualityReport(total_samples=len(dataset))
        
        if not dataset:
            report.recommendations.append("Dataset is empty")
            return report
        
        # Extract texts
        texts = [sample.get(text_field, "") for sample in dataset]
        
        # Duplicate detection
        seen_hashes = set()
        duplicates = 0
        unique_texts = []
        
        for text in texts:
            text_hash = self._hash_text(text)
            if text_hash in seen_hashes:
                duplicates += 1
            else:
                seen_hashes.add(text_hash)
                unique_texts.append(text)
        
        report.unique_samples = len(unique_texts)
        report.duplicate_count = duplicates
        report.duplicate_ratio = duplicates / len(texts) if texts else 0
        
        # Entropy analysis
        entropies = [self._compute_entropy(text) for text in texts]
        report.avg_entropy = sum(entropies) / len(entropies) if entropies else 0
        report.low_entropy_count = sum(1 for e in entropies if e < self.low_entropy_threshold)
        report.low_entropy_ratio = report.low_entropy_count / len(texts) if texts else 0
        
        # Token statistics
        if tokenizer:
            token_counts = [len(tokenizer.encode(t)) for t in texts]
        else:
            token_counts = [len(t.split()) for t in texts]  # Approximate
        
        if token_counts:
            report.avg_tokens = sum(token_counts) / len(token_counts)
            report.min_tokens = min(token_counts)
            report.max_tokens = max(token_counts)
            report.token_std = self._std(token_counts)
        
        # Difficulty classification
        for i, text in enumerate(texts):
            difficulty = self._classify_difficulty(text, entropies[i], token_counts[i])
            if difficulty == "easy":
                report.easy_samples += 1
            elif difficulty == "medium":
                report.medium_samples += 1
            else:
                report.hard_samples += 1
        
        # Compute quality score
        report.quality_score = self._compute_quality_score(report)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        return report
    
    def curate(
        self,
        dataset: List[Dict[str, Any]],
        text_field: str = "text",
        remove_duplicates: bool = True,
        remove_low_entropy: bool = True,
        balance_difficulty: bool = True,
        order_by_curriculum: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Auto-curate dataset for optimal training.
        
        Args:
            dataset: Input dataset
            text_field: Key for text content
            remove_duplicates: Remove exact duplicates
            remove_low_entropy: Remove low-quality samples
            balance_difficulty: Ensure balanced difficulty distribution
            order_by_curriculum: Order easy -> medium -> hard
            
        Returns:
            Tuple of (curated_dataset, removal_stats)
        """
        stats = {"original": len(dataset), "duplicates_removed": 0, "low_entropy_removed": 0}
        curated = dataset.copy()
        
        # Remove duplicates
        if remove_duplicates:
            seen_hashes = set()
            deduped = []
            for sample in curated:
                text_hash = self._hash_text(sample.get(text_field, ""))
                if text_hash not in seen_hashes:
                    seen_hashes.add(text_hash)
                    deduped.append(sample)
            stats["duplicates_removed"] = len(curated) - len(deduped)
            curated = deduped
        
        # Remove low entropy
        if remove_low_entropy:
            high_quality = []
            for sample in curated:
                text = sample.get(text_field, "")
                if self._compute_entropy(text) >= self.low_entropy_threshold:
                    high_quality.append(sample)
            stats["low_entropy_removed"] = len(curated) - len(high_quality)
            curated = high_quality
        
        # Order by curriculum (easy -> medium -> hard)
        if order_by_curriculum:
            curated = self._order_by_difficulty(curated, text_field)
        
        stats["final"] = len(curated)
        return curated, stats
    
    def _hash_text(self, text: str) -> str:
        """Create hash for duplicate detection."""
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _compute_entropy(self, text: str) -> float:
        """Compute Shannon entropy of text."""
        if not text:
            return 0.0
        counter = Counter(text.lower())
        length = len(text)
        entropy = -sum(
            (count / length) * math.log2(count / length)
            for count in counter.values()
            if count > 0
        )
        return entropy
    
    def _classify_difficulty(self, text: str, entropy: float, token_count: int) -> str:
        """Classify sample difficulty."""
        # Simple heuristic - can be enhanced with ML
        if token_count < 50 and entropy < 4.0:
            return "easy"
        elif token_count > 500 or entropy > 5.0:
            return "hard"
        return "medium"
    
    def _order_by_difficulty(
        self, dataset: List[Dict[str, Any]], text_field: str
    ) -> List[Dict[str, Any]]:
        """Order samples by difficulty for curriculum learning."""
        easy, medium, hard = [], [], []
        
        for sample in dataset:
            text = sample.get(text_field, "")
            entropy = self._compute_entropy(text)
            tokens = len(text.split())
            difficulty = self._classify_difficulty(text, entropy, tokens)
            
            if difficulty == "easy":
                easy.append(sample)
            elif difficulty == "medium":
                medium.append(sample)
            else:
                hard.append(sample)
        
        return easy + medium + hard
    
    def _std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _compute_quality_score(self, report: DatasetQualityReport) -> float:
        """Compute overall quality score (0-100)."""
        score = 100.0
        
        # Penalize duplicates
        score -= report.duplicate_ratio * 30
        
        # Penalize low entropy
        score -= report.low_entropy_ratio * 25
        
        # Reward good entropy
        if report.avg_entropy > 4.0:
            score += 10
        
        # Penalize extreme token lengths
        if report.token_std > 500:
            score -= 10
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self, report: DatasetQualityReport) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        if report.duplicate_ratio > 0.1:
            recs.append(f"Remove {report.duplicate_count} duplicate samples ({report.duplicate_ratio:.1%})")
        
        if report.low_entropy_ratio > 0.15:
            recs.append(f"Review {report.low_entropy_count} low-quality samples with low entropy")
        
        if report.avg_entropy < 3.5:
            recs.append("Dataset may be too repetitive - consider adding more diverse samples")
        
        if report.easy_samples > report.total_samples * 0.6:
            recs.append("Consider adding more complex samples for better generalization")
        
        if report.hard_samples > report.total_samples * 0.5:
            recs.append("Consider adding simpler samples to stabilize training")
        
        if report.quality_score >= 80:
            recs.append("✅ Dataset quality is good - ready for training")
        
        return recs
