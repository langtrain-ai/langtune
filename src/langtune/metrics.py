"""
metrics.py: Evaluation metrics for Langtune

Provides metrics for evaluating language models.
"""

import math
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity score
    """
    return math.exp(min(loss, 20))  # Clip to avoid overflow


def compute_bits_per_character(loss: float, log_base: float = 2) -> float:
    """
    Compute bits per character (BPC).
    
    Args:
        loss: Cross-entropy loss (in nats)
        log_base: Logarithm base (2 for bits)
        
    Returns:
        BPC score
    """
    return loss / math.log(log_base)


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute token prediction accuracy.
    
    Args:
        predictions: Predicted token IDs (batch, seq_len)
        labels: True token IDs (batch, seq_len)
        ignore_index: Index to ignore in labels
        
    Returns:
        Accuracy score (0-1)
    """
    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0
    
    correct = (predictions == labels) & mask
    return correct.sum().item() / mask.sum().item()


def compute_top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5, ignore_index: int = -100) -> float:
    """
    Compute top-k token prediction accuracy.
    
    Args:
        logits: Model logits (batch, seq_len, vocab_size)
        labels: True token IDs (batch, seq_len)
        k: Number of top predictions to consider
        ignore_index: Index to ignore in labels
        
    Returns:
        Top-k accuracy score (0-1)
    """
    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0
    
    top_k_preds = logits.topk(k, dim=-1).indices
    labels_expanded = labels.unsqueeze(-1).expand_as(top_k_preds)
    
    correct = (top_k_preds == labels_expanded).any(dim=-1) & mask
    return correct.sum().item() / mask.sum().item()


def compute_ngram_overlap(generated: List[str], references: List[str], n: int = 1) -> Dict[str, float]:
    """
    Compute n-gram overlap metrics (precision, recall, F1).
    
    Args:
        generated: List of generated texts
        references: List of reference texts
        n: N-gram size
        
    Returns:
        Dict with precision, recall, and F1 scores
    """
    def get_ngrams(text: str, n: int) -> Counter:
        tokens = text.lower().split()
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    total_precision = 0.0
    total_recall = 0.0
    count = 0
    
    for gen, ref in zip(generated, references):
        gen_ngrams = get_ngrams(gen, n)
        ref_ngrams = get_ngrams(ref, n)
        
        if not gen_ngrams or not ref_ngrams:
            continue
        
        overlap = sum((gen_ngrams & ref_ngrams).values())
        precision = overlap / sum(gen_ngrams.values()) if gen_ngrams else 0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0
        
        total_precision += precision
        total_recall += recall
        count += 1
    
    if count == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    avg_precision = total_precision / count
    avg_recall = total_recall / count
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": f1
    }


def compute_bleu(generated: List[str], references: List[str], max_n: int = 4) -> float:
    """
    Compute BLEU score (simplified implementation).
    
    Args:
        generated: List of generated texts
        references: List of reference texts
        max_n: Maximum n-gram size
        
    Returns:
        BLEU score (0-1)
    """
    def get_ngrams(text: str, n: int) -> Counter:
        tokens = text.lower().split()
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    def brevity_penalty(gen_len: int, ref_len: int) -> float:
        if gen_len >= ref_len:
            return 1.0
        return math.exp(1 - ref_len / max(gen_len, 1))
    
    precisions = []
    total_gen_len = 0
    total_ref_len = 0
    
    for n in range(1, max_n + 1):
        total_overlap = 0
        total_count = 0
        
        for gen, ref in zip(generated, references):
            gen_ngrams = get_ngrams(gen, n)
            ref_ngrams = get_ngrams(ref, n)
            
            overlap = sum((gen_ngrams & ref_ngrams).values())
            count = sum(gen_ngrams.values())
            
            total_overlap += overlap
            total_count += count
            
            if n == 1:
                total_gen_len += len(gen.split())
                total_ref_len += len(ref.split())
        
        precision = total_overlap / max(total_count, 1)
        precisions.append(max(precision, 1e-10))  # Avoid log(0)
    
    # Geometric mean of precisions
    log_precision = sum(math.log(p) for p in precisions) / max_n
    bleu = math.exp(log_precision)
    
    # Apply brevity penalty
    bp = brevity_penalty(total_gen_len, total_ref_len)
    
    return bp * bleu


def compute_rouge_l(generated: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE-L score based on longest common subsequence.
    
    Args:
        generated: Generated text
        reference: Reference text
        
    Returns:
        Dict with precision, recall, and F1
    """
    def lcs_length(a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()
    
    if not gen_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    lcs = lcs_length(gen_tokens, ref_tokens)
    
    precision = lcs / len(gen_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_diversity(texts: List[str], n: int = 2) -> float:
    """
    Compute n-gram diversity (ratio of unique n-grams to total n-grams).
    
    Args:
        texts: List of texts
        n: N-gram size
        
    Returns:
        Diversity score (0-1)
    """
    all_ngrams = []
    
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    
    if not all_ngrams:
        return 0.0
    
    unique = len(set(all_ngrams))
    total = len(all_ngrams)
    
    return unique / total


def compute_repetition_ratio(text: str, n: int = 3) -> float:
    """
    Compute ratio of repeated n-grams (lower is better).
    
    Args:
        text: Input text
        n: N-gram size
        
    Returns:
        Repetition ratio (0-1)
    """
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0
    
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    if not ngrams:
        return 0.0
    
    ngram_counts = Counter(ngrams)
    repeated = sum(1 for count in ngram_counts.values() if count > 1)
    
    return repeated / len(ngram_counts)


class MetricsCalculator:
    """Convenience class for computing multiple metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.total_correct = 0
        self.generated_texts = []
        self.reference_texts = []
    
    def update(
        self,
        loss: Optional[float] = None,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        generated: Optional[str] = None,
        reference: Optional[str] = None,
        ignore_index: int = -100
    ):
        """Update metrics with a batch."""
        if loss is not None and labels is not None:
            mask = labels != ignore_index
            num_tokens = mask.sum().item()
            self.total_loss += loss * num_tokens
            self.total_tokens += num_tokens
        
        if logits is not None and labels is not None:
            predictions = logits.argmax(dim=-1)
            mask = labels != ignore_index
            correct = ((predictions == labels) & mask).sum().item()
            self.total_correct += correct
        
        if generated is not None:
            self.generated_texts.append(generated)
        if reference is not None:
            self.reference_texts.append(reference)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        results = {}
        
        if self.total_tokens > 0:
            avg_loss = self.total_loss / self.total_tokens
            results["loss"] = avg_loss
            results["perplexity"] = compute_perplexity(avg_loss)
            results["accuracy"] = self.total_correct / self.total_tokens
        
        if self.generated_texts and self.reference_texts:
            results["bleu"] = compute_bleu(self.generated_texts, self.reference_texts)
            
            # Average ROUGE-L
            rouge_scores = [compute_rouge_l(g, r) for g, r in zip(self.generated_texts, self.reference_texts)]
            results["rouge_l_f1"] = sum(s["f1"] for s in rouge_scores) / len(rouge_scores)
            
            results["diversity"] = compute_diversity(self.generated_texts)
        
        return results
