"""
Langtrain Score Module

Computes a unified quality metric: Quality × Stability / GPU-Hours
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math


@dataclass  
class LangtrainScore:
    """
    Langtrain's unified training quality metric.
    
    Score = (Quality × Stability) / Cost
    
    Where:
    - Quality = weighted combination of accuracy, coherence, instruction-following
    - Stability = 1 / variance of multiple runs
    - Cost = GPU-hours
    """
    
    # Quality components (0-100)
    accuracy_score: float = 0.0
    coherence_score: float = 0.0
    instruction_following_score: float = 0.0
    hallucination_score: float = 0.0  # Lower is better
    
    # Stability (from multiple runs)
    stability_factor: float = 1.0
    run_variance: float = 0.0
    
    # Cost
    gpu_hours: float = 1.0
    
    # Weights for quality components
    weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "accuracy": 0.3,
                "coherence": 0.25,
                "instruction_following": 0.3,
                "hallucination": 0.15,
            }
    
    @property
    def quality_score(self) -> float:
        """Compute weighted quality score (0-100)."""
        quality = (
            self.accuracy_score * self.weights["accuracy"] +
            self.coherence_score * self.weights["coherence"] +
            self.instruction_following_score * self.weights["instruction_following"] +
            (100 - self.hallucination_score) * self.weights["hallucination"]
        )
        return min(100, max(0, quality))
    
    @property
    def total_score(self) -> float:
        """Compute final Langtrain Score."""
        if self.gpu_hours <= 0:
            return 0.0
        return (self.quality_score * self.stability_factor) / self.gpu_hours
    
    @property
    def efficiency_rating(self) -> str:
        """Human-readable efficiency rating."""
        score = self.total_score
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        elif score >= 20:
            return "Poor"
        return "Very Poor"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality": {
                "accuracy": round(self.accuracy_score, 2),
                "coherence": round(self.coherence_score, 2),
                "instruction_following": round(self.instruction_following_score, 2),
                "hallucination": round(self.hallucination_score, 2),
                "total": round(self.quality_score, 2),
            },
            "stability": {
                "factor": round(self.stability_factor, 3),
                "variance": round(self.run_variance, 4),
            },
            "cost": {
                "gpu_hours": round(self.gpu_hours, 2),
            },
            "langtrain_score": round(self.total_score, 2),
            "efficiency_rating": self.efficiency_rating,
        }


class ScoreCalculator:
    """
    Calculates Langtrain Score from training results.
    """
    
    def __init__(
        self,
        evaluation_prompts: List[str] = None,
        reference_responses: List[str] = None,
    ):
        self.evaluation_prompts = evaluation_prompts or []
        self.reference_responses = reference_responses or []
    
    def calculate(
        self,
        model: Any,
        tokenizer: Any,
        gpu_hours: float,
        run_results: List[float] = None,  # Results from multiple runs
    ) -> LangtrainScore:
        """
        Calculate Langtrain Score for a trained model.
        
        Args:
            model: The trained model
            tokenizer: The tokenizer
            gpu_hours: Training time in GPU-hours
            run_results: Optional results from multiple training runs
            
        Returns:
            LangtrainScore with all metrics
        """
        score = LangtrainScore(gpu_hours=gpu_hours)
        
        # Evaluate quality components
        if self.evaluation_prompts:
            responses = self._generate_responses(model, tokenizer)
            
            score.accuracy_score = self._evaluate_accuracy(responses)
            score.coherence_score = self._evaluate_coherence(responses)
            score.instruction_following_score = self._evaluate_instruction_following(responses)
            score.hallucination_score = self._evaluate_hallucination(responses)
        else:
            # Default scores if no evaluation prompts
            score.accuracy_score = 70.0
            score.coherence_score = 70.0
            score.instruction_following_score = 70.0
            score.hallucination_score = 20.0
        
        # Calculate stability from multiple runs
        if run_results and len(run_results) > 1:
            mean = sum(run_results) / len(run_results)
            variance = sum((x - mean) ** 2 for x in run_results) / len(run_results)
            score.run_variance = variance
            score.stability_factor = 1 / (1 + math.sqrt(variance))
        else:
            score.stability_factor = 1.0
        
        return score
    
    def _generate_responses(self, model: Any, tokenizer: Any) -> List[str]:
        """Generate responses for evaluation prompts."""
        responses = []
        
        for prompt in self.evaluation_prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(response)
            except Exception:
                responses.append("")
        
        return responses
    
    def _evaluate_accuracy(self, responses: List[str]) -> float:
        """Evaluate response accuracy (placeholder)."""
        # In production, this would use semantic similarity to references
        if not responses:
            return 50.0
        
        # Simple heuristic: longer, non-empty responses score higher
        scores = []
        for response in responses:
            if len(response) > 100:
                scores.append(80.0)
            elif len(response) > 50:
                scores.append(60.0)
            elif len(response) > 10:
                scores.append(40.0)
            else:
                scores.append(20.0)
        
        return sum(scores) / len(scores) if scores else 50.0
    
    def _evaluate_coherence(self, responses: List[str]) -> float:
        """Evaluate response coherence (placeholder)."""
        # Would use perplexity or coherence models in production
        return 70.0
    
    def _evaluate_instruction_following(self, responses: List[str]) -> float:
        """Evaluate instruction following (placeholder)."""
        # Would check if responses follow the prompt instructions
        return 70.0
    
    def _evaluate_hallucination(self, responses: List[str]) -> float:
        """Evaluate hallucination rate (placeholder)."""
        # Would use fact-checking or grounding in production
        # Lower is better
        return 20.0


def compute_score(
    accuracy: float,
    coherence: float,
    instruction_following: float,
    hallucination: float,
    gpu_hours: float,
    stability: float = 1.0,
) -> LangtrainScore:
    """
    Quick helper to compute Langtrain Score from individual metrics.
    """
    return LangtrainScore(
        accuracy_score=accuracy,
        coherence_score=coherence,
        instruction_following_score=instruction_following,
        hallucination_score=hallucination,
        gpu_hours=gpu_hours,
        stability_factor=stability,
    )


def compare_runs(
    runs: List[LangtrainScore],
) -> Dict[str, Any]:
    """
    Compare multiple training runs.
    """
    if not runs:
        return {"error": "No runs to compare"}
    
    scores = [r.total_score for r in runs]
    qualities = [r.quality_score for r in runs]
    
    return {
        "num_runs": len(runs),
        "best_score": max(scores),
        "worst_score": min(scores),
        "avg_score": sum(scores) / len(scores),
        "score_variance": sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores),
        "best_quality": max(qualities),
        "avg_quality": sum(qualities) / len(qualities),
        "total_gpu_hours": sum(r.gpu_hours for r in runs),
    }
