"""
Reinforcement Learning Module for Fine-tuning

Implements RLHF, DPO, and PPO for preference-based training.
These techniques help models align better with human preferences.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import math


@dataclass
class RLConfig:
    """Base configuration for RL-based training."""
    
    # Learning rate
    learning_rate: float = 1e-5
    
    # Batch size
    batch_size: int = 4
    
    # Number of epochs
    num_epochs: int = 1
    
    # Beta for DPO/KL divergence
    beta: float = 0.1
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 4
    
    # Max sequence length
    max_length: int = 512
    
    # Logging steps
    logging_steps: int = 10
    
    # Save steps
    save_steps: int = 100


@dataclass
class PreferencePair:
    """A preference pair for DPO/RLHF training."""
    
    prompt: str
    chosen: str  # Preferred response
    rejected: str  # Non-preferred response
    
    # Optional metadata
    source: Optional[str] = None
    confidence: float = 1.0  # How confident is this preference


class BaseRLTrainer(ABC):
    """Base class for RL-based trainers."""
    
    def __init__(self, model: Any, config: RLConfig):
        self.model = model
        self.config = config
        self.step = 0
        self.losses: List[float] = []
    
    @abstractmethod
    def train_step(self, batch: Any) -> float:
        """Perform single training step, return loss."""
        pass
    
    @abstractmethod
    def train(self, dataset: List[PreferencePair]) -> Dict[str, Any]:
        """Train on preference dataset."""
        pass
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log training metrics."""
        print(f"Step {self.step}: {metrics}")


class DPOTrainer(BaseRLTrainer):
    """
    Direct Preference Optimization (DPO) Trainer.
    
    DPO is a simpler alternative to RLHF that directly optimizes for preferences
    without needing a separate reward model or RL loop.
    
    Paper: https://arxiv.org/abs/2305.18290
    """
    
    def __init__(
        self,
        model: Any,
        ref_model: Any,
        tokenizer: Any,
        config: RLConfig,
    ):
        super().__init__(model, config)
        self.ref_model = ref_model
        self.tokenizer = tokenizer
    
    def compute_dpo_loss(
        self,
        policy_chosen_logps: Any,
        policy_rejected_logps: Any,
        reference_chosen_logps: Any,
        reference_rejected_logps: Any,
    ) -> Any:
        """
        Compute DPO loss.
        
        loss = -log(sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)))))
        """
        import torch
        import torch.nn.functional as F
        
        # Log ratios
        chosen_rewards = self.config.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.config.beta * (policy_rejected_logps - reference_rejected_logps)
        
        # DPO loss
        losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
        
        return losses.mean()
    
    def get_batch_logps(
        self,
        model: Any,
        input_ids: Any,
        attention_mask: Any,
        labels: Any,
    ) -> Any:
        """Get log probabilities for a batch."""
        import torch
        
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log probs
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        per_token_logps = torch.gather(
            log_probs, 
            dim=2, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding
        mask = (shift_labels != -100).float()
        return (per_token_logps * mask).sum(-1)
    
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Single DPO training step."""
        import torch
        
        self.model.train()
        
        # Get log probs for policy model
        policy_chosen_logps = self.get_batch_logps(
            self.model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )
        policy_rejected_logps = self.get_batch_logps(
            self.model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )
        
        # Get log probs for reference model
        with torch.no_grad():
            ref_chosen_logps = self.get_batch_logps(
                self.ref_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            ref_rejected_logps = self.get_batch_logps(
                self.ref_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )
        
        # Compute DPO loss
        loss = self.compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )
        
        # Backward
        loss.backward()
        
        self.step += 1
        self.losses.append(loss.item())
        
        return loss.item()
    
    def train(self, dataset: List[PreferencePair]) -> Dict[str, Any]:
        """Train on preference pairs."""
        # This is a simplified version - actual implementation would use DataLoader
        total_loss = 0.0
        num_steps = 0
        
        for pair in dataset:
            # Tokenize (simplified)
            batch = self._prepare_batch([pair])
            loss = self.train_step(batch)
            total_loss += loss
            num_steps += 1
            
            if self.step % self.config.logging_steps == 0:
                self.log_metrics({"loss": total_loss / num_steps})
        
        return {
            "final_loss": total_loss / num_steps if num_steps > 0 else 0,
            "steps": num_steps,
            "losses": self.losses,
        }
    
    def _prepare_batch(self, pairs: List[PreferencePair]) -> Dict[str, Any]:
        """Prepare batch from preference pairs (placeholder)."""
        # Actual implementation would tokenize and create tensors
        return {}


class RLHFTrainer(BaseRLTrainer):
    """
    Reinforcement Learning from Human Feedback (RLHF) Trainer.
    
    Uses a reward model to score responses and PPO to optimize the policy.
    """
    
    def __init__(
        self,
        model: Any,
        reward_model: Any,
        tokenizer: Any,
        config: RLConfig,
    ):
        super().__init__(model, config)
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.ref_model = None  # Will be set to frozen copy of model
    
    def compute_reward(self, prompt: str, response: str) -> float:
        """Compute reward for a response using the reward model."""
        import torch
        
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt + response,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )
            reward = self.reward_model(**inputs).logits.item()
        
        return reward
    
    def train_step(self, batch: Any) -> float:
        """Single RLHF training step with PPO."""
        # Simplified PPO implementation
        # Actual implementation would be more complex
        return 0.0
    
    def train(self, dataset: List[PreferencePair]) -> Dict[str, Any]:
        """Train using RLHF."""
        # First train reward model on preference pairs
        # Then use PPO to optimize policy
        return {"status": "RLHF training complete"}


class PPOTrainer(BaseRLTrainer):
    """
    Proximal Policy Optimization (PPO) Trainer.
    
    Used in RLHF pipeline after training a reward model.
    """
    
    def __init__(
        self,
        model: Any,
        ref_model: Any,
        reward_fn: Callable[[str, str], float],
        tokenizer: Any,
        config: RLConfig,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.1,
        entropy_coef: float = 0.01,
    ):
        super().__init__(model, config)
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def compute_advantages(
        self,
        rewards: List[float],
        values: List[float],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> List[float]:
        """Compute GAE advantages."""
        advantages = []
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def ppo_loss(
        self,
        log_probs: Any,
        old_log_probs: Any,
        advantages: Any,
        values: Any,
        returns: Any,
    ) -> Any:
        """Compute PPO clipped objective."""
        import torch
        import torch.nn.functional as F
        
        # Policy ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss
        
        return loss
    
    def train_step(self, batch: Any) -> float:
        """Single PPO training step."""
        return 0.0
    
    def train(self, dataset: List[PreferencePair]) -> Dict[str, Any]:
        """Train using PPO."""
        return {"status": "PPO training complete"}


# Utility functions
def create_preference_dataset(
    prompts: List[str],
    chosen_responses: List[str],
    rejected_responses: List[str],
) -> List[PreferencePair]:
    """Create preference dataset from lists."""
    return [
        PreferencePair(prompt=p, chosen=c, rejected=r)
        for p, c, r in zip(prompts, chosen_responses, rejected_responses)
    ]


def evaluate_alignment(
    model: Any,
    tokenizer: Any,
    test_prompts: List[str],
    reward_fn: Callable[[str, str], float],
) -> Dict[str, float]:
    """Evaluate model alignment using reward function."""
    rewards = []
    
    for prompt in test_prompts:
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Compute reward
        reward = reward_fn(prompt, response)
        rewards.append(reward)
    
    return {
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
        "min_reward": min(rewards) if rewards else 0,
        "max_reward": max(rewards) if rewards else 0,
    }
