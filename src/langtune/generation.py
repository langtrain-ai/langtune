"""
generation.py: Text generation utilities for Langtune
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class TextGenerator:
    """High-level text generation with sampling strategies."""
    
    def __init__(self, model: nn.Module, tokenizer=None, device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: int = 1
    ) -> Union[str, torch.Tensor]:
        """Generate text with various sampling strategies."""
        # Encode prompt
        if isinstance(prompt, str):
            if self.tokenizer:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            else:
                input_ids = torch.tensor([[ord(c) for c in prompt]], device=self.device)
        else:
            input_ids = prompt.to(self.device)
        
        for _ in range(max_length - input_ids.size(1)):
            outputs = self.model(input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            next_logits = logits[:, -1, :]
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for token in input_ids[0].unique():
                    next_logits[0, token] /= repetition_penalty
            
            # Temperature
            if do_sample and temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Top-k filtering
            if top_k:
                top_k = min(top_k, next_logits.size(-1))
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
            if top_p and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = 0
                indices_to_remove = remove.scatter(1, sorted_indices, remove)
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample:
                next_token = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == eos_token_id:
                break
        
        if self.tokenizer:
            return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return input_ids


def generate(model, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.95, **kwargs):
    """Convenience function for text generation."""
    return TextGenerator(model).generate(
        prompt, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p, **kwargs
    )
