"""
NEFTune: Noisy Embeddings for Instruction Fine-tuning.
Adds uniform noise to the embedding inputs during training to improve generalization.
"""

import math
import torch
from torch import nn
from typing import Optional, List, Dict, Any, Union
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class NEFTuneCallback(TrainerCallback):
    """
    Callback for adding uniform noise to embeddings during training (NEFTune).
    Reference: https://arxiv.org/abs/2310.05914
    """

    def __init__(self, noise_alpha: float = 5.0):
        self.noise_alpha = noise_alpha
        self.hooks = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """
        Attach forward hooks to the embedding layer(s) of the model.
        """
        print(f"🔊 Enabling NEFTune with alpha={self.noise_alpha}")
        
        # Identify embedding layers
        # Common HF models: model.embed_tokens, transformer.wte, etc.
        forward_hook = self._get_neftune_hook()
        
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if embeddings:
                self.hooks.append(embeddings.register_forward_hook(forward_hook))
                return

        # Fallback recursive search if get_input_embeddings is not standard
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                self.hooks.append(module.register_forward_hook(forward_hook))
                
    def on_train_end(self, args, state, control, **kwargs):
        """Remove hooks after training."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("🔊 Disabled NEFTune hooks")

    def _get_neftune_hook(self):
        """
        Returns a forward hook that adds noise.
        """
        def hook(module, args, output):
            # args[0] is usually input_ids, output is embeddings
            if module.training:
                # noise ~ Uniform(-1, 1) * alpha / sqrt(sequence_length * hidden_dim)
                dims = torch.tensor(output.size(1) * output.size(2))
                mag_norm = self.noise_alpha / torch.sqrt(dims)
                noise = torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
                return output + noise
            return output
            
        return hook

def activate_neftune(model: nn.Module, noise_alpha: float = 5.0):
    """
    Directly activate NEFTune on a model without using a callback (manual loop).
    """
    embeddings = model.get_input_embeddings()
    
    def neftune_forward(module, input, output):
        if module.training:
            dims = torch.tensor(output.size(1) * output.size(2))
            mag_norm = noise_alpha / torch.sqrt(dims)
            output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
        return output

    embeddings.register_forward_hook(neftune_forward)
    print(f"🔊 NEFTune activated manually (alpha={noise_alpha})")
