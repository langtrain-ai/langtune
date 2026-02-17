"""
packing.py: Sequence packing utility for efficient training.
"""

import torch
from typing import List, Tuple
import math

class SequencePacker:
    """
    Pack multiple short sequences into single training examples.
    
    This maximizes GPU utilization by reducing padding waste.
    """
    
    def __init__(
        self,
        max_seq_length: int,
        pad_token_id: int,
        efficiency_target: float = 0.9,
    ):
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.efficiency_target = efficiency_target
    
    def pack_sequences(
        self,
        sequences: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Pack sequences into fixed-length batches.
        
        Returns:
            Tuple of (packed_input_ids, attention_masks)
        """
        packed_inputs = []
        packed_masks = []
        
        current_input = []
        current_length = 0
        
        # Sort by length to optimize packing (simple approach)
        # More advanced bin packing algorithms could be used here
        for seq in sorted(sequences, key=len, reverse=True):
            seq_len = len(seq)
            
            if current_length + seq_len <= self.max_seq_length:
                # Add to current pack
                current_input.extend(seq.tolist())
                current_length += seq_len
            else:
                # Start new pack
                if current_input:
                    packed, mask = self._finalize_pack(current_input)
                    packed_inputs.append(packed)
                    packed_masks.append(mask)
                
                current_input = seq.tolist()
                current_length = seq_len
        
        # Finalize last pack
        if current_input:
            packed, mask = self._finalize_pack(current_input)
            packed_inputs.append(packed)
            packed_masks.append(mask)
        
        return packed_inputs, packed_masks
    
    def _finalize_pack(
        self,
        input_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad pack to max_seq_length and create attention mask."""
        current_len = len(input_ids)
        padding_len = self.max_seq_length - current_len
        
        # Pad input
        padded_input = input_ids + [self.pad_token_id] * padding_len
        
        # Create attention mask (1 for real tokens, 0 for padding)
        # Note: For packed sequences, we might need 2D attention masks if we want to prevent cross-contamination
        # but for causal LLMs with proper position ids or block diagonal masking, this is specific.
        # This implementation returns simple 1/0 masks.
        attention_mask = [1] * current_len + [0] * padding_len
        
        return torch.tensor(padded_input), torch.tensor(attention_mask)
    
    def calculate_efficiency(
        self,
        sequences: List[torch.Tensor],
    ) -> float:
        """Calculate packing efficiency."""
        total_tokens = sum(len(seq) for seq in sequences)
        num_packs = math.ceil(total_tokens / self.max_seq_length)
        total_capacity = num_packs * self.max_seq_length
        
        if total_capacity == 0:
            return 0.0
            
        return total_tokens / total_capacity
