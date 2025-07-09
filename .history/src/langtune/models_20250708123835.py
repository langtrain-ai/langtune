"""
models.py: LoRA-enabled transformer models for Langtune
"""

class LoRALanguageModel:
    """
    A stub for a language model with LoRA support.
    """
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, lora_config=None):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.lora_config = lora_config or {}

    def forward(self, input_ids):
        """
        Forward pass stub.
        """
        pass 