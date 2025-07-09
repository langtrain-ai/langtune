"""
utils.py: Utility functions for Langtune
"""

def encode_text(text, tokenizer=None):
    """
    Encodes text into token IDs using the provided tokenizer.
    If no tokenizer is given, splits on whitespace as a placeholder.
    """
    if tokenizer:
        return tokenizer.encode(text)
    return [ord(c) for c in text]

def decode_tokens(token_ids):
    """
    Decodes a list of token IDs back into a string (placeholder implementation).
    """
    return ''.join([chr(i) for i in token_ids])

class SimpleTokenizer:
    """
    A simple whitespace tokenizer stub for demonstration purposes.
    """
    def encode(self, text):
        return [ord(c) for c in text]
    def decode(self, token_ids):
        return ''.join([chr(i) for i in token_ids]) 