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