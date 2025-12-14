# Langtune Documentation

Welcome to the Langtune documentation! This guide covers everything you need to fine-tune language models efficiently using LoRA.

## 📚 Contents

- [Getting Started](#getting-started)
- [Tutorials](tutorials/index.md)
- [API Reference](api/index.md)
- [Best Practices](best_practices.md)
- [Troubleshooting](troubleshooting.md)

---

## Getting Started

### Installation

```bash
pip install langtune
```

### Quick Example

```python
from langtune import LoRALanguageModel, Config, Trainer
from langtune import create_data_loader, load_text_file

# Load configuration
config = Config.from_preset("small")

# Create model with LoRA
model = LoRALanguageModel(
    vocab_size=config.model.vocab_size,
    embed_dim=config.model.embed_dim,
    num_layers=config.model.num_layers,
    num_heads=config.model.num_heads,
    lora_config=config.model.lora.__dict__
)

# Train
trainer = Trainer(model, config, train_dataloader)
trainer.train()
```

---

## Core Concepts

### LoRA (Low-Rank Adaptation)

LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices, reducing trainable parameters by 97%+.

### FastLoRALanguageModel

Our optimized model class with:
- RoPE (Rotary Position Embeddings)
- Flash Attention / Memory-efficient attention
- Gradient checkpointing
- 4-bit quantization support

### FastTrainer

Optimized trainer with:
- Gradient accumulation
- Mixed precision (fp16/bf16)
- Memory monitoring

---

## CLI Usage

```bash
# Train with preset
langtune train --preset small --train-file data.txt

# Fast training with optimizations
langtune train --preset small --train-file data.txt --fast

# Check system info
langtune version
```

---

## Next Steps

- [Tutorials](tutorials/index.md) - Step-by-step guides
- [API Reference](api/index.md) - Detailed API documentation
- [Best Practices](best_practices.md) - Tips for optimal results
