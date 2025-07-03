# Langtune: Efficient LoRA Fine-Tuning for Text LLMs

<hr/>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/langtrain-ai/langtrain/main/static/langtune-use-dark.png">
    <img alt="Langtune Logo" src="https://raw.githubusercontent.com/langtrain-ai/langtrain/main/static/langtune-white.png" width="full" />
  </picture>
</p>

<!-- Badges -->
<p align="center">
  <a href="https://pypi.org/project/langtune/"><img src="https://img.shields.io/pypi/v/langtune.svg" alt="PyPI version"></a>
  <a href="https://pepy.tech/project/langtune"><img src="https://pepy.tech/badge/langtune" alt="Downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  <a href="https://img.shields.io/badge/coverage-90%25-brightgreen" alt="Coverage"> <img src="https://img.shields.io/badge/coverage-90%25-brightgreen"/></a>
  <a href="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python Version"> <img src="https://img.shields.io/badge/python-3.8%2B-blue"/></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

<p align="center">
  <b>Langtune is a Python package for fine-tuning large language models on text data using LoRA.</b><br/>
  <span style="font-size:1.1em"><i>Provides modular components for adapting language models to various NLP tasks.</i></span>
</p>
<hr/>

## Quick Links
- [Documentation](docs/index.md)
- [Tutorials](docs/tutorials/index.md)
- [Changelog](CHANGELOG.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Roadmap](ROADMAP.md)

---

## Table of Contents
- [Features](#features)
- [Showcase](#showcase)
- [Getting Started](#getting-started)
- [Supported Python Versions](#supported-python-versions)
- [Why langtune?](#why-langtune)
- [Architecture Overview](#architecture-overview)
- [Core Modules](#core-modules)
- [Performance & Efficiency](#performance--efficiency)
- [Advanced Configuration](#advanced-configuration)
- [Documentation & Resources](#documentation--resources)
- [Testing & Quality](#testing--quality)
- [Examples & Use Cases](#examples--use-cases)
- [Extending the Framework](#extending-the-framework)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Features
- LoRA adapters for efficient fine-tuning
- Modular transformer backbone
- Model zoo for language models
- Configurable and extensible codebase
- Checkpointing and resume
- Mixed precision and distributed training
- Metrics and visualization tools
- CLI for training and evaluation
- Callback support (early stopping, logging, etc.)

---

## Showcase

Langtune is intended for building and fine-tuning large language models with LoRA. It can be used for text classification, summarization, question answering, and other NLP tasks.

---

## Getting Started

Install:

```bash
pip install langtune
```

Example usage:

```python
import torch
from langtune.models.llm import LanguageModel
from langtune.utils.config import default_config

input_ids = torch.randint(0, 1000, (2, 128))
model = LanguageModel(
    vocab_size=default_config['vocab_size'],
    embed_dim=default_config['embed_dim'],
    num_layers=default_config['num_layers'],
    num_heads=default_config['num_heads'],
    mlp_ratio=default_config['mlp_ratio'],
    lora_config=default_config['lora'],
)

with torch.no_grad():
    out = model(input_ids)
    print('Output shape:', out.shape)
```

See the [Documentation](docs/index.md) and `src/langtune/cli/finetune.py` for more details.

---

## Supported Python Versions
- Python 3.8 or newer

---

## Why langtune?

- Fine-tuning with LoRA adapters
- Modular transformer design
- Unified interface for language models
- Suitable for research and production
- Efficient memory usage

---

## Architecture Overview

Langtune uses a transformer backbone with LoRA adapters in attention and MLP layers. This enables adaptation of pre-trained models with fewer trainable parameters.

### Model Data Flow

```mermaid
---
config:
  layout: dagre
---
flowchart TD
 subgraph LoRA_Adapters["LoRA Adapters in Attention and MLP"]
        LA1(["LoRA Adapter 1"])
        LA2(["LoRA Adapter 2"])
        LA3(["LoRA Adapter N"])
  end
    A(["Input Tokens"]) --> B(["Embedding Layer"])
    B --> C(["Positional Encoding"])
    C --> D1(["Encoder Layer 1"])
    D1 --> D2(["Encoder Layer 2"])
    D2 --> D3(["Encoder Layer N"])
    D3 --> E(["LayerNorm"])
    E --> F(["MLP Head"])
    F --> G(["Output Logits"])
    LA1 -.-> D1
    LA2 -.-> D2
    LA3 -.-> D3
     LA1:::loraStyle
     LA2:::loraStyle
     LA3:::loraStyle
    classDef loraStyle fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
```

---

## Core Modules

| Module | Description | Key Features |
|--------|-------------|--------------|
| Embedding | Token embedding and positional encoding | Configurable vocab size, position embeddings |
| TransformerEncoder | Multi-layer transformer backbone | Self-attention, LoRA integration, checkpointing |
| LoRALinear | Low-rank adaptation layers | Configurable rank, memory-efficient updates |
| MLPHead | Output projection layer | Classification, regression, dropout |
| Config System | Centralized configuration | YAML/JSON config, CLI overrides |
| Data Utils | Preprocessing and augmentation | Built-in tokenization, custom loaders |

---

## Performance & Efficiency

| Metric | Full Fine-tuning | LoRA Fine-tuning | Improvement |
|--------|------------------|------------------|-------------|
| Trainable Parameters | 125M | 3.2M | 97% reduction |
| Memory Usage | 16GB | 5GB | 69% reduction |
| Training Time | 6h | 2h | 67% faster |
| Storage per Task | 500MB | 12MB | 98% smaller |

*Benchmarks: Transformer-Base, WikiText-103, RTX 3090*

Supported model sizes: Transformer-Tiny, Transformer-Small, Transformer-Base, Transformer-Large

---

## Advanced Configuration

Example LoRA config:

```python
lora_config = {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.1,
    "target_modules": ["attention.qkv", "attention.proj", "mlp.fc1", "mlp.fc2"],
    "merge_weights": False
}
```

Example training config:

```yaml
model:
  name: "transformer_base"
  vocab_size: 50257
  embed_dim: 768
  num_layers: 12
  num_heads: 12
training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_steps: 1000
lora:
  rank: 16
  alpha: 32
  dropout: 0.1
```

---

## Documentation & Resources
- [API Reference](docs/api/index.md)
- [Tutorials and Examples](docs/tutorials/index.md)
- [Research Papers](#research-papers)
- [Best Practices Guide](docs/best_practices.md)
- [Troubleshooting](docs/troubleshooting.md)

### Research Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

---

## Testing & Quality

Run tests:

```bash
pytest tests/
```

Code quality tools:

```bash
flake8 src/
black src/ --check
mypy src/
bandit -r src/
```

---

## Examples & Use Cases

Text classification:

```python
from langtune import LanguageModel
from langtune.datasets import TextClassificationDataset

model = LanguageModel.from_pretrained("transformer_base")
dataset = TextClassificationDataset(train=True, tokenizer=model.tokenizer)
model.finetune(dataset, epochs=10, lora_rank=16)
```

Custom dataset:

```python
from langtune.datasets import CustomTextDataset

dataset = CustomTextDataset(
    file_path="/path/to/dataset.txt",
    split="train",
    tokenizer=model.tokenizer
)
model.finetune(dataset, config_path="configs/custom_config.yaml")
```

---

## Extending the Framework
- Add datasets in `src/langtune/data/datasets.py`
- Add callbacks in `src/langtune/callbacks/`
- Add models in `src/langtune/models/`
- Add CLI tools in `src/langtune/cli/`

## Documentation
- See code comments and docstrings for details.
- For advanced usage, see `src/langtune/cli/finetune.py`.

## Contributing
Contributions are welcome. See the [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use langtune in your research, please cite:

```bibtex
@software{langtune2025,
  author = {Pritesh Raj},
  title = {langtune: LLMs with Efficient LoRA Fine-Tuning},
  url = {https://github.com/langtrain-ai/langtune},
  year = {2025},
  version = {0.1.0}
}
```

## Acknowledgements

We thank the following projects and communities:
- [PyTorch](https://pytorch.org/)
- [HuggingFace](https://huggingface.co/)
- [PEFT](https://github.com/huggingface/peft)

<p align="center">
  <b>Made in India 🇮🇳 with ❤️ by the langtune team</b><br/>
  <i>Star ⭐ this repo if you find it useful!</i>
</p>
