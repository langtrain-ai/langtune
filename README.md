<div align="center">

<img src="https://raw.githubusercontent.com/langtrain-ai/langtune/main/static/langtune-white.png" alt="Langtune" width="400" />

<h3>The fastest way to fine-tune LLMs</h3>

<p>
  <strong>Production-ready LoRA fine-tuning in minutes, not days.</strong><br>
  Built for ML engineers who need results, not complexity.
</p>

<p>
  <a href="https://www.producthunt.com/products/langtrain-2" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1049974&theme=light" alt="Product Hunt" width="200" /></a>
</p>

<p>
  <a href="https://pypi.org/project/langtune/"><img src="https://img.shields.io/pypi/v/langtune.svg?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI" /></a>
  <a href="https://pepy.tech/project/langtune"><img src="https://img.shields.io/pepy/dt/langtune?style=for-the-badge&logo=python&logoColor=white&label=downloads" alt="Downloads" /></a>
  <a href="https://github.com/langtrain-ai/langtune/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" alt="License" /></a>
</p>

<p>
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#why-langtune">Why Langtune</a> •
  <a href="https://langtrain.xyz/docs">Docs</a>
</p>

</div>

---

## ⚡ Quick Start

### 1-Click Install (Recommended)
The fastest way to get started. Installs Langtune in an isolated environment.

```bash
curl -fsSL https://raw.githubusercontent.com/langtrain-ai/langtune/main/scripts/install.sh | bash
```

### Or using pip
```bash
pip install langtune
```

Fine-tune your first model in **3 lines of code**:

```python
from langtune import LoRATrainer

trainer = LoRATrainer(model_name="meta-llama/Llama-2-7b-hf")
trainer.train_from_file("data.jsonl")
```

That's it. Your fine-tuned model is ready.

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🚀 **Blazing Fast**
Train 7B models in under 30 minutes on a single GPU. Our optimized kernels squeeze every last FLOP.

### 🎯 **Zero Config Required**
Smart defaults that just work. No PhD required. Start training in seconds.

### 💾 **Memory Efficient**
4-bit quantization + gradient checkpointing = Train 70B models on consumer hardware.

</td>
<td width="50%">

### 🔧 **Production Ready**
Battle-tested at scale. Used by teams fine-tuning thousands of models daily.

### 🌐 **Any Model, Any Data**
Works with Llama, Mistral, Qwen, Phi, and more. JSONL, CSV, or HuggingFace datasets.

### ☁️ **Cloud Native**
One-click deployment to Langtrain Cloud. Or export to GGUF, ONNX, HuggingFace.

</td>
</tr>
</table>

---

## 🎯 Why Langtune?

| | Langtune | Others |
|---|:---:|:---:|
| **Time to first training** | 30 seconds | 2+ hours |
| **Lines of code** | 3 | 100+ |
| **Memory usage** | 8GB | 24GB+ |
| **Learning curve** | Minutes | Days |

---

## 📖 Full Example

```python
from langtune import LoRATrainer
from langtune.config import TrainingConfig, LoRAConfig

# Configure your training
config = TrainingConfig(
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    lora=LoRAConfig(rank=16, alpha=32)
)

# Initialize and train
trainer = LoRATrainer(
    model_name="mistralai/Mistral-7B-v0.1",
    output_dir="./my-model",
    config=config
)

# Train on your data
trainer.train_from_file("training_data.jsonl")

# Push to Hub (optional)
trainer.push_to_hub("my-username/my-fine-tuned-model")
```

---

## 🛠️ Advanced Usage

<details>
<summary><b>Custom Dataset Format</b></summary>

```python
# JSONL format (recommended)
{"text": "Your training example here"}
{"text": "Another example"}

# Or instruction format
{"instruction": "Summarize this:", "input": "Long text...", "output": "Summary"}
```

</details>

<details>
<summary><b>Distributed Training</b></summary>

```python
trainer = LoRATrainer(
    model_name="meta-llama/Llama-2-70b-hf",
    device_map="auto",  # Automatic multi-GPU
)
```

</details>

<details>
<summary><b>Export Formats</b></summary>

```python
# Export to different formats
trainer.export("gguf")  # For llama.cpp
trainer.export("onnx")  # For ONNX Runtime
trainer.export("hf")    # HuggingFace format
```

</details>

---

## 🤝 Community

<p align="center">
  <a href="https://discord.gg/langtrain">Discord</a> •
  <a href="https://twitter.com/langtrainai">Twitter</a> •
  <a href="https://langtrain.xyz">Website</a>
</p>

---

<div align="center">

**Built with ❤️ by [Langtrain AI](https://langtrain.xyz)**

*Making LLM fine-tuning accessible to everyone.*

</div>
