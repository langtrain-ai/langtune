# API Reference

Complete API documentation for Langtune.

---

## Models

### LoRALanguageModel
```python
from langtune import LoRALanguageModel

model = LoRALanguageModel(
    vocab_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    max_seq_len: int = 512,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    lora_config: Optional[Dict] = None
)
```

### FastLoRALanguageModel
```python
from langtune import FastLoRALanguageModel

model = FastLoRALanguageModel(
    vocab_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    max_seq_len: int = 2048,
    use_rope: bool = True,
    use_flash_attention: bool = True,
    use_gradient_checkpointing: bool = True,
    lora_config: Optional[Dict] = None
)
```

---

## Training

### Trainer
```python
from langtune import Trainer

trainer = Trainer(
    model: LoRALanguageModel,
    config: Config,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None
)

trainer.train(resume_from_checkpoint: Optional[str] = None)
```

### FastTrainer
```python
from langtune import FastTrainer, create_fast_trainer

trainer = create_fast_trainer(
    config: Config,
    train_dataloader: DataLoader,
    gradient_accumulation_steps: int = 4,
    mixed_precision: str = "fp16"
)
```

---

## Configuration

### Config
```python
from langtune import Config, get_preset_config

# Load preset
config = get_preset_config("small")  # tiny, small, base, large

# Or create custom
config = Config(
    model=ModelConfig(...),
    training=TrainingConfig(...),
    data=DataConfig(...)
)
```

---

## Optimizations

### OptimizationConfig
```python
from langtune import OptimizationConfig

opt_config = OptimizationConfig(
    use_4bit: bool = False,
    use_flash_attention: bool = True,
    use_gradient_checkpointing: bool = True,
    mixed_precision: str = "fp16",
    gradient_accumulation_steps: int = 4
)
```

### Memory Utilities
```python
from langtune import get_memory_stats, cleanup_memory

stats = get_memory_stats()  # Returns GPU memory info
cleanup_memory()            # Frees unused GPU memory
```

---

## Data

### TextDataset
```python
from langtune import TextDataset

dataset = TextDataset(
    data: List[str],
    max_length: int = 512
)
```

### DataCollator
```python
from langtune import DataCollator

collator = DataCollator(
    pad_token_id: int = 0,
    max_length: int = 512
)
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `langtune train` | Train a model |
| `langtune evaluate` | Evaluate a model |
| `langtune generate` | Generate text |
| `langtune version` | Show version info |
| `langtune info` | Quick start guide |
| `langtune auth login` | Login with API key |
