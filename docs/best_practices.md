# Best Practices

Tips and recommendations for optimal fine-tuning with Langtune.

---

## 🎯 General Guidelines

### 1. Start Small
- Begin with `tiny` or `small` presets
- Verify your pipeline works before scaling

### 2. Use Appropriate LoRA Rank
| Task | Recommended Rank |
|------|------------------|
| Simple adaptation | 4-8 |
| General fine-tuning | 16-32 |
| Complex tasks | 64-128 |

### 3. Learning Rate
- Start with `1e-4` for LoRA
- Use warmup (5-10% of training)

---

## 💾 Memory Optimization

### Enable Gradient Checkpointing
```python
model = FastLoRALanguageModel(
    ...,
    use_gradient_checkpointing=True
)
```

### Use Mixed Precision
```bash
langtune train --mixed-precision fp16
```

### Gradient Accumulation
```bash
langtune train --gradient-accumulation 4
```

---

## ⚡ Speed Optimization

### Use FastTrainer
```python
from langtune import create_fast_trainer

trainer = create_fast_trainer(
    config, dataloader,
    gradient_accumulation_steps=4,
    mixed_precision="fp16"
)
```

### Enable Flash Attention
Flash attention is enabled by default in `FastLoRALanguageModel`.

---

## 📊 Data Best Practices

1. **Quality over quantity** - Clean data beats more data
2. **Diverse examples** - Cover edge cases
3. **Consistent formatting** - Use same format as inference
4. **Shuffle data** - Prevent ordering bias

---

## 🔍 Monitoring

### Track with W&B
```bash
wandb login
langtune train --preset small --train-file data.txt
```

### Log Memory Usage
```python
from langtune import log_gpu_memory
log_gpu_memory()
```

---

## ❌ Common Mistakes

| Mistake | Solution |
|---------|----------|
| Too high learning rate | Start with 1e-4 |
| No validation set | Use 10-20% for validation |
| Overfitting | Use early stopping |
| OOM errors | Reduce batch size, use grad checkpointing |
