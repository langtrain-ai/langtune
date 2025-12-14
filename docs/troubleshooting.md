# Troubleshooting

Common issues and solutions when using Langtune.

---

## 🔴 Installation Issues

### "No module named 'langtune'"
```bash
pip install langtune
# Or install from source:
pip install -e .
```

### CUDA not detected
```python
import torch
print(torch.cuda.is_available())  # Should be True
```
- Ensure you have CUDA-compatible PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

---

## 💾 Memory Errors

### "CUDA out of memory"

**Solutions:**
1. Reduce batch size
2. Enable gradient checkpointing:
   ```bash
   langtune train --gradient-checkpointing
   ```
3. Use gradient accumulation:
   ```bash
   langtune train --gradient-accumulation 4 --batch-size 2
   ```
4. Use mixed precision:
   ```bash
   langtune train --mixed-precision fp16
   ```

### Memory keeps growing
```python
from langtune import cleanup_memory
cleanup_memory()  # Call periodically
```

---

## 🐢 Training Issues

### Training is slow

1. **Use FastTrainer:**
   ```bash
   langtune train --fast
   ```
2. **Check GPU utilization:**
   ```bash
   nvidia-smi
   ```
3. **Increase batch size** if memory allows

### Loss not decreasing
- Learning rate too high/low
- Data quality issues
- Try different LoRA rank

### NaN loss
- Learning rate too high
- Gradient explosion → reduce lr or use grad clipping

---

## 🔧 CLI Issues

### "langtune: command not found"
```bash
pip install langtune
# Or ensure PATH includes pip packages
python -m langtune --help
```

### Authentication error
```bash
langtune auth login
```

---

## 📊 Specific Errors

| Error | Solution |
|-------|----------|
| `ImportError: No module named 'torch'` | `pip install torch` |
| `RuntimeError: CUDA error` | Check CUDA version compatibility |
| `ValueError: Invalid config` | Check config file syntax |
| `FileNotFoundError` | Verify file paths |

---

## 🆘 Getting Help

1. Check existing [GitHub Issues](https://github.com/langtrain-ai/langtune/issues)
2. Open a new issue with:
   - Error message
   - Code to reproduce
   - Environment info (`langtune version`)
