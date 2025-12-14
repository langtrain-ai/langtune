# Langtune Roadmap

## 🎯 Vision

Langtune aims to be the go-to library for efficient LLM fine-tuning, providing state-of-the-art optimizations while remaining simple and accessible.

---

## ✅ Completed (v0.1.x)

- [x] LoRA adapters for efficient fine-tuning
- [x] Modular transformer architecture
- [x] CLI for training and evaluation
- [x] Mixed precision training (AMP)
- [x] Gradient checkpointing
- [x] Early stopping and checkpointing
- [x] W&B integration
- [x] **Efficient Fine-Tuning (Unsloth-inspired)**
  - [x] 4-bit quantization (QLoRA)
  - [x] Rotary Position Embeddings (RoPE)
  - [x] Memory-efficient attention
  - [x] Fused cross-entropy loss
  - [x] Gradient accumulation
- [x] Multi-accelerator support (NVIDIA, TPU, Apple MPS)
- [x] API key authentication

---

## 🚧 In Progress (v0.2.x)

- [ ] HuggingFace model integration
- [ ] Pre-trained model loading
- [ ] Distributed training (DDP)
- [ ] QLoRA with bitsandbytes
- [ ] Improved tokenizer support

---

## 📋 Planned (v0.3.x)

### Model Support
- [ ] LLaMA 2/3 integration
- [ ] Mistral support
- [ ] Qwen support
- [ ] Custom model architectures

### Training Features
- [ ] FSDP (Fully Sharded Data Parallel)
- [ ] DeepSpeed integration
- [ ] Sequence packing
- [ ] Dynamic batching

### RLHF & Alignment
- [ ] Full RLHF implementation
- [ ] DPO training
- [ ] PPO training
- [ ] Reward modeling

### Deployment
- [ ] ONNX export
- [ ] TensorRT optimization
- [ ] Quantized inference
- [ ] Streaming generation

---

## 🌟 Future Considerations

- Multi-modal support (vision-language)
- Continuous pre-training
- Mixture of Experts (MoE)
- Flash Attention 2/3
- Speculative decoding
- KV cache optimization

---

## 📊 Version Timeline

| Version | Target | Focus |
|---------|--------|-------|
| v0.1.x | ✅ Released | Core LoRA + Optimizations |
| v0.2.x | Q1 2025 | HuggingFace Integration |
| v0.3.x | Q2 2025 | RLHF & Distributed |
| v0.4.x | Q3 2025 | Deployment & Inference |
| v1.0.0 | Q4 2025 | Production Ready |

---

## 💡 Feature Requests

Have an idea? Open an issue with the `enhancement` label!

---

*Last updated: December 2024*
