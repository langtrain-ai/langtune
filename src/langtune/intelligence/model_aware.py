"""
Model-Aware Optimization Module

Automatically detects model architecture and selects optimal training config.
Provides zero-YAML configuration based on model characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import re


class ModelFamily(Enum):
    """Known model families with specific optimizations."""
    LLAMA = "llama"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    QWEN = "qwen"
    PHI = "phi"
    FALCON = "falcon"
    MPT = "mpt"
    BLOOM = "bloom"
    OPT = "opt"
    UNKNOWN = "unknown"


class AttentionType(Enum):
    """Attention mechanism types."""
    MHA = "multi_head_attention"  # Standard
    GQA = "grouped_query_attention"  # LLaMA 2+
    MQA = "multi_query_attention"  # Falcon, etc.


@dataclass
class ModelAnalysis:
    """Analysis of model architecture."""
    
    family: ModelFamily = ModelFamily.UNKNOWN
    attention_type: AttentionType = AttentionType.MHA
    
    # Architecture details
    num_layers: int = 0
    hidden_size: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    vocab_size: int = 0
    
    # Context
    max_position_embeddings: int = 2048
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Memory characteristics
    estimated_params_b: float = 0.0  # Billions
    estimated_vram_fp16_gb: float = 0.0
    
    # Quantization tolerance (0-1)
    quantization_tolerance: float = 0.8
    
    # Training characteristics
    recommended_batch_size: int = 4
    gradient_checkpointing_recommended: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family.value,
            "attention_type": self.attention_type.value,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "estimated_params_b": round(self.estimated_params_b, 2),
            "estimated_vram_fp16_gb": round(self.estimated_vram_fp16_gb, 1),
            "quantization_tolerance": self.quantization_tolerance,
            "gradient_checkpointing_recommended": self.gradient_checkpointing_recommended,
        }


class ModelAnalyzer:
    """
    Analyzes model architecture for optimal training configuration.
    """
    
    FAMILY_PATTERNS = {
        ModelFamily.LLAMA: [r"llama", r"meta-llama"],
        ModelFamily.MISTRAL: [r"mistral"],
        ModelFamily.GEMMA: [r"gemma"],
        ModelFamily.QWEN: [r"qwen"],
        ModelFamily.PHI: [r"phi"],
        ModelFamily.FALCON: [r"falcon"],
        ModelFamily.MPT: [r"mpt"],
        ModelFamily.BLOOM: [r"bloom"],
        ModelFamily.OPT: [r"opt"],
    }
    
    def analyze(self, model: Any, model_name: str = "") -> ModelAnalysis:
        """
        Analyze model architecture.
        
        Args:
            model: The model to analyze
            model_name: Optional model name for family detection
            
        Returns:
            ModelAnalysis with detected characteristics
        """
        analysis = ModelAnalysis()
        
        # Detect family from name
        analysis.family = self._detect_family(model_name, model)
        
        # Get config if available
        config = getattr(model, 'config', None)
        
        if config:
            analysis.num_layers = getattr(config, 'num_hidden_layers', 0)
            analysis.hidden_size = getattr(config, 'hidden_size', 0)
            analysis.num_heads = getattr(config, 'num_attention_heads', 0)
            analysis.num_kv_heads = getattr(config, 'num_key_value_heads', analysis.num_heads)
            analysis.vocab_size = getattr(config, 'vocab_size', 0)
            analysis.max_position_embeddings = getattr(config, 'max_position_embeddings', 2048)
            
            # RoPE scaling
            rope = getattr(config, 'rope_scaling', None)
            if rope:
                analysis.rope_scaling = dict(rope)
        
        # Detect attention type
        analysis.attention_type = self._detect_attention_type(analysis)
        
        # Estimate parameters and VRAM
        analysis.estimated_params_b = self._estimate_params(analysis)
        analysis.estimated_vram_fp16_gb = analysis.estimated_params_b * 2  # ~2GB per billion params in fp16
        
        # Quantization tolerance based on family
        analysis.quantization_tolerance = self._get_quantization_tolerance(analysis.family)
        
        # Gradient checkpointing recommendation
        analysis.gradient_checkpointing_recommended = analysis.estimated_params_b > 3
        
        return analysis
    
    def _detect_family(self, model_name: str, model: Any) -> ModelFamily:
        """Detect model family from name or architecture."""
        name_lower = model_name.lower()
        
        for family, patterns in self.FAMILY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    return family
        
        # Try to detect from model class name
        class_name = model.__class__.__name__.lower()
        for family, patterns in self.FAMILY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, class_name):
                    return family
        
        return ModelFamily.UNKNOWN
    
    def _detect_attention_type(self, analysis: ModelAnalysis) -> AttentionType:
        """Detect attention mechanism type."""
        if analysis.num_kv_heads == 1:
            return AttentionType.MQA
        elif analysis.num_kv_heads < analysis.num_heads:
            return AttentionType.GQA
        return AttentionType.MHA
    
    def _estimate_params(self, analysis: ModelAnalysis) -> float:
        """Estimate parameter count in billions."""
        if not analysis.hidden_size or not analysis.num_layers:
            return 0.0
        
        # Rough estimation based on standard transformer architecture
        # Embedding: vocab_size * hidden_size
        # Attention: 4 * hidden_size^2 per layer
        # FFN: 8 * hidden_size^2 per layer (assuming 4x expansion)
        
        embedding_params = analysis.vocab_size * analysis.hidden_size
        attention_params = 4 * (analysis.hidden_size ** 2) * analysis.num_layers
        ffn_params = 8 * (analysis.hidden_size ** 2) * analysis.num_layers
        
        total = embedding_params + attention_params + ffn_params
        return total / 1e9  # Convert to billions
    
    def _get_quantization_tolerance(self, family: ModelFamily) -> float:
        """Get quantization tolerance for model family."""
        # Based on empirical observations
        tolerances = {
            ModelFamily.LLAMA: 0.9,
            ModelFamily.MISTRAL: 0.85,
            ModelFamily.GEMMA: 0.8,
            ModelFamily.QWEN: 0.85,
            ModelFamily.PHI: 0.75,
            ModelFamily.FALCON: 0.7,
        }
        return tolerances.get(family, 0.7)


@dataclass 
class AutoConfig:
    """Auto-generated training configuration."""
    
    # Training type
    training_type: str = "lora"  # lora, qlora, full
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=list)
    
    # Precision
    precision: str = "bf16"  # bf16, fp16, fp32
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-4
    
    # Memory optimization
    use_flash_attention: bool = True
    use_unsloth: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "training_type": self.training_type,
            "lora": {
                "r": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "target_modules": self.target_modules,
            },
            "precision": self.precision,
            "quantization": {
                "load_in_4bit": self.load_in_4bit,
                "load_in_8bit": self.load_in_8bit,
            },
            "training": {
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "gradient_checkpointing": self.gradient_checkpointing,
                "learning_rate": self.learning_rate,
            },
            "optimizations": {
                "flash_attention": self.use_flash_attention,
                "unsloth": self.use_unsloth,
            },
        }


class AutoConfigurator:
    """
    Automatically generates optimal training configuration.
    
    Zero-YAML configuration: just provide model and VRAM budget.
    """
    
    # Target modules for common architectures
    TARGET_MODULES = {
        ModelFamily.LLAMA: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ModelFamily.MISTRAL: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ModelFamily.GEMMA: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ModelFamily.QWEN: ["c_attn", "c_proj", "w1", "w2"],
        ModelFamily.PHI: ["Wqkv", "out_proj", "fc1", "fc2"],
    }
    
    def __init__(self, analyzer: ModelAnalyzer = None):
        self.analyzer = analyzer or ModelAnalyzer()
    
    def configure(
        self,
        model: Any,
        model_name: str = "",
        vram_budget_gb: float = 24.0,
        task: str = "instruction",  # instruction, chat, code, creative
        quality_priority: float = 0.5,  # 0=speed, 1=quality
    ) -> AutoConfig:
        """
        Generate optimal training configuration.
        
        Args:
            model: The model to configure
            model_name: Model name/path
            vram_budget_gb: Available VRAM in GB
            task: Training task type
            quality_priority: Balance between speed and quality
            
        Returns:
            AutoConfig with optimal settings
        """
        # Analyze model
        analysis = self.analyzer.analyze(model, model_name)
        
        config = AutoConfig()
        
        # Determine training type based on VRAM and model size
        vram_for_fp16 = analysis.estimated_vram_fp16_gb
        
        if vram_budget_gb >= vram_for_fp16 * 3:
            # Enough for full fine-tuning
            config.training_type = "full" if quality_priority > 0.7 else "lora"
        elif vram_budget_gb >= vram_for_fp16 * 1.5:
            # LoRA
            config.training_type = "lora"
        else:
            # QLoRA for constrained VRAM
            config.training_type = "qlora"
            config.load_in_4bit = True
        
        # LoRA rank based on task and quality priority
        if quality_priority > 0.7:
            config.lora_r = 32
        elif quality_priority > 0.4:
            config.lora_r = 16
        else:
            config.lora_r = 8
        
        config.lora_alpha = config.lora_r * 2
        
        # Target modules
        config.target_modules = self.TARGET_MODULES.get(
            analysis.family,
            ["q_proj", "v_proj"]  # Minimal default
        )
        
        # Precision
        config.precision = "bf16" if self._supports_bf16() else "fp16"
        
        # Batch size based on VRAM
        available_for_batch = vram_budget_gb - (vram_for_fp16 / (4 if config.load_in_4bit else 1))
        config.batch_size = max(1, min(8, int(available_for_batch / 2)))
        
        # Gradient accumulation to reach effective batch size of 32
        config.gradient_accumulation_steps = max(1, 32 // config.batch_size)
        
        # Gradient checkpointing
        config.gradient_checkpointing = analysis.gradient_checkpointing_recommended
        
        # Learning rate based on task
        lr_map = {
            "instruction": 2e-4,
            "chat": 1e-4,
            "code": 5e-5,
            "creative": 3e-4,
        }
        config.learning_rate = lr_map.get(task, 2e-4)
        
        # Optimizations
        config.use_flash_attention = analysis.attention_type in [AttentionType.GQA, AttentionType.MHA]
        config.use_unsloth = analysis.family in [ModelFamily.LLAMA, ModelFamily.MISTRAL]
        
        return config
    
    def _supports_bf16(self) -> bool:
        """Check if bf16 is supported."""
        try:
            import torch
            return torch.cuda.is_bf16_supported()
        except Exception:
            return False
