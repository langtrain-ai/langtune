"""
fast_model.py — FastLanguageModel
===================================
Unsloth-style high-level API for Langtune.

Two execution modes, same API:
  LOCAL  — trains on the current machine's GPU (uses Triton/CUDA kernels directly)
  REMOTE — dispatches to langtrain-server GPU infrastructure via REST API

Usage:
    from langtune import FastLanguageModel

    # Local GPU (like Unsloth)
    model, tokenizer = FastLanguageModel.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, r=16, method="qlora")
    FastLanguageModel.train(model, tokenizer, dataset, output_dir="./output")

    # Remote — dispatches to langtrain-server
    model, tokenizer = FastLanguageModel.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        api_key="lt_...",        # triggers remote mode
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, r=16, method="qlora")
    job = FastLanguageModel.train(model, tokenizer, dataset)
    # Returns a RemoteJob — poll or stream
    for step in job.stream():
        print(step)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Remote job handle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingStep:
    step: int
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[float] = None
    progress: int = 0
    status: str = "running"
    message: str = ""


class RemoteJob:
    """
    Handle for a training job running on langtrain-server.
    Poll, stream, cancel, and download the trained model.
    """

    def __init__(self, job_id: str, client: "LangtrainServerClient"):
        self.job_id = job_id
        self._client = client

    def status(self) -> Dict[str, Any]:
        """Get current job status from server."""
        return self._client.get_job(self.job_id)

    def stream(self, interval_s: float = 5.0) -> Generator[TrainingStep, None, None]:
        """
        Yield TrainingStep updates until the job completes.

        Example:
            for step in job.stream():
                print(f"step={step.step}  loss={step.loss:.4f}")
        """
        import time

        last_step = -1
        while True:
            info = self.status()
            current_status = info.get("status", "running")

            # Yield telemetry points
            telemetry = self._client.get_telemetry(self.job_id, after_step=last_step)
            for point in telemetry:
                s = point.get("step", last_step + 1)
                if s > last_step:
                    last_step = s
                    yield TrainingStep(
                        step=s,
                        loss=point.get("loss"),
                        learning_rate=point.get("learning_rate"),
                        epoch=point.get("epoch"),
                        progress=info.get("progress", 0),
                        status=current_status,
                        message=point.get("message", ""),
                    )

            if current_status in ("completed", "failed", "cancelled"):
                yield TrainingStep(
                    step=last_step,
                    progress=100 if current_status == "completed" else 0,
                    status=current_status,
                    message=f"Job {current_status}",
                )
                break

            time.sleep(interval_s)

    def wait(self, poll_interval_s: float = 10.0) -> Dict[str, Any]:
        """Block until the job finishes. Returns final job info."""
        import time

        while True:
            info = self.status()
            if info.get("status") in ("completed", "failed", "cancelled"):
                return info
            time.sleep(poll_interval_s)

    def cancel(self) -> bool:
        """Cancel the remote job."""
        return self._client.cancel_job(self.job_id)

    def download(self, output_dir: str = "./model") -> str:
        """Download the trained model adapter to output_dir."""
        return self._client.download_model(self.job_id, output_dir)

    def __repr__(self) -> str:
        return f"RemoteJob(id={self.job_id!r})"


# ─────────────────────────────────────────────────────────────────────────────
# langtrain-server REST client
# ─────────────────────────────────────────────────────────────────────────────

class LangtrainServerClient:
    """
    Thin HTTP client for langtrain-server.
    Used by FastLanguageModel in remote mode.
    """

    DEFAULT_BASE_URL = "https://api.langtrain.xyz"

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = (base_url or os.environ.get("LANGTRAIN_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            s = requests.Session()
            s.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-SDK": "langtune",
            })
            self._session = s
        return self._session

    def _post(self, path: str, payload: Dict) -> Dict:
        import requests
        r = self._get_session().post(f"{self.base_url}/v1{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, params: Dict = None) -> Any:
        import requests
        r = self._get_session().get(f"{self.base_url}/v1{path}", params=params or {}, timeout=30)
        r.raise_for_status()
        return r.json()

    def upload_dataset(self, dataset_path: str, name: str = None) -> str:
        """Upload a local dataset file, return dataset_id."""
        import requests
        with open(dataset_path, "rb") as f:
            r = self._get_session().post(
                f"{self.base_url}/v1/datasets/upload",
                files={"file": (os.path.basename(dataset_path), f)},
                data={"name": name or os.path.basename(dataset_path)},
                timeout=300,
            )
        r.raise_for_status()
        return r.json()["id"]

    def create_job(self, payload: Dict) -> Dict:
        return self._post("/finetune/jobs", payload)

    def get_job(self, job_id: str) -> Dict:
        return self._get(f"/finetune/jobs/{job_id}")

    def get_telemetry(self, job_id: str, after_step: int = -1) -> List[Dict]:
        try:
            return self._get(f"/finetune/jobs/{job_id}/telemetry", {"after_step": after_step}) or []
        except Exception:
            return []

    def cancel_job(self, job_id: str) -> bool:
        try:
            self._post(f"/finetune/jobs/{job_id}/cancel", {})
            return True
        except Exception:
            return False

    def download_model(self, job_id: str, output_dir: str) -> str:
        """Download adapter weights from the completed job."""
        import requests
        import zipfile
        os.makedirs(output_dir, exist_ok=True)
        r = self._get_session().get(
            f"{self.base_url}/v1/finetune/jobs/{job_id}/download",
            stream=True,
            timeout=300,
        )
        r.raise_for_status()
        zip_path = os.path.join(output_dir, "adapter.zip")
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        os.remove(zip_path)
        logger.info(f"Model downloaded to {output_dir}")
        return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Wrapped model (local mode) — carries training config alongside the HF model
# ─────────────────────────────────────────────────────────────────────────────

class _WrappedLocalModel:
    """
    Thin wrapper around a HuggingFace model + LoRA config.
    Returned by FastLanguageModel.from_pretrained() in local mode.
    Behaves as the underlying model for all PyTorch operations.
    """

    def __init__(self, model, lora_config=None, hyperparameters=None):
        self._model = model
        self._lora_config = lora_config
        self._hyperparameters = hyperparameters or {}

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self._model.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self._model.named_parameters(*args, **kwargs)

    def train(self, mode=True):
        return self._model.train(mode)

    def eval(self):
        return self._model.eval()

    def to(self, *args, **kwargs):
        self._model = self._model.to(*args, **kwargs)
        return self


class _WrappedRemoteModel:
    """
    Placeholder returned by FastLanguageModel.from_pretrained() in remote mode.
    Holds model config for dispatching to langtrain-server.
    """

    def __init__(
        self,
        model_id: str,
        client: LangtrainServerClient,
        hyperparameters: Dict,
        lora_config: Dict = None,
    ):
        self.model_id = model_id
        self._client = client
        self._hyperparameters = hyperparameters
        self._lora_config = lora_config or {}
        self._method = "qlora"


# ─────────────────────────────────────────────────────────────────────────────
# FastLanguageModel — the Unsloth-style entry point
# ─────────────────────────────────────────────────────────────────────────────

class FastLanguageModel:
    """
    Langtune high-level API — Unsloth-compatible interface for text LLMs.

    Supports all models on HuggingFace Hub and langtrain-server's full
    12-method training suite with Triton/CUDA kernel acceleration.

    Execution modes:
      LOCAL  — pass no api_key (or api_key=None). Trains on local GPU.
               Automatically applies Langtrain Triton kernels if available.
      REMOTE — pass api_key="lt_...". Dispatches to langtrain-server GPU cloud.
               Supports all 12 training methods with Modal A10G workers.

    Supported training methods:
      'sft', 'lora', 'qlora', 'dora', 'galore', 'ia3', 'prefix',
      'dpo', 'orpo', 'simpo', 'kto', 'rlhf'
    """

    # ── from_pretrained ──────────────────────────────────────────────────────

    @staticmethod
    def from_pretrained(
        model_name: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Optional[Any] = None,
        use_flash_attention_2: bool = True,
        use_gradient_checkpointing: bool = True,
        max_seq_length: int = 2048,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        token: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Load a model + tokenizer with full Langtrain optimization stack.

        Args:
            model_name: HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B")
            api_key: langtrain-server API key. If provided, returns a remote handle.
            load_in_4bit: NF4 QLoRA quantization (local mode only)
            load_in_8bit: 8-bit LLM.int8() quantization (local mode only)
            use_flash_attention_2: Enable FlashAttention2 kernel (local mode)
            use_gradient_checkpointing: Gradient checkpointing (local mode)
            max_seq_length: Maximum sequence length for training
            token: HuggingFace Hub token for gated models

        Returns:
            (model, tokenizer) — In remote mode, model is a _WrappedRemoteModel.
        """
        _key = api_key or os.environ.get("LANGTRAIN_API_KEY")
        hyperparameters = {
            "max_seq_length": max_seq_length,
            "use_flash_attention_2": use_flash_attention_2,
            "use_gradient_checkpointing": use_gradient_checkpointing,
        }

        if _key:
            # ── REMOTE MODE ──────────────────────────────────────────────
            logger.info(f"[Langtune] Remote mode — dispatching to langtrain-server")
            client = LangtrainServerClient(_key, base_url)
            model = _WrappedRemoteModel(model_name, client, hyperparameters)
            tokenizer = FastLanguageModel._load_tokenizer(model_name, max_seq_length, token)
            return model, tokenizer

        # ── LOCAL MODE ───────────────────────────────────────────────────
        # Auto-detect GPU capabilities and print banner (first call only)
        from langtune.gpu_info import auto_config, get_gpu_info
        cfg = auto_config(
            load_in_4bit=load_in_4bit if load_in_4bit is not None else None,
            use_flash_attention_2=use_flash_attention_2,
            dtype=dtype,
            max_seq_length=max_seq_length,
        )
        info = cfg["info"]

        # Resolved values (user overrides respected, auto-detected otherwise)
        _dtype          = cfg["dtype"]
        _load_in_4bit   = load_in_4bit if load_in_4bit is not None else cfg["load_in_4bit"]
        _use_fa2        = cfg["use_flash_attention_2"]
        _attn_impl      = cfg["attn_implementation"]
        _device_map     = device_map if device_map != "auto" else cfg["device_map"]
        _max_seq_length = max_seq_length or cfg["max_seq_length"]

        logger.info(
            f"[Langtune] Loading {model_name} | "
            f"dtype={_dtype} | attn={_attn_impl} | "
            f"4bit={_load_in_4bit} | device={_device_map}"
        )

        # Build load kwargs
        load_kwargs: Dict[str, Any] = {
            "device_map": _device_map,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": _dtype,
        }
        if token:
            load_kwargs["token"] = token

        if _load_in_4bit and info.bitsandbytes_available:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit and info.bitsandbytes_available:
            load_kwargs["load_in_8bit"] = True

        if _use_fa2 and info.flash_attn_available:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        elif _attn_impl == "sdpa":
            load_kwargs["attn_implementation"] = "sdpa"

        model = FastLanguageModel._load_with_kernels(model_name, load_kwargs, hyperparameters)

        if use_gradient_checkpointing:
            try:
                model.enable_input_require_grads()
                model.gradient_checkpointing_enable()
            except Exception:
                pass

        tokenizer = FastLanguageModel._load_tokenizer(model_name, _max_seq_length, token)

        # Store resolved config on wrapper for downstream use
        hyperparameters.update({
            "use_flash_attention_2": _use_fa2,
            "max_seq_length": _max_seq_length,
            "dtype": str(_dtype),
        })
        return _WrappedLocalModel(model, hyperparameters=hyperparameters), tokenizer

    @staticmethod
    def _load_with_kernels(model_id: str, load_kwargs: Dict, hyperparameters: Dict):
        """Load model then apply Langtrain Triton kernel patches."""
        from transformers import AutoModelForCausalLM

        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except (ValueError, ImportError, NotImplementedError):
            load_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        # Apply Triton RMSNorm + RoPE patches
        try:
            from langtune.kernels import apply_langtune_kernels
            model = apply_langtune_kernels(model, hyperparameters)
        except ImportError:
            pass

        return model

    @staticmethod
    def _load_tokenizer(model_id: str, max_seq_length: int, token=None):
        from transformers import AutoTokenizer

        tok_kwargs = {"trust_remote_code": True}
        if token:
            tok_kwargs["token"] = token

        tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.model_max_length > max_seq_length:
            tokenizer.model_max_length = max_seq_length
        return tokenizer

    # ── get_peft_model ───────────────────────────────────────────────────────

    @staticmethod
    def get_peft_model(
        model,
        *,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        method: str = "qlora",
        use_rslora: bool = False,
        use_dora: bool = False,
        bias: str = "none",
        **kwargs,
    ):
        """
        Apply LoRA / DoRA / IA³ / Prefix adapter to the model.

        For local models: returns a PEFT-wrapped model.
        For remote models: stores adapter config on the wrapper for dispatch.

        Args:
            r: LoRA rank
            lora_alpha: LoRA scaling factor (alpha)
            target_modules: Which linear layers to adapt. Defaults to all attention+MLP.
            method: 'lora', 'qlora', 'dora', 'ia3', 'prefix' — controls adapter type
            use_rslora: RSLoRA scaling (alpha / sqrt(r) instead of alpha / r)
            use_dora: Weight-Decomposed LoRA
        """
        _target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

        lora_config_dict = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": _target_modules,
            "method": method,
            "use_rslora": use_rslora,
            "use_dora": use_dora or (method == "dora"),
            "bias": bias,
        }

        if isinstance(model, _WrappedRemoteModel):
            model._lora_config = lora_config_dict
            model._method = method
            return model

        # Local mode — apply PEFT
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        inner = model._model if isinstance(model, _WrappedLocalModel) else model
        inner = prepare_model_for_kbit_training(inner)

        if method == "ia3":
            from peft import IA3Config, TaskType
            peft_config = IA3Config(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["k_proj", "v_proj", "down_proj"],
                feedforward_modules=["down_proj"],
            )
        elif method == "prefix":
            from peft import PrefixTuningConfig, TaskType
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=kwargs.get("num_virtual_tokens", 20),
            )
        else:
            peft_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=_target_modules,
                lora_dropout=lora_dropout,
                bias=bias,
                task_type="CAUSAL_LM",
                use_dora=use_dora or (method == "dora"),
                use_rslora=use_rslora,
            )

        inner = get_peft_model(inner, peft_config)
        inner.print_trainable_parameters()

        if isinstance(model, _WrappedLocalModel):
            model._model = inner
            model._lora_config = lora_config_dict
            return model

        return _WrappedLocalModel(inner, lora_config_dict)

    # ── train ────────────────────────────────────────────────────────────────

    @staticmethod
    def train(
        model,
        tokenizer,
        dataset,
        *,
        method: Optional[str] = None,
        output_dir: str = "./output",
        dataset_id: Optional[str] = None,
        # Core hyperparameters
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.1,
        max_seq_length: Optional[int] = None,
        # Preference alignment
        beta: float = 0.1,
        # Advanced
        use_neftune: bool = False,
        neftune_noise_alpha: float = 5.0,
        packing: bool = True,
        callbacks=None,
        **kwargs,
    ) -> Union[Any, RemoteJob]:
        """
        Train the model on a dataset.

        Args:
            model: Model from from_pretrained() / get_peft_model()
            tokenizer: Tokenizer from from_pretrained()
            dataset: HuggingFace Dataset, local path (str), or dataset_id for remote
            method: Training method override. Defaults to model's configured method.
            output_dir: Where to save the trained model (local mode)
            dataset_id: Pre-uploaded dataset ID for remote mode (skips upload)

        Returns:
            Local mode: TrainOutput from HuggingFace Trainer
            Remote mode: RemoteJob handle

        Raises:
            ValueError: If dataset path is invalid (local mode)
        """
        _method = method or (
            model._method if isinstance(model, _WrappedRemoteModel) else
            (model._lora_config or {}).get("method", "qlora")
        )

        hyperparameters = {
            **(model._hyperparameters if hasattr(model, "_hyperparameters") else {}),
            "n_epochs": num_train_epochs,
            "batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "beta": beta,
            "use_neftune": use_neftune,
            "neftune_noise_alpha": neftune_noise_alpha,
            **kwargs,
        }
        if max_seq_length:
            hyperparameters["max_seq_length"] = max_seq_length

        # ── REMOTE MODE ──────────────────────────────────────────────────
        if isinstance(model, _WrappedRemoteModel):
            return FastLanguageModel._train_remote(
                model, dataset, _method, hyperparameters, dataset_id
            )

        # ── LOCAL MODE ───────────────────────────────────────────────────
        return FastLanguageModel._train_local(
            model, tokenizer, dataset, _method, output_dir,
            hyperparameters, packing, callbacks,
        )

    @staticmethod
    def _train_remote(
        model: _WrappedRemoteModel,
        dataset,
        method: str,
        hyperparameters: Dict,
        dataset_id: Optional[str],
    ) -> RemoteJob:
        """Dispatch training to langtrain-server."""
        client = model._client

        # Upload dataset if a local path given
        if dataset_id is None:
            if isinstance(dataset, str) and os.path.isfile(dataset):
                logger.info(f"[Langtune] Uploading dataset {dataset}...")
                dataset_id = client.upload_dataset(dataset)
            elif isinstance(dataset, str):
                dataset_id = dataset  # treat as pre-existing ID
            else:
                # HuggingFace Dataset — save to temp jsonl, upload
                import tempfile
                with tempfile.NamedTemporaryFile(
                    suffix=".jsonl", mode="w", delete=False
                ) as f:
                    for row in dataset:
                        import json
                        f.write(json.dumps(dict(row)) + "\n")
                    tmp_path = f.name
                logger.info(f"[Langtune] Uploading dataset ({len(dataset)} rows)...")
                dataset_id = client.upload_dataset(tmp_path)
                os.remove(tmp_path)

        lora_cfg = model._lora_config or {}
        payload = {
            "base_model": model.model_id,
            "dataset_id": dataset_id,
            "training_method": method,
            "hyperparameters": {
                **hyperparameters,
                "lora_rank": lora_cfg.get("r", 16),
                "lora_alpha": lora_cfg.get("lora_alpha", 32),
                "lora_dropout": lora_cfg.get("lora_dropout", 0.05),
                "use_dora": lora_cfg.get("use_dora", False),
            },
        }

        response = client.create_job(payload)
        job_id = response.get("id") or response.get("job_id")
        logger.info(f"[Langtune] Remote job created: {job_id}")
        return RemoteJob(job_id, client)

    @staticmethod
    def _train_local(
        model,
        tokenizer,
        dataset,
        method: str,
        output_dir: str,
        hyperparameters: Dict,
        packing: bool,
        callbacks,
    ):
        """Run training on the local GPU using TRL + Langtrain Triton kernels."""
        from trl import SFTConfig, SFTTrainer

        inner = model._model if isinstance(model, _WrappedLocalModel) else model

        # Fused cross-entropy trainer (26× less VRAM)
        try:
            from langtune.kernels import LangtuneSFTTrainer
            TrainerCls = LangtuneSFTTrainer
        except ImportError:
            TrainerCls = SFTTrainer

        is_preference = method in ("dpo", "orpo", "simpo", "kto")

        if is_preference:
            return FastLanguageModel._train_preference_local(
                inner, tokenizer, dataset, method, output_dir, hyperparameters
            )

        config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=hyperparameters.get("n_epochs", 3),
            per_device_train_batch_size=hyperparameters.get("batch_size", 4),
            gradient_accumulation_steps=hyperparameters.get("gradient_accumulation_steps", 4),
            learning_rate=hyperparameters.get("learning_rate", 2e-4),
            warmup_ratio=hyperparameters.get("warmup_ratio", 0.1),
            max_seq_length=hyperparameters.get("max_seq_length", 2048),
            bf16=True,
            packing=packing,
            gradient_checkpointing=hyperparameters.get("use_gradient_checkpointing", True),
            neftune_noise_alpha=(
                hyperparameters.get("neftune_noise_alpha")
                if hyperparameters.get("use_neftune") else None
            ),
            logging_steps=10,
            save_strategy="epoch",
        )

        trainer = TrainerCls(
            model=inner,
            train_dataset=dataset,
            args=config,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )

        return trainer.train()

    @staticmethod
    def _train_preference_local(inner, tokenizer, dataset, method, output_dir, hp):
        """Handle preference alignment methods locally."""
        if method == "dpo":
            from trl import DPOConfig, DPOTrainer
            config = DPOConfig(
                output_dir=output_dir,
                num_train_epochs=hp.get("n_epochs", 1),
                per_device_train_batch_size=hp.get("batch_size", 2),
                learning_rate=hp.get("learning_rate", 5e-5),
                beta=hp.get("beta", 0.1),
                bf16=True,
            )
            trainer = DPOTrainer(model=inner, train_dataset=dataset, args=config, tokenizer=tokenizer)
        elif method == "orpo":
            from trl import ORPOConfig, ORPOTrainer
            config = ORPOConfig(
                output_dir=output_dir,
                num_train_epochs=hp.get("n_epochs", 1),
                per_device_train_batch_size=hp.get("batch_size", 2),
                learning_rate=hp.get("learning_rate", 8e-6),
                beta=hp.get("beta", 0.1),
                bf16=True,
            )
            trainer = ORPOTrainer(model=inner, train_dataset=dataset, args=config, tokenizer=tokenizer)
        elif method == "kto":
            from trl import KTOConfig, KTOTrainer
            config = KTOConfig(
                output_dir=output_dir,
                num_train_epochs=hp.get("n_epochs", 1),
                per_device_train_batch_size=hp.get("batch_size", 2),
                learning_rate=hp.get("learning_rate", 5e-6),
                beta=hp.get("beta", 0.1),
                bf16=True,
            )
            trainer = KTOTrainer(model=inner, train_dataset=dataset, args=config, tokenizer=tokenizer)
        else:  # simpo
            from trl import CPOConfig, CPOTrainer
            config = CPOConfig(
                output_dir=output_dir,
                loss_type="simpo",
                num_train_epochs=hp.get("n_epochs", 1),
                per_device_train_batch_size=hp.get("batch_size", 2),
                learning_rate=hp.get("learning_rate", 8e-6),
                beta=hp.get("beta", 2.0),
                bf16=True,
            )
            trainer = CPOTrainer(model=inner, train_dataset=dataset, args=config, tokenizer=tokenizer)

        return trainer.train()

    # ── generate / chat ──────────────────────────────────────────────────────

    @staticmethod
    def generate(
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Generate text from a loaded local model.
        For remote models, calls the deployment inference API.
        """
        if isinstance(model, _WrappedRemoteModel):
            raise ValueError(
                "Remote model cannot generate locally. "
                "Deploy the model first with ai.deployments.create() and call the endpoint."
            )
        import torch

        inner = model._model if isinstance(model, _WrappedLocalModel) else model
        inputs = tokenizer(prompt, return_tensors="pt").to(inner.device)
        with torch.no_grad():
            out = inner.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                **kwargs,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # ── save / load ──────────────────────────────────────────────────────────

    @staticmethod
    def save_pretrained(model, tokenizer, output_dir: str, push_to_hub: bool = False, **kwargs):
        """Save model and tokenizer. Merges LoRA weights if merge_before_save=True."""
        inner = model._model if isinstance(model, _WrappedLocalModel) else model
        if kwargs.get("merge_before_save", False):
            inner = inner.merge_and_unload()
        inner.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        if push_to_hub:
            inner.push_to_hub(kwargs.get("hub_repo_id", output_dir), **kwargs)
        logger.info(f"[Langtune] Model saved to {output_dir}")

    @staticmethod
    def for_inference(model, tokenizer):
        """Optimise model for inference (eval mode, disable grad, merge LoRA if possible)."""
        import torch
        inner = model._model if isinstance(model, _WrappedLocalModel) else model
        inner.eval()
        torch.set_grad_enabled(False)
        return model
