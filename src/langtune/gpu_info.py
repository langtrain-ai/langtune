"""
gpu_info.py — Langtune GPU Detection & Auto-Configuration
==========================================================

Detects available hardware and automatically selects the optimal
kernel stack, attention implementation, and data type.

Works seamlessly in:
  - Jupyter notebooks (rich HTML or plain-text output)
  - Python scripts (terminal ANSI color or plain)
  - Colab / Kaggle / SageMaker / Modal

Detection hierarchy:
  1. NVIDIA CUDA  — FA2, BnB 4-bit, Triton JIT, full stack
  2. Apple MPS    — No FA2/BnB, bfloat16 on M2+, float16 on M1
  3. Google TPU   — torch_xla, no BnB, no FA2
  4. CPU only     — no quantization, float32

Usage:
    from langtune.gpu_info import get_gpu_info, print_gpu_info, auto_config

    info = get_gpu_info()         # dict with all detected capabilities
    auto_config()                 # prints banner on first call
    cfg = auto_config(silent=True)  # returns config dict, no print
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    # Accelerator type
    has_cuda: bool = False
    has_mps: bool = False
    has_tpu: bool = False
    is_cpu_only: bool = False

    # CUDA / NVIDIA details
    gpu_name: str = ""
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    compute_capability: Tuple[int, int] = (0, 0)   # e.g. (8, 0) for A100
    cuda_version: str = ""
    multi_gpu: bool = False

    # Apple details
    mps_version: str = ""
    apple_chip: str = ""          # "M1", "M2", "M3", "M4", etc.

    # TPU details
    tpu_version: str = ""
    tpu_cores: int = 0

    # Capabilities derived from hardware
    supports_flash_attention_2: bool = False   # Ampere+ (sm_80+), CUDA only
    supports_bfloat16: bool = False            # Ampere+ CUDA, M2+ MPS
    supports_4bit_quant: bool = False          # CUDA only (bitsandbytes)
    supports_8bit_quant: bool = False          # CUDA only

    # Package availability
    flash_attn_available: bool = False
    bitsandbytes_available: bool = False
    triton_available: bool = False
    peft_available: bool = False
    trl_available: bool = False

    # Recommended settings
    recommended_dtype: str = "float32"         # "bfloat16" | "float16" | "float32"
    recommended_attn: str = "eager"            # "flash_attention_2" | "sdpa" | "eager"
    recommended_load_in_4bit: bool = False
    recommended_max_seq_length: int = 2048


@lru_cache(maxsize=1)
def get_gpu_info() -> GPUInfo:
    """
    Detect all available hardware and package capabilities.
    Result is cached after first call — safe to call repeatedly.
    """
    info = GPUInfo()

    # ── Import torch ─────────────────────────────────────────────────────────
    try:
        import torch
    except ImportError:
        info.is_cpu_only = True
        info.recommended_dtype = "float32"
        return info

    # ── NVIDIA CUDA ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        info.has_cuda = True
        info.gpu_count = torch.cuda.device_count()
        info.multi_gpu = info.gpu_count > 1

        try:
            props = torch.cuda.get_device_properties(0)
            info.gpu_name = props.name
            info.gpu_memory_gb = props.total_memory / (1024 ** 3)
            info.compute_capability = (props.major, props.minor)
        except Exception:
            info.gpu_name = torch.cuda.get_device_name(0)

        info.cuda_version = torch.version.cuda or ""

        cc_major = info.compute_capability[0]

        # Ampere (sm_80) = A100/A30; sm_86 = A10G/RTX 30xx
        # Ada (sm_89) = L4/RTX 40xx; Hopper (sm_90) = H100
        info.supports_flash_attention_2 = cc_major >= 8
        info.supports_bfloat16 = cc_major >= 8
        info.supports_4bit_quant = True
        info.supports_8bit_quant = True

        # Dtype selection: bfloat16 on Ampere+, float16 on older
        info.recommended_dtype = "bfloat16" if info.supports_bfloat16 else "float16"

        # Attention: FA2 if available and Ampere+, else SDPA
        info.recommended_load_in_4bit = info.gpu_memory_gb < 24

    # ── Apple MPS ────────────────────────────────────────────────────────────
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info.has_mps = True

        # Detect Apple chip generation from platform
        try:
            import platform
            cpu_brand = platform.processor() or ""
            import subprocess
            brand_raw = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode().strip()
            if "M4" in brand_raw:
                info.apple_chip = "M4"
            elif "M3" in brand_raw:
                info.apple_chip = "M3"
            elif "M2" in brand_raw:
                info.apple_chip = "M2"
            elif "M1" in brand_raw:
                info.apple_chip = "M1"
            else:
                info.apple_chip = "Apple Silicon"
        except Exception:
            info.apple_chip = "Apple Silicon"

        # M2+ supports bfloat16 in MPS
        info.supports_bfloat16 = info.apple_chip not in ("M1",)
        info.recommended_dtype = "bfloat16" if info.supports_bfloat16 else "float16"

        # FA2 and BnB don't support MPS
        info.supports_flash_attention_2 = False
        info.supports_4bit_quant = False
        info.supports_8bit_quant = False
        info.recommended_attn = "sdpa"

    # ── Google TPU (torch_xla) ────────────────────────────────────────────────
    elif _check_tpu():
        info.has_tpu = True
        try:
            import torch_xla.core.xla_model as xm
            info.tpu_cores = xm.xrt_world_size()
            tpu_name = os.environ.get("TPU_NAME", "")
            for v in ("v5", "v4", "v3", "v2"):
                if v in tpu_name.lower():
                    info.tpu_version = v
                    break
            info.tpu_version = info.tpu_version or "v4"
        except Exception:
            info.tpu_cores = 1
            info.tpu_version = "?"
        info.supports_bfloat16 = True
        info.recommended_dtype = "bfloat16"

    # ── CPU only ─────────────────────────────────────────────────────────────
    else:
        info.is_cpu_only = True
        info.recommended_dtype = "float32"

    # ── Attention implementation ──────────────────────────────────────────────
    if info.has_cuda and info.supports_flash_attention_2:
        # Prefer flash_attn2 if the package is installed
        try:
            import flash_attn  # noqa
            info.flash_attn_available = True
            info.recommended_attn = "flash_attention_2"
        except ImportError:
            # No flash_attn package — use torch SDPA (still fast on Ampere+)
            info.recommended_attn = "sdpa"
    elif info.has_cuda:
        info.recommended_attn = "sdpa"
    elif info.has_mps:
        info.recommended_attn = "sdpa"

    # ── Package availability ──────────────────────────────────────────────────
    info.bitsandbytes_available = _pkg_available("bitsandbytes") and info.has_cuda
    info.triton_available = _pkg_available("triton") and info.has_cuda
    info.peft_available = _pkg_available("peft")
    info.trl_available = _pkg_available("trl")

    # ── Final recommended seq length ──────────────────────────────────────────
    if info.has_cuda:
        if info.gpu_memory_gb >= 80:     # A100 80GB / H100
            info.recommended_max_seq_length = 8192
        elif info.gpu_memory_gb >= 40:   # A100 40GB
            info.recommended_max_seq_length = 4096
        elif info.gpu_memory_gb >= 24:   # A10G / 3090 / 4090
            info.recommended_max_seq_length = 4096
        elif info.gpu_memory_gb >= 16:   # T4 / 2080 Ti / 4080
            info.recommended_max_seq_length = 2048
        else:
            info.recommended_max_seq_length = 1024

    return info


def _check_tpu() -> bool:
    try:
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        return "xla" in str(dev).lower() or "tpu" in str(dev).lower()
    except Exception:
        return False


def _pkg_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Jupyter detection
# ─────────────────────────────────────────────────────────────────────────────

def _in_jupyter() -> bool:
    """True if running inside a Jupyter kernel (notebook, lab, Colab, Kaggle)."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = type(shell).__name__
        return "ZMQ" in shell_name or "Colab" in str(type(shell))
    except Exception:
        return False


def _in_colab() -> bool:
    try:
        import google.colab  # noqa
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Pretty banner printer
# ─────────────────────────────────────────────────────────────────────────────

_BANNER_PRINTED = False


def print_gpu_info(info: Optional[GPUInfo] = None, force: bool = False) -> None:
    """
    Print a hardware summary banner. Outputs HTML in Jupyter, ANSI in terminals.
    No-op if already printed (pass force=True to re-print).
    """
    global _BANNER_PRINTED
    if _BANNER_PRINTED and not force:
        return
    _BANNER_PRINTED = True

    if info is None:
        info = get_gpu_info()

    if _in_jupyter():
        _print_jupyter(info)
    else:
        _print_terminal(info)


def _accel_label(info: GPUInfo) -> str:
    if info.has_cuda:
        mem = f"{info.gpu_memory_gb:.0f}GB" if info.gpu_memory_gb else ""
        cc = f"sm_{info.compute_capability[0]}{info.compute_capability[1]}"
        count = f" ×{info.gpu_count}" if info.multi_gpu else ""
        return f"NVIDIA {info.gpu_name}{count} ({mem}, CUDA {info.cuda_version}, {cc})"
    elif info.has_mps:
        return f"Apple {info.apple_chip} (MPS)"
    elif info.has_tpu:
        return f"Google TPU {info.tpu_version} ({info.tpu_cores} cores)"
    else:
        return "CPU only (no GPU detected)"


def _print_terminal(info: GPUInfo) -> None:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    LIME   = "\033[38;2;204;255;0m"    # #ccff00 — Langtrain accent

    def tick(ok: bool) -> str:
        return f"{GREEN}✓{RESET}" if ok else f"{YELLOW}✗{RESET}"

    lines = [
        f"\n{LIME}{BOLD}  ██╗      █████╗ ███╗   ██╗ ██████╗████████╗██╗   ██╗███╗   ██╗███████╗{RESET}",
        f"{LIME}{BOLD}  ██║     ██╔══██╗████╗  ██║██╔════╝╚══██╔══╝██║   ██║████╗  ██║██╔════╝{RESET}",
        f"{LIME}{BOLD}  ██║     ███████║██╔██╗ ██║██║  ███╗  ██║   ██║   ██║██╔██╗ ██║█████╗  {RESET}",
        f"{LIME}{BOLD}  ██║     ██╔══██║██║╚██╗██║██║   ██║  ██║   ██║   ██║██║╚██╗██║██╔══╝  {RESET}",
        f"{LIME}{BOLD}  ███████╗██║  ██║██║ ╚████║╚██████╔╝  ██║   ╚██████╔╝██║ ╚████║███████╗{RESET}",
        f"{LIME}{BOLD}  ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝{RESET}",
        f"\n  {DIM}Efficient LoRA Fine-Tuning for Text LLMs — Unsloth-compatible API{RESET}",
        "",
        f"  {BOLD}Hardware{RESET}",
        f"    {tick(not info.is_cpu_only)} {_accel_label(info)}",
        "",
        f"  {BOLD}Kernel Stack{RESET}",
        f"    {tick(info.flash_attn_available)}  FlashAttention 2  (2–4× attention)",
        f"    {tick(info.triton_available)}  Triton JIT kernels (RMSNorm 2.6×, RoPE 3.5×)",
        f"    {tick(info.bitsandbytes_available)}  BitsAndBytes 4-bit (26× less VRAM)",
        f"    {tick(info.peft_available)}  PEFT",
        f"    {tick(info.trl_available)}  TRL",
        "",
        f"  {BOLD}Recommended config{RESET}",
        f"    dtype      : {CYAN}{info.recommended_dtype}{RESET}",
        f"    attention  : {CYAN}{info.recommended_attn}{RESET}",
        f"    load_in_4bit: {CYAN}{info.recommended_load_in_4bit}{RESET}",
        f"    max_seq_len: {CYAN}{info.recommended_max_seq_length}{RESET}",
        "",
        f"  {DIM}Set LANGTUNE_NO_BANNER=1 to suppress this message{RESET}\n",
    ]

    if not info.flash_attn_available and info.has_cuda and info.supports_flash_attention_2:
        lines.insert(-1, f"  {YELLOW}💡 Install flash-attn for 2–4× faster attention:{RESET}")
        lines.insert(-1, f"     {DIM}pip install flash-attn --no-build-isolation{RESET}")

    print("\n".join(lines))


def _print_jupyter(info: GPUInfo) -> None:
    try:
        from IPython.display import display, HTML
    except ImportError:
        _print_terminal(info)
        return

    def badge(ok: bool, label: str, detail: str = "") -> str:
        color = "#22c55e" if ok else "#6b7280"
        icon = "✓" if ok else "✗"
        detail_html = f" <span style='color:#94a3b8;font-size:12px'>{detail}</span>" if detail else ""
        return (
            f"<span style='display:inline-flex;align-items:center;gap:6px;margin:3px 0'>"
            f"<span style='color:{color};font-weight:900'>{icon}</span>"
            f"<span style='color:#e2e8f0;font-size:14px'>{label}</span>"
            f"{detail_html}"
            f"</span>"
        )

    accel_ok = not info.is_cpu_only
    accel_color = "#ccff00" if accel_ok else "#fbbf24"

    fa2_hint = ""
    if not info.flash_attn_available and info.has_cuda and info.supports_flash_attention_2:
        fa2_hint = (
            "<div style='margin-top:12px;padding:8px 12px;background:#1e293b;"
            "border-left:3px solid #fbbf24;border-radius:6px;font-size:12px;color:#fbbf24'>"
            "💡 Install flash-attn for 2–4× faster attention: "
            "<code style='background:#0f172a;padding:2px 6px;border-radius:4px'>"
            "pip install flash-attn --no-build-isolation</code>"
            "</div>"
        )

    html = f"""
<div style="font-family:'JetBrains Mono',monospace;background:#050505;border:1px solid rgba(204,255,0,0.2);
            border-radius:16px;padding:24px 28px;margin:12px 0;max-width:680px">
  <div style="color:#ccff00;font-size:22px;font-weight:900;letter-spacing:-0.5px;margin-bottom:4px">
    langtune
  </div>
  <div style="color:#475569;font-size:12px;margin-bottom:20px">
    Efficient LoRA Fine-Tuning · Unsloth-compatible API
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
    <div>
      <div style="color:#64748b;font-size:11px;font-weight:700;letter-spacing:0.1em;margin-bottom:10px">HARDWARE</div>
      <div style="color:{accel_color};font-weight:700;font-size:13px">{_accel_label(info)}</div>
    </div>
    <div>
      <div style="color:#64748b;font-size:11px;font-weight:700;letter-spacing:0.1em;margin-bottom:10px">RECOMMENDED</div>
      <div style="color:#94a3b8;font-size:12px;line-height:1.8">
        dtype: <span style="color:#ccff00">{info.recommended_dtype}</span><br>
        attention: <span style="color:#ccff00">{info.recommended_attn}</span><br>
        4-bit: <span style="color:#ccff00">{info.recommended_load_in_4bit}</span><br>
        max_seq_len: <span style="color:#ccff00">{info.recommended_max_seq_length}</span>
      </div>
    </div>
  </div>

  <div style="border-top:1px solid rgba(255,255,255,0.06);margin:16px 0 14px"></div>
  <div style="color:#64748b;font-size:11px;font-weight:700;letter-spacing:0.1em;margin-bottom:10px">KERNEL STACK</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 24px;font-size:13px">
    {badge(info.flash_attn_available, "FlashAttention 2", "2–4× attention")}
    {badge(info.bitsandbytes_available, "BitsAndBytes 4-bit", "26× less VRAM")}
    {badge(info.triton_available, "Triton JIT", "RMSNorm 2.6×, RoPE 3.5×")}
    {badge(info.peft_available, "PEFT")}
  </div>
  {fa2_hint}
</div>
"""
    display(HTML(html))


# ─────────────────────────────────────────────────────────────────────────────
# Auto-config — the main entry point used by FastLanguageModel
# ─────────────────────────────────────────────────────────────────────────────

_AUTO_CONFIG_CALLED = False


def auto_config(
    *,
    silent: bool = False,
    load_in_4bit: Optional[bool] = None,
    use_flash_attention_2: Optional[bool] = None,
    dtype=None,
    max_seq_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Detect GPU, print banner (first call only), and return optimal config.

    Any explicitly passed argument overrides the auto-detected value.

    Returns:
        {
            "load_in_4bit": bool,
            "use_flash_attention_2": bool,
            "attn_implementation": str,
            "dtype": torch.dtype,
            "max_seq_length": int,
            "device_map": str,
            "info": GPUInfo,
        }
    """
    global _AUTO_CONFIG_CALLED

    info = get_gpu_info()

    if not silent and not _AUTO_CONFIG_CALLED:
        no_banner = os.environ.get("LANGTUNE_NO_BANNER", "0") == "1"
        if not no_banner:
            print_gpu_info(info)
    _AUTO_CONFIG_CALLED = True

    try:
        import torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }
        auto_dtype = dtype_map.get(info.recommended_dtype, torch.float32)
    except ImportError:
        auto_dtype = None

    resolved_4bit = load_in_4bit if load_in_4bit is not None else (
        info.recommended_load_in_4bit and info.bitsandbytes_available
    )
    resolved_fa2 = use_flash_attention_2 if use_flash_attention_2 is not None else (
        info.flash_attn_available
    )
    resolved_dtype = dtype if dtype is not None else auto_dtype
    resolved_seqlen = max_seq_length if max_seq_length is not None else info.recommended_max_seq_length

    attn_impl = "flash_attention_2" if resolved_fa2 else info.recommended_attn

    return {
        "load_in_4bit":          resolved_4bit,
        "use_flash_attention_2": resolved_fa2,
        "attn_implementation":   attn_impl,
        "dtype":                 resolved_dtype,
        "max_seq_length":        resolved_seqlen,
        "device_map":            "auto" if (info.has_cuda or info.has_mps) else "cpu",
        "info":                  info,
    }
