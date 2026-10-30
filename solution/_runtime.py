"""Feature switches and capability checks shared by every module of the solution.

Every switch is an environment variable read once at import time, so a run can be
reproduced (or bisected on Kaggle) without editing code:

    SOUP_KERNELS=0                 apply() patches nothing: plain base Soup
    SOUP_KERNELS_UNPAD=0           keep pad tokens in the decoder (default: drop them)
    SOUP_KERNELS_FP16_EMBED=0      keep the fp32 tied embedding / lm_head copy
    SOUP_KERNELS_WARMUP=0          do not pre-compile the kernels while the model loads
    SOUP_KERNELS_CE_CHUNK=256      rows of logits per chunk of the fused lm_head + loss
    SOUP_KERNELS_MLP_RECOMPUTE=0   keep the fp16 gate/up outputs for backward instead of
                                   recomputing them from fp16 x (default 1: recompute; the
                                   two [tokens, 8960] tensors are ~35 KB per token per layer,
                                   ~1.7 GB of activations on the largest batch, against two
                                   extra GEMMs per layer in backward that reuse the weight the
                                   input gradient unpacks anyway)
    SOUP_KERNELS_VARLEN_ATTN=0     attention on the padded [B, H, S, D] layout with a mask (the
                                   default 1 runs it per row on the row's real tokens when the
                                   batch is right-padded: ~42% of the padded attention work on
                                   this task's data, and no q/k/v/output storage for pads)
    SOUP_KERNELS_FUSED_MLP_NORM=0  separate post-attention RMSNorm (fp32 output + cast) and MLP
                                   (default 1: one autograd function, fp16 norm output, x16 not
                                   saved for backward)
    SOUP_KERNELS_VERBOSE=1         print why a step falls back to the plain path

The kernels target a Tesla T4 (sm_75): fp16 storage, fp32 math, no bf16.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


ENABLED = _flag("SOUP_KERNELS", True)
UNPAD = _flag("SOUP_KERNELS_UNPAD", True)
FP16_EMBED = _flag("SOUP_KERNELS_FP16_EMBED", True)
WARMUP = _flag("SOUP_KERNELS_WARMUP", True)
MLP_RECOMPUTE = _flag("SOUP_KERNELS_MLP_RECOMPUTE", True)
VARLEN_ATTN = _flag("SOUP_KERNELS_VARLEN_ATTN", True)
FUSED_MLP_NORM = _flag("SOUP_KERNELS_FUSED_MLP_NORM", True)
CE_CHUNK = max(16, _int("SOUP_KERNELS_CE_CHUNK", 256))
VERBOSE = _flag("SOUP_KERNELS_VERBOSE", False)
# Only for the kernel tests on a machine without a GPU: TRITON_INTERPRET=1 runs the
# kernels on CPU tensors, and this lets the wrappers accept them.
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def next_pow2(n: int) -> int:
    """Smallest power of two >= n (Triton block sizes must be powers of two)."""
    return 1 << max(0, int(n) - 1).bit_length()


def log(message: str) -> None:
    print(f"[solution] {message}", file=sys.stderr, flush=True)


def debug(message: str) -> None:
    if VERBOSE:
        log(message)


@lru_cache(maxsize=None)
def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except Exception:  # noqa: BLE001 - any import problem means "no Triton"
        return False
    return True


@lru_cache(maxsize=None)
def cuda_kernels_supported() -> bool:
    """Triton on a CUDA GPU of compute capability 7.5 (T4) or newer."""
    if not triton_available():
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001
        return False
    return (major, minor) >= (7, 5)


def device_ok(tensor) -> bool:
    """True when the kernels may run on this tensor's device."""
    if tensor.is_cuda:
        return cuda_kernels_supported()
    return INTERPRET and triton_available()
