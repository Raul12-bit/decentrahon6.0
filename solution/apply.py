"""The single entry point: `apply()` plugs our kernels into `soup train` (the way Unsloth patches
models). The harness calls `solution.apply()` once, in the same process as `soup train`, before
Soup builds the model and the trainer.

What it installs (class-level patches, looked up at call time; details in each module):
- patches/patch_qwen.py   the fused training forward of Soup's 4-bit Qwen2 + LoRA (no pad
                          tokens, no [B, S, V] logits), the TRL metrics / rewind hook, the fp16
                          copy of the tied embedding table, kernel pre-compilation + self-check;
- kernels/cross_entropy.py lm_head + cross-entropy + entropy / accuracy in token chunks;
- kernels/rmsnorm.py, rope.py, swiglu.py   residual add + RMSNorm, LoRA add + RoPE, fp16 SwiGLU;
- kernels/nf4.py          NF4 double-quant unpacking in one kernel.

Set SOUP_KERNELS=0 to make apply() a no-op (plain base Soup) for A/B runs.
"""

from __future__ import annotations

from . import _runtime

_APPLIED: list[str] = []


def apply() -> list[str]:
    """Install the patches. Safe to call twice; never raises (falls back to base Soup)."""
    if _APPLIED or not _runtime.ENABLED:
        return list(_APPLIED)
    if not _runtime.triton_available():
        _runtime.log("Triton is not importable: running plain Soup")
        return []
    try:
        from .patches.patch_qwen import install

        _APPLIED.extend(install())
    except Exception as exc:  # noqa: BLE001 - a broken patch must not break `soup train`
        _runtime.log(f"patches not installed ({exc!r}): running plain Soup")
        return []
    if not _runtime.cuda_kernels_supported():
        _runtime.log("no CUDA GPU with compute capability >= 7.5: the fast path stays off")
    _runtime.log("installed: " + ", ".join(_APPLIED))
    return list(_APPLIED)


def apply_t4_triton_patches(model):
    """Prepare an already built model by hand (outside `soup train`)."""
    from .patches.patch_qwen import apply_t4_triton_patches as prepare

    return prepare(model)
