"""Compile and self-check the kernels while Soup loads the model.

Every measured run starts with an empty Triton cache, and the wall time of `soup train` includes
compilation. Model loading (cache read, 4-bit quantization on the GPU, LoRA setup, tokenization)
takes many seconds before the first step, so a thread started at `SFTTrainerWrapper.setup`
launches each kernel once on small tensors with the model's real dimensions: same dtypes, same
constexprs, same launch options, hence the same binaries the training step looks up.

Each launch is also checked against plain PyTorch on fixed (non-random) data. If a kernel fails
to compile or disagrees, `apply_t4_triton_patches` turns the fast path off and the run is plain
Soup, instead of a crash or a wrong adapter. The thread never draws from torch's RNGs (the LoRA
dropout masks and the data order depend on them) and is part of the measured process.
"""

from __future__ import annotations

import atexit
import threading
from typing import Optional

import torch

from . import _runtime

_STATE: dict = {"thread": None, "problems": None}


def start_background_warmup(base: Optional[str]) -> None:
    if _STATE["thread"] is not None or not base or not _runtime.cuda_kernels_supported():
        return
    thread = threading.Thread(target=_run, args=(base,), name="solution-kernel-warmup",
                              daemon=True)
    _STATE["thread"] = thread
    thread.start()
    atexit.register(thread.join, 120)  # never tear the interpreter down mid-compile


def _run(base: str) -> None:
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(base)
        device = torch.device("cuda", torch.cuda.current_device())
        _STATE["problems"] = compile_kernels(config, device)
    except Exception as exc:  # noqa: BLE001
        _STATE["problems"] = [f"warm-up failed: {exc!r}"]


def ensure_checked(config, device, timeout: float = 600.0) -> list[str]:
    """Problems found by the kernel self-check (empty: all good). Waits for the background
    warm-up if it runs, otherwise compiles and checks right here."""
    thread = _STATE["thread"]
    if thread is not None:
        thread.join(timeout)
        if thread.is_alive():
            return ["kernel warm-up still running"]
        problems = _STATE["problems"]
        if problems is not None and not any("warm-up failed" in p for p in problems):
            return problems
    try:
        return compile_kernels(config, device)
    except Exception as exc:  # noqa: BLE001
        return [f"kernel self-check failed: {exc!r}"]


def _close(name, got, want, tol, problems):
    got, want = got.float().cpu(), want.float().cpu()
    if got.shape != want.shape or not torch.isfinite(got).all() or \
            not torch.allclose(got, want, rtol=tol, atol=tol):
        err = (got - want).abs().max().item() if got.shape == want.shape else "shape"
        problems.append(f"{name}: max error {err}")


def compile_kernels(config, device) -> list[str]:
    from .triton_cross_entropy import _launch_ce_kernel
    from .triton_nf4_lora import BLOCK_BYTES, DEQUANT_WARPS, _nf4_dequant_nested_kernel
    from .triton_ops import (_next_pow2, _norm_bwd, _norm_fwd, ref_add_rmsnorm, ref_rope,
                             rope_lora_scatter, swiglu_bwd_inplace, swiglu_fwd)

    hidden = int(config.hidden_size)
    n_heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", None) or hidden // n_heads)
    vocab = int(config.vocab_size)
    inter = int(config.intermediate_size)
    eps = float(config.rms_norm_eps)
    f16, f32 = torch.float16, torch.float32
    problems: list[str] = []

    def pattern(rows, cols, dtype, scale=1.0, requires_grad=False):
        # Fixed, non-random values: this must not touch torch's RNG state.
        n = torch.arange(rows * cols, device=device, dtype=f32)
        values = (torch.sin(n * 0.7311) * scale).reshape(rows, cols).to(dtype)
        return values.requires_grad_(requires_grad)

    with torch.no_grad():
        # residual add + RMSNorm
        resid, delta = pattern(3, hidden, f32, 2.0), pattern(3, hidden, f16, 0.5)
        weight = pattern(1, hidden, f32, 0.3).reshape(-1) + 1.0
        out, y, rstd = _norm_fwd(resid, delta, weight, eps)
        ref_out, ref_y = ref_add_rmsnorm(resid, delta, weight, eps)
        _close("rmsnorm forward", y, ref_y, 1e-4, problems)
        _close("residual add", out, ref_out, 1e-6, problems)
        dx, _ = _norm_bwd(torch.ones_like(out), None, out, weight, rstd, True)
        _close("rmsnorm backward finite", dx, dx.nan_to_num(), 0.0, problems)
        # SwiGLU (fp16)
        g, u = pattern(2, inter, f16, 3.0), pattern(2, inter, f16, 1.0)
        ref_p = (torch.nn.functional.silu(g.float()) * u.float()).to(f16)
        _close("swiglu forward", swiglu_fwd(g, u), ref_p, 2e-3, problems)
        swiglu_bwd_inplace(ref_p, g, u)
        # NF4 unpacking against the same table lookups done by PyTorch
        n_bytes = 128
        packed = (torch.arange(n_bytes, device=device) * 37 % 256).to(torch.uint8)
        absmax8 = (torch.arange(n_bytes // 32, device=device) * 91 % 256).to(torch.uint8)
        code2 = torch.linspace(-1.0, 1.0, 256, device=device, dtype=f32)
        absmax2 = torch.tensor([0.75], device=device, dtype=f32)
        offset = torch.tensor([0.25], device=device, dtype=f32)
        nf4 = torch.linspace(-1.0, 1.0, 16, device=device, dtype=f32)
        out16 = torch.empty(2 * n_bytes, device=device, dtype=f16)
        _nf4_dequant_nested_kernel[(1,)](
            packed, absmax8, code2, absmax2, offset, nf4, out16, n_bytes,
            BLOCK_BYTES=BLOCK_BYTES, QBLOCK=64, QBLOCK2=256, num_warps=DEQUANT_WARPS,
            enable_fp_fusion=False)
        scale = code2[absmax8.long()] * absmax2[0] + offset[0]
        codes = torch.stack(((packed >> 4).long(), (packed & 15).long()), 1).reshape(-1)
        ref_w = (nf4[codes] * scale.repeat_interleave(64)).to(f16)
        _close("nf4 unpack", out16, ref_w, 0.0, problems)
        # lm_head loss kernel
        logits = pattern(2, vocab, f16, 4.0)
        targets = torch.tensor([3, -100], device=device, dtype=torch.int64)
        loss, entropy = (torch.empty(2, device=device, dtype=f32) for _ in range(2))
        hits = torch.empty(2, device=device, dtype=torch.int32)
        z = logits.float()
        lse = torch.logsumexp(z, -1)
        _launch_ce_kernel(logits, targets, loss, entropy, hits,
                          torch.ones(1, device=device, dtype=f32), 0, True)
        _close("ce loss", loss[0], lse[0] - z[0, 3], 1e-4, problems)
        p = torch.softmax(z, -1)
        _close("ce entropy", entropy, -(p * torch.log_softmax(z, -1)).sum(-1), 1e-3, problems)
        want_grad = p[0].clone()
        want_grad[3] -= 1.0
        _close("ce gradient", logits[0], want_grad.to(f16), 1e-3, problems)
    # LoRA add + RoPE scatter, forward and backward (autograd builds the backward call)
    seq = 4
    base16 = pattern(2, n_heads * head_dim, f16, 1.0, True)
    lora16 = pattern(2, n_heads * head_dim, f16, 0.1, True)
    cos = pattern(seq, head_dim, f32, 1.0)
    sin = pattern(seq, head_dim, f32, 0.5)
    rows = torch.tensor([0, 1], device=device, dtype=torch.int64)
    rotated = rope_lora_scatter(base16, lora16, cos, sin, rows, rows, 1, seq, n_heads, True,
                                _next_pow2(n_heads))
    rotated.backward(torch.ones_like(rotated))
    x = (base16.detach().float() + lora16.detach().float()).view(2, n_heads, head_dim)
    want = ref_rope(x, cos[:2, None, :], sin[:2, None, :])
    _close("rope forward", rotated[0, :, :2].transpose(0, 1), want, 1e-5, problems)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return problems
