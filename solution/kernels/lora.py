"""The LoRA branch of q_proj / v_proj without keeping its input for backward.

What base keeps (peft's LoRA under fp16 autocast, per projection with LoRA):

    x_drop   = lora_dropout(x)                  fp32, x = the input RMSNorm's fp32 output
    x16      = fp16(x_drop)                     autocast's cast for lora_A: SAVED for dA
    a16      = x16 @ A16^T                      [T, r]: saved for dB (small)
    out16    = (a16 @ B16^T) * scaling

x16 is [tokens, 1536] fp16, 3 KB per token, twice per layer (q and v): ~330 MB of the
activation peak on this task's largest batch. It is a pure function of three things backward
can get without storing it:

    x       = RMSNorm(resid)     the residual and rstd the input norm keeps anyway; the same
                                 `_add_rmsnorm_fwd_kernel` on the same row gives the same bits
    mask    = dropout's keep mask, 1 byte per element and only for the real tokens (base's
                                 dropout keeps it too, for every padded position)
    scale   = 1 / (1 - p)        as float32

so `LoraRematFunction` saves (mask, a16) and rebuilds x16 = fp16(where(mask, x * scale, 0)) in
backward: the exact tensor of the forward pass. Every GEMM is the call peft/autocast makes
(F.linear on fp16 operands, weight gradients as grad^T @ input), every rounding point is the
same, the gradient of x is dropout's backward (grad * mask * scale) of the fp16 -> fp32 cast of
dx16. The only extra work in backward is one RMSNorm row recompute (our Triton kernel) and the
masked scale.

Dropout's random stream is base's. The mask is drawn by the same kernel base uses, over the
same padded [B * S, H] layout, in the same order (q's LoRA, then v's): on CUDA F.dropout is
`native_dropout` (PyTorch, aten/src/ATen/native/Dropout.cpp: `dropout` calls `native_dropout`
when the fused kernel is acceptable, i.e. a CUDA tensor and 0 < p < 1), which also returns the
mask; on the CPU F.dropout is `bernoulli_(1 - p)` on an empty tensor and `x * noise / (1 - p)`,
which `_dropout_with_mask` reproduces call for call. `dropout_self_check` verifies on the run's
own device, before training, that both give F.dropout's exact output, RNG advance and gradient;
if anything differs the fast path keeps peft's own LoRA branch (`ENABLED` turns False).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .. import _runtime
from .rmsnorm import _norm_fwd

# Turned off by `dropout_self_check` if this device's dropout does not match.
STATE = {"enabled": True, "checked": False}


def _scale(p: float) -> torch.Tensor:
    """1 / (1 - p) as dropout's kernels apply it, in float32 (forward and backward: for this
    task's p = 0.05 the float division, the CUDA kernel's double division and the backward's
    double scale all round to the same float32, checked in `dropout_self_check`)."""
    return torch.tensor(1.0, dtype=torch.float32) / torch.tensor(1.0 - p, dtype=torch.float32)


def _dropout_with_mask(x: torch.Tensor, p: float):
    """(F.dropout(x, p, training=True), keep mask), drawing the same random numbers."""
    if x.is_cuda:
        out, mask = torch.native_dropout(x, p, True)
        return out, mask
    noise = torch.empty_like(x).bernoulli_(1.0 - p)
    out = x * noise.div_(1.0 - p)
    return out, noise != 0


class LoraRematFunction(torch.autograd.Function):
    """out16 = (fp16(dropout(x32)) @ A^T @ B^T) * scaling, the LoRA term of q_proj / v_proj.

    Inputs: x32 [T, H] (the input norm's fp32 output), A [r, H], B [out, r] (fp32 trainable),
    and, not differentiated: resid [T, H] fp32 + rstd [T] (to rebuild x32), the norm's weight
    and eps, scaling, p, and the dropout layout (dst, padded rows, a shared padded copy of x32).
    """

    @staticmethod
    def forward(ctx, x32, lora_a, lora_b, resid, rstd, norm_weight, eps, scaling, p, layout):
        with torch.autocast(device_type=x32.device.type, enabled=False):
            if p > 0:
                if layout.full:
                    dropped, mask = _dropout_with_mask(x32.detach(), p)
                else:  # base's layout: the padded [B * S, H] rows, then the real ones
                    dropped, mask = _dropout_with_mask(layout.padded(), p)
                    dropped = dropped.index_select(0, layout.dst)
                    mask = mask.index_select(0, layout.dst)
            else:
                dropped, mask = x32.detach(), None
            x16 = dropped.to(torch.float16)
            del dropped
            a16 = F.linear(x16, lora_a.to(torch.float16))
            out = F.linear(a16, lora_b.to(torch.float16)) * scaling
        del x16
        ctx.save_for_backward(resid, rstd, norm_weight, lora_a, lora_b, a16,
                              *(() if mask is None else (mask,)))
        ctx.eps, ctx.scaling, ctx.p = eps, scaling, p
        ctx.x_dtype = x32.dtype
        return out

    @staticmethod
    def backward(ctx, dout):
        saved = ctx.saved_tensors
        resid, rstd, norm_weight, lora_a, lora_b, a16 = saved[:6]
        mask = saved[6] if len(saved) > 6 else None
        a16_w, b16_w = lora_a.to(torch.float16), lora_b.to(torch.float16)
        db = dout.to(torch.float16) * ctx.scaling
        grad_b = db.t().mm(a16)  # F.linear's weight gradient: grad^T @ input
        da = db.mm(b16_w)
        del db, a16
        # The forward's x16, rebuilt: the same norm kernel on the same residual row, then the
        # dropout's own masked scale.
        _, x32, _ = _norm_fwd(resid, None, norm_weight, ctx.eps)
        if mask is not None:
            scale = _scale(ctx.p).to(x32.device)
            x32 = torch.where(mask, x32 * scale, torch.zeros((), dtype=x32.dtype,
                                                            device=x32.device))
        grad_a = da.t().mm(x32.to(torch.float16))
        del x32
        grad_x = None
        if ctx.needs_input_grad[0]:
            dx = da.mm(a16_w).to(torch.float32)  # autocast's cast back to x's dtype
            if mask is not None:  # dropout's backward: grad * mask * scale
                dx = torch.where(mask, dx * scale, torch.zeros((), dtype=dx.dtype,
                                                               device=dx.device))
            grad_x = dx.to(ctx.x_dtype)
        return (grad_x, grad_a.to(lora_a.dtype), grad_b.to(lora_b.dtype), None, None, None,
                None, None, None, None)


def lora_remat(x32, lora_a, lora_b, resid, rstd, norm_weight, eps, scaling, p, layout):
    return LoraRematFunction.apply(x32, lora_a, lora_b, resid, rstd, norm_weight, float(eps),
                                   float(scaling), float(p), layout)


def enabled() -> bool:
    return _runtime.LORA_REMAT and STATE["enabled"]


def dropout_self_check(device: torch.device, p: float) -> Optional[str]:
    """None if `_dropout_with_mask` + the backward formulas give F.dropout's exact output,
    gradient and RNG advance on `device`; else why not (and the feature turns off).

    Runs in the main thread before training, inside torch.random.fork_rng, so the training run's
    RNG state is exactly what it was before the check."""
    STATE["checked"] = True
    try:
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            n = torch.arange(3 * 4096, device=device, dtype=torch.float32)
            x = (torch.sin(n * 0.37) * 3.0).reshape(3, 4096)
            torch.manual_seed(1234)
            xr = x.clone().requires_grad_(True)
            ref = F.dropout(xr, p, True)
            after_ref = torch.rand(8, device=device)
            g = torch.cos(n * 0.11).reshape(3, 4096)
            ref.backward(g)
            torch.manual_seed(1234)
            out, mask = _dropout_with_mask(x, p)
            after_ours = torch.rand(8, device=device)
            scale = _scale(p).to(device)
            zero = torch.zeros((), device=device)
            if not torch.equal(out, ref.detach()):
                problem = "output differs"
            elif not torch.equal(after_ours, after_ref):
                problem = "RNG advance differs"
            elif not torch.equal(torch.where(mask, x * scale, zero), out):
                problem = "masked-scale recompute differs"
            elif not torch.equal(torch.where(mask, g * scale, zero), xr.grad):
                problem = "backward differs"
            else:
                problem = None
    except Exception as exc:  # noqa: BLE001
        problem = f"check raised {exc!r}"
    if problem is not None:
        STATE["enabled"] = False
        _runtime.log(f"LoRA input rematerialization off: dropout check: {problem}")
    return problem
