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
    mask    = dropout's keep mask, only for the real tokens, packed 8 per byte (base's dropout
                                 keeps it as 1 byte per element, for every padded position)
    scale   = 1 / (1 - p)        as float32

so `LoraRematFunction` saves (mask bits, a16) and rebuilds x16 = fp16(where(mask, x * scale, 0))
in backward: the exact tensor of the forward pass. Every GEMM is the call peft/autocast makes
(F.linear on fp16 operands, weight gradients as grad^T @ input), every rounding point is the
same, the gradient of x is dropout's backward (grad * mask * scale) of the fp16 -> fp32 cast of
dx16. The only extra work in backward is one RMSNorm row recompute (our Triton kernel) and the
masked scale.

The mask bits (phase 4): `_dropout_mask_pack_kernel` reads the bool keep mask dropout returns on
the padded [B * S, H] layout, picks the real tokens' rows (the gather `index_select` used to do)
and writes bit j of byte k = column 8k + j: 0.19 KB per token instead of 1.5 KB, twice per
layer. In backward `_dropout_bits_bwd_kernel` unpacks a row once and applies dropout's masked
scale to both tensors that need it: fp16(where(keep, x * scale, 0)), the rebuilt input of lora_A,
and where(keep, fp32(dx16) * scale, 0), the fp32 gradient of x. The same fp32 products and the
same rounding as the torch.where chains it replaces (one kernel instead of six), so the values
are bit for bit the same. Two small binaries to compile, both built by the warm-up thread.

Dropout's random stream is base's. The mask is drawn by the same kernel base uses, over the
same padded [B * S, H] layout, in the same order (q's LoRA, then v's): on CUDA F.dropout is
`native_dropout` (PyTorch, aten/src/ATen/native/Dropout.cpp: `dropout` calls `native_dropout`
when the fused kernel is acceptable, i.e. a CUDA tensor and 0 < p < 1), which also returns the
mask; on the CPU F.dropout is `bernoulli_(1 - p)` on an empty tensor and `x * noise / (1 - p)`,
which `_dropout_with_mask` reproduces call for call. `dropout_self_check` verifies on the run's
own device, before training, that both give F.dropout's exact output, RNG advance and gradient,
also through the packed bits; if the mask itself differs the fast path keeps peft's own LoRA
branch (`ENABLED` turns False), if only the bits do, the bool mask is kept instead.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .. import _runtime
from .rmsnorm import _norm_fwd

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 - no Triton: the bool mask and torch.where are used
    triton = None
    tl = None

# Turned off by `dropout_self_check` if this device's dropout does not match ("enabled"), or if
# only the packed bits do not ("packed").
STATE = {"enabled": True, "checked": False, "packed": True}
MASK_WARPS = 4
SCALE_WARPS = 8


if triton is not None:

    @triton.jit
    def _dropout_mask_pack_kernel(
        mask_ptr,  # *u8 [R, H]: dropout's keep mask (a bool tensor viewed as bytes, 0 or 1)
        rows_ptr,  # *i64 [T]: row of the mask that holds token t (padded layout: b * S + s)
        bits_ptr,  # *u8 [T, H // 8]: bit j of byte k = keep[rows[t], 8 * k + j]
        n_cols,  # H, a multiple of 8
        BLOCK_BYTES: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        row = tl.load(rows_ptr + t)
        n_bytes = n_cols // 8
        k = tl.arange(0, BLOCK_BYTES)
        j = tl.arange(0, 8)
        live = k < n_bytes
        keep = tl.load(mask_ptr + row * n_cols + k[:, None] * 8 + j[None, :],
                       mask=live[:, None], other=0).to(tl.int32)
        packed = tl.sum((keep != 0).to(tl.int32) << j[None, :], axis=1)
        tl.store(bits_ptr + t * n_bytes + k, packed.to(tl.uint8), mask=live)

    @triton.jit(do_not_specialize=["has_dx"])
    def _dropout_bits_bwd_kernel(
        x_ptr,  # *f32 [T, H]: the input norm's output, recomputed
        dx16_ptr,  # *f16 [T, H]: gradient of lora_A's fp16 input (if has_dx)
        bits_ptr,  # *u8 [T, H // 8] from `_dropout_mask_pack_kernel`
        scale_ptr,  # *f32 [1]: 1 / (1 - p)
        x16_ptr,  # *f16 [T, H] out: fp16(where(keep, x * scale, 0)), lora_A's forward input
        dx_ptr,  # *f32 [T, H] out: where(keep, fp32(dx16) * scale, 0), dropout's backward
        n_cols,
        has_dx,
        BLOCK: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        live = cols < n_cols
        byte = tl.load(bits_ptr + t * (n_cols // 8) + cols // 8, mask=live, other=0).to(tl.int32)
        keep = ((byte >> (cols % 8)) & 1) != 0
        scale = tl.load(scale_ptr)
        x = tl.load(x_ptr + t * n_cols + cols, mask=live, other=0.0)
        tl.store(x16_ptr + t * n_cols + cols, tl.where(keep, x * scale, 0.0).to(tl.float16),
                 mask=live)
        if has_dx != 0:
            g = tl.load(dx16_ptr + t * n_cols + cols, mask=live, other=0.0).to(tl.float32)
            tl.store(dx_ptr + t * n_cols + cols, tl.where(keep, g * scale, 0.0), mask=live)


def _scale(p: float) -> torch.Tensor:
    """1 / (1 - p) as dropout's kernels apply it, in float32 (forward and backward: for this
    task's p = 0.05 the float division, the CUDA kernel's double division and the backward's
    double scale all round to the same float32, checked in `dropout_self_check`)."""
    return torch.tensor(1.0, dtype=torch.float32) / torch.tensor(1.0 - p, dtype=torch.float32)


_SCALES: dict = {}


def _scale_on(p: float, device: torch.device) -> torch.Tensor:
    """`_scale(p)` on `device`, copied there once: a host-to-device copy waits for the GPU to
    finish its queue, and the LoRA backward needs the scale 56 times a step (q and v, 28 layers)."""
    key = (float(p), device)
    scale = _SCALES.get(key)
    if scale is None:
        scale = _SCALES[key] = _scale(p).to(device)
    return scale


def _dropout_with_mask(x: torch.Tensor, p: float):
    """(F.dropout(x, p, training=True), keep mask), drawing the same random numbers."""
    if x.is_cuda:
        out, mask = torch.native_dropout(x, p, True)
        return out, mask
    noise = torch.empty_like(x).bernoulli_(1.0 - p)
    out = x * noise.div_(1.0 - p)
    return out, noise != 0


def _bits_ok(mask: torch.Tensor) -> bool:
    return (triton is not None and _runtime.PACKED_MASKS and STATE["packed"]
            and mask.dtype == torch.bool and mask.dim() == 2 and mask.shape[1] % 8 == 0
            and _runtime.device_ok(mask))


def pack_mask(mask: torch.Tensor, rows: Optional[torch.Tensor]) -> torch.Tensor:
    """The keep mask of the real tokens (`rows` of `mask`; all rows if None), as kept for
    backward: [T, H // 8] uint8 bits when the kernel can run, else the bool rows themselves."""
    if not _bits_ok(mask):
        return mask if rows is None else mask.index_select(0, rows)
    mask = mask.contiguous()
    n_cols = mask.shape[1]
    if rows is None:
        rows = torch.arange(mask.shape[0], device=mask.device)
    bits = torch.empty((rows.numel(), n_cols // 8), dtype=torch.uint8, device=mask.device)
    if rows.numel():
        _dropout_mask_pack_kernel[(rows.numel(),)](
            mask.view(torch.uint8), rows, bits, n_cols,
            BLOCK_BYTES=_runtime.next_pow2(n_cols // 8), num_warps=MASK_WARPS)
    return bits


def dropout_backward(x32: torch.Tensor, dx16: Optional[torch.Tensor], keep: torch.Tensor,
                     scale: torch.Tensor):
    """Dropout's masked scale for the LoRA backward, (fp16(where(keep, x32 * scale, 0)),
    where(keep, fp32(dx16) * scale, 0) or None): the rebuilt fp16 input of lora_A and the fp32
    gradient of x. `keep` is what `pack_mask` returned (bits: one kernel; bool: torch.where)."""
    if keep.dtype == torch.uint8:
        x32 = x32.contiguous()
        n_rows, n_cols = x32.shape
        x16 = torch.empty((n_rows, n_cols), dtype=torch.float16, device=x32.device)
        dx = None if dx16 is None else torch.empty_like(x32)
        if n_rows:
            _dropout_bits_bwd_kernel[(n_rows,)](
                x32, x16 if dx16 is None else dx16.contiguous(), keep, scale.reshape(1), x16,
                x32 if dx is None else dx, n_cols, int(dx16 is not None),
                BLOCK=_runtime.next_pow2(n_cols), num_warps=SCALE_WARPS)
        return x16, dx
    zero = torch.zeros((), dtype=torch.float32, device=x32.device)
    x16 = torch.where(keep, x32 * scale, zero).to(torch.float16)
    if dx16 is None:
        return x16, None
    return x16, torch.where(keep, dx16.to(torch.float32) * scale, zero)


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
                    rows = None
                else:  # base's layout: the padded [B * S, H] rows, then the real ones
                    dropped, mask = _dropout_with_mask(layout.padded(), p)
                    dropped = dropped.index_select(0, layout.dst)
                    rows = layout.dst
                keep = pack_mask(mask, rows)
                del mask
            else:
                dropped, keep = x32.detach(), None
            x16 = dropped.to(torch.float16)
            del dropped
            a16 = F.linear(x16, lora_a.to(torch.float16))
            out = F.linear(a16, lora_b.to(torch.float16)) * scaling
        del x16
        ctx.save_for_backward(resid, rstd, norm_weight, lora_a, lora_b, a16,
                              *(() if keep is None else (keep,)))
        ctx.eps, ctx.scaling, ctx.p = eps, scaling, p
        ctx.x_dtype = x32.dtype
        return out

    @staticmethod
    def backward(ctx, dout):
        saved = ctx.saved_tensors
        resid, rstd, norm_weight, lora_a, lora_b, a16 = saved[:6]
        keep = saved[6] if len(saved) > 6 else None
        a16_w, b16_w = lora_a.to(torch.float16), lora_b.to(torch.float16)
        db = dout.to(torch.float16) * ctx.scaling
        grad_b = db.t().mm(a16)  # F.linear's weight gradient: grad^T @ input
        da = db.mm(b16_w)
        del db, a16
        # The gradient of lora_A's fp16 input (autocast casts it back to x's dtype), and the
        # forward's x16, rebuilt: the same norm kernel on the same residual row, then the
        # dropout's own masked scale (with the mask: one pass for both, see dropout_backward).
        dx16 = da.mm(a16_w) if ctx.needs_input_grad[0] else None
        _, x32, _ = _norm_fwd(resid, None, norm_weight, ctx.eps)
        if keep is not None:
            x16, grad_x = dropout_backward(x32, dx16, keep, _scale_on(ctx.p, x32.device))
        else:
            x16, grad_x = x32.to(torch.float16), (None if dx16 is None
                                                  else dx16.to(torch.float32))
        del x32, dx16
        grad_a = da.t().mm(x16)
        del x16
        if grad_x is not None:
            grad_x = grad_x.to(ctx.x_dtype)
        return (grad_x, grad_a.to(lora_a.dtype), grad_b.to(lora_b.dtype), None, None, None,
                None, None, None, None)


def lora_remat(x32, lora_a, lora_b, resid, rstd, norm_weight, eps, scaling, p, layout):
    return LoraRematFunction.apply(x32, lora_a, lora_b, resid, rstd, norm_weight, float(eps),
                                   float(scaling), float(p), layout)


def enabled() -> bool:
    return _runtime.LORA_REMAT and STATE["enabled"]


def dropout_self_check(device: torch.device, p: float, n_cols: int = 4096) -> Optional[str]:
    """None if `_dropout_with_mask` + the backward formulas give F.dropout's exact output,
    gradient and RNG advance on `device`; else why not (and the feature turns off). The packed
    mask bits are checked the same way; if only they differ, the bool mask is kept instead.

    Runs in the main thread before training, inside torch.random.fork_rng, so the training run's
    RNG state is exactly what it was before the check. `n_cols` should be the model's hidden
    size, so the kernels it launches are the binaries training uses (compiled by the warm-up)."""
    STATE["checked"] = True
    try:
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            n = torch.arange(3 * n_cols, device=device, dtype=torch.float32)
            x = (torch.sin(n * 0.37) * 3.0).reshape(3, n_cols)
            torch.manual_seed(1234)
            xr = x.clone().requires_grad_(True)
            ref = F.dropout(xr, p, True)
            after_ref = torch.rand(8, device=device)
            g = torch.cos(n * 0.11).reshape(3, n_cols)
            ref.backward(g)
            torch.manual_seed(1234)
            out, mask = _dropout_with_mask(x, p)
            after_ours = torch.rand(8, device=device)
            scale = _scale_on(p, device)  # the tensor the LoRA backward will use
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
            if problem is None:
                _check_bits(mask, x, g, scale)
                _check_padded_layout(x, p)
    except Exception as exc:  # noqa: BLE001
        problem = f"check raised {exc!r}"
    if problem is not None:
        STATE["enabled"] = False
        _runtime.log(f"LoRA input rematerialization off: dropout check: {problem}")
    return problem


def _check_padded_layout(x, p) -> None:
    """Base draws its masks on [B, S, H], the fast path on the flat [B * S, H]: same keep mask?
    Only logged: every dropout of the fast path uses the flat layout, so this says whether the
    masks are base's, not whether rematerialization is exact (test_kernels.py asserts it)."""
    rows = x.repeat(2, 1)
    torch.manual_seed(77)
    ref = F.dropout(rows.reshape(2, x.shape[0], x.shape[1]), p, True)
    torch.manual_seed(77)
    flat, _ = _dropout_with_mask(rows, p)
    STATE["layout"] = torch.equal(flat, ref.reshape(rows.shape))
    if not STATE["layout"]:
        _runtime.log("LoRA dropout masks on [B * S, H] differ from base's [B, S, H]")


def _check_bits(mask, x, g, scale) -> None:
    """The packed path against the bool one, as training uses it: rows picked from a larger
    mask, fp32 -> fp16 for the rebuilt input, fp16 -> fp32 for the gradient. Bits off if not
    bit for bit equal."""
    if not _bits_ok(mask):
        return
    problem = None
    try:
        rows = torch.tensor([2, 0], device=mask.device)
        keep = pack_mask(mask, rows)
        picked = mask.index_select(0, rows)
        x2, g2 = x.index_select(0, rows), g.index_select(0, rows).to(torch.float16)
        want_x, want_g = dropout_backward(x2, g2, picked, scale)  # the torch.where path
        got_x, got_g = (None, None) if keep.dtype != torch.uint8 else \
            dropout_backward(x2, g2, keep, scale)
        if got_x is None:
            problem = "not packed"
        elif not torch.equal(got_x, want_x):
            problem = "rebuilt input differs"
        elif not torch.equal(got_g, want_g):
            problem = "input gradient differs"
    except Exception as exc:  # noqa: BLE001
        problem = f"raised {exc!r}"
    if problem is not None:
        STATE["packed"] = False
        _runtime.log(f"LoRA dropout masks stay bool (1 byte per element): bits check: {problem}")
