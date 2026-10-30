"""Fused residual add + RMSNorm (forward and backward), with base Soup's numerics.

In base Soup on a T4 the residual stream is fp32 (prepare_model_for_kbit_training keeps the
embedding and the norms in fp32), so every Qwen2RMSNorm is a chain of fp32 PyTorch kernels (pow,
mean, add, rsqrt, mul, mul) after a separate residual add, and twice that in backward. Here:

  _add_rmsnorm_fwd_kernel  one program per token row:
      resid_new = resid + delta          (fp32 + fp16 -> fp32: the previous block's residual add)
      y = weight * (resid_new * rstd),   rstd = 1 / sqrt(mean(resid_new^2) + eps), kept in fp32
  _add_rmsnorm_bwd_kernel  one program per row, closed form for the frozen-weight case:
      dresid = rstd * dn - x * rstd^3 * mean(dn * x) + dresid_new,   dn = dy * weight
      ddelta = fp16(dresid)              (the gradient base's add hands the fp16 block output)

The norm weight is frozen in LoRA training, so no weight gradient (and no cross-row reduction) is
needed. Only the fp32 row (which base saves too) and one fp32 rstd per row are kept for backward.
"""

from __future__ import annotations

from typing import Optional

import torch

from .. import _runtime

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 - no Triton: the torch path below is used
    triton = None
    tl = None

NORM_WARPS = 8


if triton is not None:

    @triton.jit(do_not_specialize=["has_delta"])
    def _add_rmsnorm_fwd_kernel(
        resid_ptr,  # *f32 [N, H]  residual stream before the add
        delta_ptr,  # *f16 [N, H]  block output to add (ignored if has_delta == 0)
        resid_out_ptr,  # *f32 [N, H] resid + delta (written if has_delta)
        y_ptr,  # *f32 [N, H]  normalized output
        weight_ptr,  # *f32 [H]
        rstd_ptr,  # *f32 [N]
        n_cols,
        eps,
        has_delta,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(resid_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        if has_delta != 0:
            d = tl.load(delta_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
            x = x + d
            tl.store(resid_out_ptr + row * n_cols + cols, x, mask=mask)
        var = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = w * (x * rstd)
        tl.store(y_ptr + row * n_cols + cols, y, mask=mask)
        tl.store(rstd_ptr + row, rstd)

    @triton.jit(do_not_specialize=["has_dres", "has_delta"])
    def _add_rmsnorm_bwd_kernel(
        dy_ptr,  # *f32 [N, H] gradient of the normalized output
        dres_ptr,  # *f32 [N, H] gradient of resid_out from the residual path (if has_dres)
        x_ptr,  # *f32 [N, H] the normalized input (resid_out), saved in forward
        weight_ptr,  # *f32 [H]
        rstd_ptr,  # *f32 [N]
        dx_ptr,  # *f32 [N, H] gradient of the residual input
        ddelta_ptr,  # *f16 [N, H] gradient of the added block output (if has_delta)
        n_cols,
        has_dres,
        has_delta,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        dy = tl.load(dy_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)
        dn = dy * w
        dot = tl.sum(dn * x, axis=0)
        dx = rstd * dn - x * (rstd * rstd * rstd * dot / n_cols)
        if has_dres != 0:
            dx = dx + tl.load(dres_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(dx_ptr + row * n_cols + cols, dx, mask=mask)
        if has_delta != 0:
            tl.store(ddelta_ptr + row * n_cols + cols, dx.to(tl.float16), mask=mask)


def _kernels_ok(*tensors: torch.Tensor) -> bool:
    return triton is not None and all(_runtime.device_ok(t) for t in tensors)


def _norm_fwd(resid: torch.Tensor, delta: Optional[torch.Tensor], weight: torch.Tensor,
              eps: float):
    n_rows, n_cols = resid.shape
    y = torch.empty_like(resid, dtype=torch.float32)
    rstd = torch.empty(n_rows, dtype=torch.float32, device=resid.device)
    resid_out = torch.empty_like(resid) if delta is not None else resid
    if _kernels_ok(resid):
        dummy = delta if delta is not None else resid.new_empty(1, dtype=torch.float16)
        if n_rows:
            _add_rmsnorm_fwd_kernel[(n_rows,)](
                resid, dummy, resid_out, y, weight, rstd, n_cols, eps, int(delta is not None),
                BLOCK=_runtime.next_pow2(n_cols), num_warps=NORM_WARPS)
        return resid_out, y, rstd
    x = resid + delta.to(torch.float32) if delta is not None else resid
    if delta is not None:
        resid_out.copy_(x)
    rstd.copy_(torch.rsqrt(x.pow(2).mean(-1) + eps))
    y.copy_(weight * (x * rstd[:, None]))
    return resid_out, y, rstd


def _norm_bwd(dy, dres, x, weight, rstd, with_delta: bool):
    n_rows, n_cols = x.shape
    dx = torch.empty_like(x)
    ddelta = torch.empty(x.shape, dtype=torch.float16, device=x.device) if with_delta else None
    dy = dy.contiguous().to(torch.float32)
    if dres is not None:
        dres = dres.contiguous().to(torch.float32)
    if _kernels_ok(x):
        if n_rows:
            _add_rmsnorm_bwd_kernel[(n_rows,)](
                dy, dres if dres is not None else dy, x, weight, rstd, dx,
                ddelta if ddelta is not None else x.new_empty(1, dtype=torch.float16),
                n_cols, int(dres is not None), int(with_delta),
                BLOCK=_runtime.next_pow2(n_cols), num_warps=NORM_WARPS)
        return dx, ddelta
    dn = dy * weight
    r = rstd[:, None]
    dot = (dn * x).sum(-1, keepdim=True)
    dx.copy_(r * dn - x * (r * r * r * dot / n_cols))
    if dres is not None:
        dx += dres
    if ddelta is not None:
        ddelta.copy_(dx.to(torch.float16))
    return dx, ddelta


class AddRMSNormFunction(torch.autograd.Function):
    """(resid + delta, RMSNorm(resid + delta)) in one pass. resid fp32, delta fp16."""

    @staticmethod
    def forward(ctx, resid, delta, weight, eps):
        resid_out, y, rstd = _norm_fwd(resid, delta, weight, eps)
        ctx.save_for_backward(resid_out, weight, rstd)
        return resid_out, y

    @staticmethod
    def backward(ctx, dres, dy):
        x, weight, rstd = ctx.saved_tensors
        if dy is None:
            dy = torch.zeros_like(x)
        dx, ddelta = _norm_bwd(dy, dres, x, weight, rstd, with_delta=True)
        return dx, ddelta, None, None


class RMSNormFunction(torch.autograd.Function):
    """RMSNorm of an fp32 residual stream, no add (the first layer's input norm)."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        _, y, rstd = _norm_fwd(x, None, weight, eps)
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        dx, _ = _norm_bwd(dy, None, x, weight, rstd, with_delta=False)
        return dx, None, None


def add_rmsnorm(resid: torch.Tensor, delta: Optional[torch.Tensor], weight: torch.Tensor,
                eps: float):
    """Returns (new residual, normalized). With delta None the residual is returned as is."""
    if delta is None:
        return resid, RMSNormFunction.apply(resid, weight, eps)
    return AddRMSNormFunction.apply(resid, delta, weight, eps)


def ref_add_rmsnorm(resid, delta, weight, eps):
    """Base Soup's math, for the tests: fp32 residual add, then Qwen2RMSNorm in fp32."""
    x = resid + delta.to(torch.float32) if delta is not None else resid
    variance = x.pow(2).mean(-1, keepdim=True)
    return x, weight * (x * torch.rsqrt(variance + eps))
