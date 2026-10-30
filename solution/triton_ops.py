"""Module 2: the non-matmul parts of a Qwen2 decoder layer, fused, with base Soup's numerics.

Why these ops. In base Soup on a T4 the residual stream is fp32 (prepare_model_for_kbit_training
keeps the embedding and the norms in fp32) and bitsandbytes returns every projection in its input
dtype, so RMSNorm, RoPE, the LoRA add and SwiGLU all run as chains of fp32 PyTorch kernels over
[tokens, 1536] or [tokens, 8960] tensors: about 44% of the GPU time of a step. Each kernel below
reads its inputs once and writes its outputs once, and keeps the base math (fp32 arithmetic, the
same rounding points to fp16), so the loss curve does not move.

Kernels (all fp32 math, fp16 or fp32 storage exactly where base has it):

  _add_rmsnorm_fwd_kernel / _add_rmsnorm_bwd_kernel
      resid_new = resid + delta (fp32 + fp16 -> fp32, the residual add of the previous block)
      y = weight * (resid_new * rsqrt(mean(resid_new^2) + eps)), rstd kept in fp32 for backward.
      Backward: dresid = rstd * (dn - x * rstd^2 * mean(dn * x)) + dresid_new, ddelta = fp16(.).
  _rope_lora_scatter_fwd_kernel / _rope_gather_bwd_kernel
      q = fp32(base16) + fp32(lora16)           (peft: result + lora_B(lora_A(dropout(x)))*s)
      q_rot = q * cos + rotate_half(q) * sin    (one program per token, all heads)
      and the result is written straight into the padded [B, heads, S, D] layout SDPA takes, so
      unpadded tokens need no separate scatter. Backward applies the inverse rotation (the
      conjugate, -sin) to the incoming gradient and gathers it back to the unpadded rows: no
      activation of RoPE is stored at all.
  _swiglu_fp16_fwd_kernel / _swiglu_fp16_bwd_inplace_kernel
      p = silu(g) * u from fp16 g, u (the gate/up GEMM outputs) to fp16 p, which is exactly the
      fp16 tensor base's down_proj consumes. Only g and u are saved (fp16, not the fp32 copies
      plus silu(g) of base: -147 MB per layer at 2048 tokens) and silu(g) is recomputed in
      backward; dg and du are written over g and u in place, so backward allocates nothing.
      (The starter kit's example is a SwiGLU too; this one is a different, fp16 in-place
      variant, and we do not count it as our own kernel.)

All wrappers fall back to plain PyTorch when Triton or a capable GPU is missing.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import _runtime

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None
    tl = None


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


if triton is not None:

    # ------------------------------------------------------------------ residual add + RMSNorm

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

    # ----------------------------------------------- LoRA add + RoPE + scatter to padded layout

    @triton.jit(do_not_specialize=["S", "n_heads", "has_lora", "rotate"])
    def _rope_lora_scatter_fwd_kernel(
        base_ptr,  # *f16 [T, n_heads * D] base projection (bias included)
        lora_ptr,  # *f16 [T, n_heads * D] LoRA branch output (ignored if has_lora == 0)
        cos_ptr,  # *f32 [rows, D]
        sin_ptr,  # *f32 [rows, D]
        cos_row_ptr,  # *i64 [T] row of cos/sin for token t
        dst_ptr,  # *i64 [T] flat index b * S + s of token t in the padded batch
        out_ptr,  # *f32 [B, n_heads, S, D]
        S,
        n_heads,
        has_lora,
        rotate,
        HEAD_DIM: tl.constexpr,
        BLOCK_HEADS: tl.constexpr,
    ):
        HALF: tl.constexpr = HEAD_DIM // 2
        t = tl.program_id(0).to(tl.int64)
        dst = tl.load(dst_ptr + t)
        b = dst // S
        s = dst - b * S
        crow = tl.load(cos_row_ptr + t)
        heads = tl.arange(0, BLOCK_HEADS)[:, None]
        j = tl.arange(0, HALF)[None, :]
        mask = (heads < n_heads) & (j < HALF)
        src = t * n_heads * HEAD_DIM + heads * HEAD_DIM + j
        x1 = tl.load(base_ptr + src, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(base_ptr + src + HALF, mask=mask, other=0.0).to(tl.float32)
        if has_lora != 0:
            x1 = x1 + tl.load(lora_ptr + src, mask=mask, other=0.0).to(tl.float32)
            x2 = x2 + tl.load(lora_ptr + src + HALF, mask=mask, other=0.0).to(tl.float32)
        out = out_ptr + ((b * n_heads + heads) * S + s) * HEAD_DIM + j
        if rotate != 0:
            c1 = tl.load(cos_ptr + crow * HEAD_DIM + j)
            c2 = tl.load(cos_ptr + crow * HEAD_DIM + HALF + j)
            s1 = tl.load(sin_ptr + crow * HEAD_DIM + j)
            s2 = tl.load(sin_ptr + crow * HEAD_DIM + HALF + j)
            # q * cos + rotate_half(q) * sin, rotate_half(q) = cat(-q2, q1)
            tl.store(out, x1 * c1 - x2 * s1, mask=mask)
            tl.store(out + HALF, x2 * c2 + x1 * s2, mask=mask)
        else:
            tl.store(out, x1, mask=mask)
            tl.store(out + HALF, x2, mask=mask)

    @triton.jit(do_not_specialize=["S", "n_heads", "rotate"])
    def _rope_gather_bwd_kernel(
        gout_ptr,  # *f32 [B, n_heads, S, D] gradient of the padded rotated output
        cos_ptr,
        sin_ptr,
        cos_row_ptr,
        dst_ptr,
        gin_ptr,  # *f16 [T, n_heads * D] gradient of base (and of the LoRA branch)
        S,
        n_heads,
        rotate,
        HEAD_DIM: tl.constexpr,
        BLOCK_HEADS: tl.constexpr,
    ):
        HALF: tl.constexpr = HEAD_DIM // 2
        t = tl.program_id(0).to(tl.int64)
        dst = tl.load(dst_ptr + t)
        b = dst // S
        s = dst - b * S
        crow = tl.load(cos_row_ptr + t)
        heads = tl.arange(0, BLOCK_HEADS)[:, None]
        j = tl.arange(0, HALF)[None, :]
        mask = (heads < n_heads) & (j < HALF)
        gptr = gout_ptr + ((b * n_heads + heads) * S + s) * HEAD_DIM + j
        g1 = tl.load(gptr, mask=mask, other=0.0)
        g2 = tl.load(gptr + HALF, mask=mask, other=0.0)
        dst_ptrs = gin_ptr + t * n_heads * HEAD_DIM + heads * HEAD_DIM + j
        if rotate != 0:
            c1 = tl.load(cos_ptr + crow * HEAD_DIM + j)
            c2 = tl.load(cos_ptr + crow * HEAD_DIM + HALF + j)
            s1 = tl.load(sin_ptr + crow * HEAD_DIM + j)
            s2 = tl.load(sin_ptr + crow * HEAD_DIM + HALF + j)
            # the transpose of the rotation: the conjugate (-sin) applied to the gradient
            tl.store(dst_ptrs, (g1 * c1 + g2 * s2).to(tl.float16), mask=mask)
            tl.store(dst_ptrs + HALF, (g2 * c2 - g1 * s1).to(tl.float16), mask=mask)
        else:
            tl.store(dst_ptrs, g1.to(tl.float16), mask=mask)
            tl.store(dst_ptrs + HALF, g2.to(tl.float16), mask=mask)

    # ------------------------------------------------------------------ SwiGLU, fp16 in place

    @triton.jit(do_not_specialize=["n"])
    def _swiglu_fp16_fwd_kernel(g_ptr, u_ptr, p_ptr, n, BLOCK: tl.constexpr):
        idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n
        g = tl.load(g_ptr + idx, mask=live, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + idx, mask=live, other=0.0).to(tl.float32)
        act = g / (1.0 + tl.exp(-g))  # torch's silu formula
        tl.store(p_ptr + idx, (act * u).to(tl.float16), mask=live)

    @triton.jit(do_not_specialize=["n"])
    def _swiglu_fp16_bwd_inplace_kernel(dp_ptr, g_ptr, u_ptr, n, BLOCK: tl.constexpr):
        idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n
        dp = tl.load(dp_ptr + idx, mask=live, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + idx, mask=live, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + idx, mask=live, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g))
        act = g / (1.0 + tl.exp(-g))
        da = dp * u
        dg = da * sig * (1.0 + g * (1.0 - sig))  # torch's silu_backward formula
        du = dp * act
        tl.store(g_ptr + idx, dg.to(tl.float16), mask=live)
        tl.store(u_ptr + idx, du.to(tl.float16), mask=live)


NORM_WARPS = 8
ROPE_WARPS = 4
SWIGLU_BLOCK = 1024
SWIGLU_WARPS = 4


def _kernels_ok(*tensors: torch.Tensor) -> bool:
    return triton is not None and all(_runtime.device_ok(t) for t in tensors)


# ---------------------------------------------------------------------------- residual + norm


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
                BLOCK=_next_pow2(n_cols), num_warps=NORM_WARPS)
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
                BLOCK=_next_pow2(n_cols), num_warps=NORM_WARPS)
        return dx, ddelta
    dn = dy * weight
    r = rstd[:, None]
    dot = (dn * x).sum(-1, keepdim=True)
    dx.copy_(r * dn - x * (r * r * r * dot / n_cols))
    if dres is not None:
        dx += dres
    if with_delta:
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


# ------------------------------------------------------------------------------------ RoPE


class RopeLoraScatterFunction(torch.autograd.Function):
    """fp32(base16) + fp32(lora16), rotated (or not), written to padded [B, heads, S, D] fp32."""

    @staticmethod
    def forward(ctx, base, lora, cos, sin, cos_row, dst, batch, seq, n_heads, rotate,
                block_heads):
        n_tok = base.shape[0]
        head_dim = cos.shape[-1]
        cos2 = cos.reshape(-1, head_dim).contiguous()
        sin2 = sin.reshape(-1, head_dim).contiguous()
        full = n_tok == batch * seq
        out = (torch.empty if full else torch.zeros)(
            (batch, n_heads, seq, head_dim), dtype=torch.float32, device=base.device)
        # One BLOCK_HEADS for q and k/v (the larger head count): one compiled kernel, not two.
        block_heads = max(int(block_heads or 0), _next_pow2(n_heads))
        if _kernels_ok(base) and n_tok:
            _rope_lora_scatter_fwd_kernel[(n_tok,)](
                base, lora if lora is not None else base, cos2, sin2, cos_row, dst, out,
                seq, n_heads, int(lora is not None), int(rotate),
                HEAD_DIM=head_dim, BLOCK_HEADS=block_heads, num_warps=ROPE_WARPS,
                enable_fp_fusion=False)
        elif n_tok:
            x = base.float() + (lora.float() if lora is not None else 0.0)
            x = x.view(n_tok, n_heads, head_dim)
            if rotate:
                c = cos2.index_select(0, cos_row)[:, None, :]
                s = sin2.index_select(0, cos_row)[:, None, :]
                half = head_dim // 2
                rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
                x = x * c + rot * s
            flat = out.permute(0, 2, 1, 3).reshape(batch * seq, n_heads, head_dim)
            flat.index_copy_(0, dst, x)
            out.copy_(flat.view(batch, seq, n_heads, head_dim).permute(0, 2, 1, 3))
        ctx.save_for_backward(cos2, sin2, cos_row, dst)
        ctx.meta = (n_tok, seq, n_heads, head_dim, bool(rotate), lora is not None, block_heads)
        return out

    @staticmethod
    def backward(ctx, gout):
        cos2, sin2, cos_row, dst = ctx.saved_tensors
        n_tok, seq, n_heads, head_dim, rotate, has_lora, block_heads = ctx.meta
        gin = torch.empty((n_tok, n_heads * head_dim), dtype=torch.float16, device=gout.device)
        gout = gout.contiguous().to(torch.float32)
        if _kernels_ok(gout) and n_tok:
            _rope_gather_bwd_kernel[(n_tok,)](
                gout, cos2, sin2, cos_row, dst, gin, seq, n_heads, int(rotate),
                HEAD_DIM=head_dim, BLOCK_HEADS=block_heads, num_warps=ROPE_WARPS,
                enable_fp_fusion=False)
        elif n_tok:
            g = gout.permute(0, 2, 1, 3).reshape(-1, n_heads, head_dim).index_select(0, dst)
            if rotate:
                c = cos2.index_select(0, cos_row)[:, None, :]
                s = sin2.index_select(0, cos_row)[:, None, :]
                half = head_dim // 2
                gs = g * s
                g = g * c + torch.cat((gs[..., half:], -gs[..., :half]), dim=-1)
            gin.copy_(g.reshape(n_tok, -1).to(torch.float16))
        return (gin, (gin if has_lora else None), None, None, None, None, None, None, None, None,
                None)


def rope_lora_scatter(base16, lora16, cos, sin, cos_row, dst, batch, seq, n_heads, rotate=True,
                      block_heads=None):
    return RopeLoraScatterFunction.apply(base16, lora16, cos, sin, cos_row, dst, batch, seq,
                                         n_heads, rotate, block_heads)


# ---------------------------------------------------------------------------------- SwiGLU


def swiglu_fwd(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    p = torch.empty_like(g)
    n = g.numel()
    if _kernels_ok(g) and n:
        _swiglu_fp16_fwd_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](
            g, u, p, n, BLOCK=SWIGLU_BLOCK, num_warps=SWIGLU_WARPS)
    elif n:
        gf = g.float()
        p.copy_((gf / (1.0 + torch.exp(-gf)) * u.float()).to(p.dtype))
    return p


def swiglu_bwd_inplace(dp: torch.Tensor, g: torch.Tensor, u: torch.Tensor):
    """Overwrites g with dL/dg and u with dL/du (fp16), returns them."""
    dp = dp.contiguous()
    n = g.numel()
    if _kernels_ok(g) and n:
        _swiglu_fp16_bwd_inplace_kernel[(triton.cdiv(n, SWIGLU_BLOCK),)](
            dp, g, u, n, BLOCK=SWIGLU_BLOCK, num_warps=SWIGLU_WARPS)
    elif n:
        gf, uf, dpf = g.float(), u.float(), dp.float()
        sig = torch.sigmoid(gf)
        act = gf / (1.0 + torch.exp(-gf))
        g.copy_((dpf * uf * sig * (1.0 + gf * (1.0 - sig))).to(g.dtype))
        u.copy_((dpf * act).to(u.dtype))
    return g, u


class SwiGLUFunction(torch.autograd.Function):
    """silu(g) * u for fp16 g, u; saves only g and u.

    With inplace=True backward writes dg, du over g, u: only for g, u that nothing else reads
    (the GEMM outputs inside the MLP). Otherwise it works on copies.
    """

    @staticmethod
    def forward(ctx, g, u, inplace):
        g = g.contiguous()
        u = u.contiguous()
        ctx.save_for_backward(g, u)
        ctx.inplace = bool(inplace)
        return swiglu_fwd(g, u)

    @staticmethod
    def backward(ctx, dp):
        g, u = ctx.saved_tensors
        if not ctx.inplace:
            g, u = g.clone(), u.clone()
        dg, du = swiglu_bwd_inplace(dp, g, u)
        return dg, du, None


def swiglu(g: torch.Tensor, u: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return SwiGLUFunction.apply(g, u, inplace)


# ------------------------------------------------------------------- references for the tests


def ref_add_rmsnorm(resid, delta, weight, eps):
    """Base Soup's math: fp32 residual add, then Qwen2RMSNorm in fp32."""
    x = resid + delta.to(torch.float32) if delta is not None else resid
    variance = x.pow(2).mean(-1, keepdim=True)
    return x, weight * (x * torch.rsqrt(variance + eps))


def ref_rope(x32, cos, sin):
    """transformers' apply_rotary_pos_emb for one tensor: x * cos + rotate_half(x) * sin."""
    half = x32.shape[-1] // 2
    rot = torch.cat((-x32[..., half:], x32[..., :half]), dim=-1)
    return x32 * cos + rot * sin
