"""Fused LoRA add + rotary position embedding (RoPE) + scatter to SDPA's padded layout.

Base Soup computes q (and v) as peft's `result + lora_B(lora_A(dropout(x))) * scaling` in fp32,
then transformers' `q * cos + rotate_half(q) * sin` as a chain of fp32 kernels (mul, slice, neg,
cat, mul, add), and autocast casts the result to fp16 for SDPA. Here, one program per
(unpadded) token:

  _rope_lora_scatter_fwd_kernel
      x = fp32(base16) + fp32(lora16)               the LoRA add, in fp32 as in base
      out = fp16(x * cos + rotate_half(x) * sin)    (or fp16(x) for v: no rotation)
      written straight to [B, heads, S, D] fp16 at the token's padded position: the tensor SDPA
      reads, rounded once from fp32 exactly where autocast rounds it in base, so no fp32 copy and
      no cast kernel; dropping pad tokens needs no separate scatter; one launch covers all heads
      of a token (GQA: q with 12 heads and k/v with 2 use the same compiled kernel).
  _rope_gather_bwd_kernel
      reads SDPA's fp16 gradient (base upcasts that same fp16 gradient to fp32 before the RoPE
      backward), applies the transpose of the rotation, i.e. the conjugate (-sin), in fp32, and
      gathers it back to the unpadded rows; the same fp16 gradient goes to the base projection and
      to the LoRA branch (both were added in fp32). No RoPE activation is stored at all.

cos/sin are the tables transformers' rotary embedding builds for the batch (positions per row),
so any position_ids work. FMA contraction is off so each product is rounded as in base.
"""

from __future__ import annotations

import torch

from .. import _runtime

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None
    tl = None

ROPE_WARPS = 4


if triton is not None:

    @triton.jit(do_not_specialize=["S", "n_heads", "has_lora", "rotate"])
    def _rope_lora_scatter_fwd_kernel(
        base_ptr,  # *f16 [T, n_heads * D] base projection (bias included)
        lora_ptr,  # *f16 [T, n_heads * D] LoRA branch output (ignored if has_lora == 0)
        cos_ptr,  # *f32 [rows, D]
        sin_ptr,  # *f32 [rows, D]
        cos_row_ptr,  # *i64 [T] row of cos/sin for token t
        dst_ptr,  # *i64 [T] flat index b * S + s of token t in the padded batch
        out_ptr,  # *f16 [B, n_heads, S, D]: SDPA's input
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
            # q * cos + rotate_half(q) * sin, rotate_half(q) = cat(-q2, q1); rounded to fp16 once
            tl.store(out, (x1 * c1 - x2 * s1).to(tl.float16), mask=mask)
            tl.store(out + HALF, (x2 * c2 + x1 * s2).to(tl.float16), mask=mask)
        else:
            tl.store(out, x1.to(tl.float16), mask=mask)
            tl.store(out + HALF, x2.to(tl.float16), mask=mask)

    @triton.jit(do_not_specialize=["S", "n_heads", "rotate"])
    def _rope_gather_bwd_kernel(
        gout_ptr,  # *f16 [B, n_heads, S, D] SDPA's gradient of the padded rotated output
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
        g1 = tl.load(gptr, mask=mask, other=0.0).to(tl.float32)
        g2 = tl.load(gptr + HALF, mask=mask, other=0.0).to(tl.float32)
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


def _kernels_ok(*tensors: torch.Tensor) -> bool:
    return triton is not None and all(_runtime.device_ok(t) for t in tensors)


class RopeLoraScatterFunction(torch.autograd.Function):
    """fp32(base16) + fp32(lora16), rotated (or not), rounded to fp16 and written to padded
    [B, heads, S, D] (zeros at pad positions)."""

    @staticmethod
    def forward(ctx, base, lora, cos, sin, cos_row, dst, batch, seq, n_heads, rotate,
                block_heads):
        n_tok = base.shape[0]
        head_dim = cos.shape[-1]
        cos2 = cos.reshape(-1, head_dim).contiguous()
        sin2 = sin.reshape(-1, head_dim).contiguous()
        full = n_tok == batch * seq
        out = (torch.empty if full else torch.zeros)(
            (batch, n_heads, seq, head_dim), dtype=torch.float16, device=base.device)
        # One BLOCK_HEADS for q and k/v (the larger head count): one compiled kernel, not two.
        block_heads = max(int(block_heads or 0), _runtime.next_pow2(n_heads))
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
            # With one head (or one batch) reshape can alias out. Copying a permuted
            # view back into that storage is rejected by PyTorch's overlap check.
            flat = out.permute(0, 2, 1, 3).reshape(batch * seq, n_heads, head_dim).clone()
            flat.index_copy_(0, dst, x.to(torch.float16))
            out.copy_(flat.view(batch, seq, n_heads, head_dim).permute(0, 2, 1, 3))
        ctx.save_for_backward(cos2, sin2, cos_row, dst)
        ctx.meta = (n_tok, seq, n_heads, head_dim, bool(rotate), lora is not None, block_heads)
        return out

    @staticmethod
    def backward(ctx, gout):
        cos2, sin2, cos_row, dst = ctx.saved_tensors
        n_tok, seq, n_heads, head_dim, rotate, has_lora, block_heads = ctx.meta
        gin = torch.empty((n_tok, n_heads * head_dim), dtype=torch.float16, device=gout.device)
        # SDPA hands back fp16 (what base upcasts before its fp32 RoPE backward). A gradient in
        # any other dtype is rounded to fp16 here, like the fp16 output it belongs to.
        gout = gout.to(torch.float16).contiguous()
        if _kernels_ok(gout) and n_tok:
            _rope_gather_bwd_kernel[(n_tok,)](
                gout, cos2, sin2, cos_row, dst, gin, seq, n_heads, int(rotate),
                HEAD_DIM=head_dim, BLOCK_HEADS=block_heads, num_warps=ROPE_WARPS,
                enable_fp_fusion=False)
        elif n_tok:
            g = gout.float().permute(0, 2, 1, 3).reshape(-1, n_heads, head_dim)
            g = g.index_select(0, dst)
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


def ref_rope(x32, cos, sin):
    """transformers' apply_rotary_pos_emb for one tensor: x * cos + rotate_half(x) * sin."""
    half = x32.shape[-1] // 2
    rot = torch.cat((-x32[..., half:], x32[..., :half]), dim=-1)
    return x32 * cos + rot * sin
