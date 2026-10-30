"""SwiGLU `silu(gate) * up` in fp16 storage with fp32 math, backward written in place.

In base Soup gate_proj / up_proj return fp32 (bitsandbytes returns its input's dtype and the
post-attention norm output is fp32), so silu and the product run in fp32 and backward keeps
gate, up and silu(gate) in fp32: three [tokens, 8960] fp32 tensors per layer. Here gate and up
stay the fp16 GEMM outputs (the same values base casts up), silu * up is computed in fp32 and
rounded once to the fp16 tensor base's down_proj reads anyway:

  _swiglu_fp16_fwd_kernel           p = fp16(silu(g) * u), torch's silu formula g / (1 + e^-g)
  _swiglu_fp16_bwd_inplace_kernel   silu(g) recomputed; dg, du written over g and u

Only g and u are saved (fp16): -147 MB per layer at 2048 tokens against base.

This is a SwiGLU like the starter kit's example; we do not count it as our own kernel.
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

SWIGLU_BLOCK = 1024
SWIGLU_WARPS = 4


if triton is not None:

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


def _kernels_ok(*tensors: torch.Tensor) -> bool:
    return triton is not None and all(_runtime.device_ok(t) for t in tensors)


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
