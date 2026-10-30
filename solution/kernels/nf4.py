"""Module 3: NF4 (bitsandbytes 4-bit, double-quantized) weights unpacked by one Triton kernel.

What base does for every 4-bit projection, forward and backward (bitsandbytes 0.50.2 picks this
path on a T4 for every layer once a batch has more than ~64 tokens):

    absmax8 -> fp32       kernel 1  (8-bit "nested" absmax, blockwise over 256 entries)
    absmax += offset      kernel 2
    W4 -> fp16 W          kernel 3  (one absmax per 64 weights)
    x @ W^T               cuBLAS

`_nf4_dequant_nested_kernel` does kernels 1-3 in one pass: each program reads a run of packed
bytes, decodes both nibbles through the 16-entry NF4 table, rebuilds the absmax of their block
from its 8-bit code in registers (code2[a8] * absmax2 + offset) and writes the fp16 weights in
order. It is launched 2 x 7 x 28 = 392 times a step (forward, and again in backward instead of
keeping the dense weight), so it is also where most of our own kernel time goes.

The result is bit-for-bit the weight bitsandbytes produces: the same table (taken from the
weight's own quant_state), the same fp32 products rounded once to fp16, and fp-fusion off so
`code * absmax2 + offset` is rounded twice, like the two bitsandbytes kernels. The GEMM after it
is the same F.linear call, so the projection outputs are unchanged.

Not done on purpose: merging LoRA into the dequantized weight (`W + s * B @ A`). The task trains
LoRA with dropout 0.05 on the LoRA input only, so `x @ (W + sBA)^T` is a different function in
training mode; the adapter would change. LoRA stays a separate branch (see patches/patch_qwen.py).

Idea credits: blockwise NF4 / double quantization from QLoRA (Dettmers et al. 2023) and the
bitsandbytes format; the kernel is our own.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import _runtime
from .swiglu import swiglu_bwd_inplace, swiglu_fwd

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None
    tl = None

BLOCK_BYTES = 1024  # packed bytes per program -> 2048 fp16 weights (4 KB) written
DEQUANT_WARPS = 4


if triton is not None:

    @triton.jit(do_not_specialize=["n_bytes"])
    def _nf4_dequant_nested_kernel(
        q_ptr,  # *u8 [n_bytes]: two 4-bit codes per byte, element 2i in the high nibble
        absmax8_ptr,  # *u8 [n_blocks]: 8-bit code of each block's absmax
        code2_ptr,  # *f32 [256]: code book of the nested (absmax) quantization
        absmax2_ptr,  # *f32 [ceil(n_blocks / QBLOCK2)]
        offset_ptr,  # *f32 [1]: mean subtracted from the absmax before it was quantized
        nf4_ptr,  # *f32 [16]: the NF4 code book of this weight
        out_ptr,  # *f16 [2 * n_bytes]
        n_bytes,
        BLOCK_BYTES: tl.constexpr,
        QBLOCK: tl.constexpr,  # weights per absmax (64)
        QBLOCK2: tl.constexpr,  # absmax values per nested absmax (256)
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
        live = offs < n_bytes
        packed = tl.load(q_ptr + offs, mask=live, other=0)
        block = offs // (QBLOCK // 2)  # weights 2i and 2i+1 share a block (QBLOCK is even)
        a8 = tl.load(absmax8_ptr + block, mask=live, other=0).to(tl.int32)
        absmax = tl.load(code2_ptr + a8) * tl.load(absmax2_ptr + block // QBLOCK2, mask=live,
                                                    other=0.0)
        absmax = absmax + tl.load(offset_ptr)
        hi = tl.load(nf4_ptr + (packed >> 4).to(tl.int32))
        lo = tl.load(nf4_ptr + (packed & 15).to(tl.int32))
        values = tl.interleave(hi * absmax, lo * absmax)  # element order: hi, lo, hi, lo, ...
        out_offs = pid * (2 * BLOCK_BYTES) + tl.arange(0, 2 * BLOCK_BYTES)
        tl.store(out_ptr + out_offs, values.to(tl.float16), mask=out_offs < 2 * n_bytes)


def _quant_state(module):
    weight = getattr(module, "weight", None)
    state = getattr(weight, "quant_state", None)
    return state if state is not None else getattr(module, "quant_state", None)


def is_linear4bit(module) -> bool:
    try:
        import bitsandbytes as bnb
    except Exception:  # noqa: BLE001
        return False
    return isinstance(module, bnb.nn.Linear4bit)


def _pow2(value) -> bool:
    value = int(value)
    return value >= 2 and value & (value - 1) == 0


def nf4_eligible(module) -> bool:
    """A bitsandbytes Linear4bit with NF4, double quantization and fp16 weights.

    Other 4-bit layers still take the fast path, through bitsandbytes' own forward."""
    if triton is None or not is_linear4bit(module):
        return False
    weight = module.weight
    state = _quant_state(module)
    if state is None or weight is None or not _runtime.device_ok(weight):
        return False
    state2 = getattr(state, "state2", None)
    return (state.quant_type == "nf4" and bool(getattr(state, "nested", False))
            and state2 is not None and _pow2(state2.blocksize) and _pow2(state.blocksize)
            and state.dtype == torch.float16
            and weight.dtype == torch.uint8 and state.absmax.dtype == torch.uint8
            and len(state.shape) == 2 and (state.shape[0] * state.shape[1]) % 2 == 0
            and not getattr(state, "packing_format_for_cpu", False))


def dequantize_nf4(module) -> torch.Tensor:
    """The fp16 [out, in] weight of an eligible Linear4bit (what bitsandbytes would give)."""
    state = _quant_state(module)
    rows, cols = int(state.shape[0]), int(state.shape[1])
    packed = module.weight.data.reshape(-1)
    out = torch.empty((rows, cols), dtype=torch.float16, device=packed.device)
    n_bytes = packed.numel()
    offset = state.offset
    if not torch.is_tensor(offset):
        offset = torch.tensor(float(offset), device=packed.device)
    offset = offset.to(torch.float32).reshape(1)
    _nf4_dequant_nested_kernel[(triton.cdiv(n_bytes, BLOCK_BYTES),)](
        packed, state.absmax, state.state2.code, state.state2.absmax, offset,
        state.code, out, n_bytes,
        BLOCK_BYTES=BLOCK_BYTES, QBLOCK=int(state.blocksize), QBLOCK2=int(state.state2.blocksize),
        num_warps=DEQUANT_WARPS, enable_fp_fusion=False)
    return out


def _bias16(module, dtype):
    bias = getattr(module, "bias", None)
    return None if bias is None else bias.detach().to(dtype)


class NF4LinearFunction(torch.autograd.Function):
    """x16 @ dequant(W)^T + b, the frozen 4-bit weight unpacked again in backward."""

    @staticmethod
    def forward(ctx, x16, module):
        with torch.autocast(device_type=x16.device.type, enabled=False):
            weight = dequantize_nf4(module)
            out = F.linear(x16, weight, _bias16(module, x16.dtype))
        ctx.module = module
        return out

    @staticmethod
    def backward(ctx, grad):
        weight = dequantize_nf4(ctx.module)
        return torch.matmul(grad, weight.to(grad.dtype)), None


def nf4_linear(module, x16: torch.Tensor) -> torch.Tensor:
    return NF4LinearFunction.apply(x16, module)


def _nf4_linear_nograd(module, x16):
    return F.linear(x16, dequantize_nf4(module), _bias16(module, x16.dtype))


class GateUpSwiGLUFunction(torch.autograd.Function):
    """silu(gate(x)) * up(x) for a Qwen2 MLP with 4-bit gate/up: fp32 x in, fp16 out.

    Same values as base (gate/up see fp16(x) and return fp16, silu * up in fp32, rounded once to
    the fp16 that down_proj reads). Saves fp16 gate and up (or, with recompute, only fp16 x and
    runs the two 4-bit GEMMs again in backward). The two input gradients are added in fp32, as
    autograd adds the two fp16 -> fp32 cast gradients in base.
    """

    @staticmethod
    def forward(ctx, x32, gate, up, recompute):
        with torch.autocast(device_type=x32.device.type, enabled=False):
            x16 = x32.to(torch.float16)
            g = _nf4_linear_nograd(gate, x16)
            u = _nf4_linear_nograd(up, x16)
            p = swiglu_fwd(g, u)
        if recompute:
            del g, u
            ctx.save_for_backward(x16)
        else:
            ctx.save_for_backward(g, u)
        ctx.gate, ctx.up, ctx.recompute = gate, up, recompute
        ctx.x_dtype = x32.dtype
        return p

    @staticmethod
    def backward(ctx, dp):
        if ctx.recompute:
            (x16,) = ctx.saved_tensors
            g = _nf4_linear_nograd(ctx.gate, x16)
            u = _nf4_linear_nograd(ctx.up, x16)
        else:
            g, u = ctx.saved_tensors
        dg, du = swiglu_bwd_inplace(dp.to(torch.float16), g, u)
        dx = torch.matmul(dg, dequantize_nf4(ctx.gate)).to(torch.float32)
        dx += torch.matmul(du, dequantize_nf4(ctx.up)).to(torch.float32)
        return dx.to(ctx.x_dtype), None, None, None


def gate_up_swiglu(x32, gate, up, recompute: bool = False):
    return GateUpSwiGLUFunction.apply(x32, gate, up, bool(recompute))


def reference_dequantize(module) -> torch.Tensor:
    """bitsandbytes' own dequantization, for the tests."""
    import bitsandbytes.functional as bnbf

    return bnbf.dequantize_4bit(module.weight.data, _quant_state(module)).to(torch.float16)
