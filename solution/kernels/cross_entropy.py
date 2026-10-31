"""Module 1: the lm_head and the causal-LM loss in token chunks, without [B, S, V] logits.

Base Soup on this task builds the logits of a whole batch, [4, <=512, 151936] in fp16, and then
keeps several more tensors of that shape alive: ForCausalLMLoss upcasts them to fp32 and saves
log_softmax for backward, accelerate converts the model output to fp32, TRL makes a contiguous
copy for its token accuracy, and backward allocates two more fp32 gradients. At 2048 tokens each
of those is 1.24 GB; together they are most of the base run's peak.

Here the head runs `chunk` rows at a time (rows = real, unpadded tokens):

    logits_chunk = h16[rows] @ W16^T                       cuBLAS, [chunk, V] fp16, as in base
    _cross_entropy_fwd_bwd_inplace_kernel (Triton)         one program per row:
        pass 1  online max / sum-exp / entropy / argmax over V in fp32
        pass 2  dL/dz = (softmax - onehot) * coef, written in place over the logits
    grad_h16[rows] = dL/dz @ W16                          cuBLAS, the gradient lm_head's
                                                          backward would compute

so only one [chunk, V] fp16 buffer exists at a time (78 MB at chunk=256). The kernel also returns
what the trainer logs from the full logits in base: per-token entropy (TRL `entropy`), argmax hits
(TRL `mean_token_accuracy`) and per-token cross-entropy (Soup's rewind.jsonl row losses).

Numerics. Every per-row term is computed in fp32 from the same fp16 logits base computes, and the
gradient is rounded to fp16 at the same point and the same scale as base: base backpropagates
`GradScaler.scale * loss`, so the fp16 dL/dz it rounds is `(p - y) * scale / num_items`. The
gradient here is taken in forward, so the trainer hands in the scale it is about to use (see
patches/patch_qwen.py); backward multiplies by `grad_output / scale`, which is exactly 1.0 in training.
Rows without a label get zero gradient, so the second GEMM runs over labelled rows only: they are
sorted first.

Idea credits: online softmax (Milakov & Gimelshein 2018, "Online normalizer calculation for
softmax"); chunked lm_head + loss as in Soup's own `stream_loss_chunk` (#1693) and in Cut
Cross-Entropy (Wijmans et al. 2024). The kernel and the wrapper are written for this task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from .. import _runtime

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 - no Triton: the torch path below is used
    triton = None
    tl = None

IGNORE_INDEX = -100
# Vocabulary tile per loop iteration: 2048 fp16 logits = 4 KB per row and iteration; with 8 warps
# that is 8 values per thread for each of the 5 per-lane running vectors. The T4 binary uses
# 32 bytes of shared memory (the final reductions), no scratch.
BLOCK_V = 2048
NUM_WARPS = 8


if triton is not None:

    @triton.jit(do_not_specialize=["row_start", "write_grad"])
    def _cross_entropy_fwd_bwd_inplace_kernel(
        logits_ptr,  # *fp16 [n, V]: this chunk's logits in, dL/dz out (when write_grad)
        stride_row,  # elements between two rows of logits
        targets_ptr,  # *i64 [N]: next-token label of every row, IGNORE_INDEX for none
        loss_ptr,  # *f32 [N]: cross-entropy of the label, 0 on unlabelled rows
        entropy_ptr,  # *f32 [N]: entropy of the row's softmax (nats)
        hit_ptr,  # *i32 [N]: 1 where argmax == label on a labelled row
        coef_ptr,  # *f32 [1]: d(scaled loss) / d(row cross-entropy) = grad_scale / num_items
        row_start,  # index of this chunk's first row in the [N] arrays
        V,  # vocabulary size
        write_grad,  # 0: metrics only; 1: also overwrite the logits with dL/dz
        BLOCK_V: tl.constexpr,
    ):
        NEG: tl.constexpr = -1.0e30
        pid = tl.program_id(0)
        row = row_start + pid
        logits_row = logits_ptr + pid.to(tl.int64) * stride_row
        target = tl.load(targets_ptr + row).to(tl.int32)
        labelled = target != -100

        # Pass 1, per lane: running max m, s = sum exp(z - m), t = sum exp(z - m) * (z - m) (for
        # the entropy) and the first column holding the lane's largest logit. Lanes are merged
        # after the loop, so no scalar is carried through it.
        lanes = tl.arange(0, BLOCK_V)
        m = tl.full([BLOCK_V], NEG, tl.float32)
        s = tl.zeros([BLOCK_V], dtype=tl.float32)
        t = tl.zeros([BLOCK_V], dtype=tl.float32)
        best = tl.full([BLOCK_V], NEG, tl.float32)
        best_col = tl.zeros([BLOCK_V], dtype=tl.int32)
        for start in range(0, V, BLOCK_V):
            cols = start + lanes
            valid = cols < V
            z = tl.load(logits_row + cols, mask=valid, other=0.0).to(tl.float32)
            z = tl.where(valid, z, NEG)
            better = z > best  # strict: a lane keeps the first column of its maximum
            best_col = tl.where(better, cols, best_col)
            best = tl.where(better, z, best)
            m_new = tl.maximum(m, z)
            alpha = tl.exp(m - m_new)
            e = tl.where(valid, tl.exp(z - m_new), 0.0)
            t = alpha * (t + (m - m_new) * s) + e * (z - m_new)
            s = alpha * s + e
            m = m_new

        m_row = tl.max(m, axis=0)
        w = tl.exp(m - m_row)  # 0 for lanes that never saw a column
        s_row = tl.sum(s * w, axis=0)
        t_row = tl.sum(w * (t + (m - m_row) * s), axis=0)
        log_s = tl.log(s_row)
        lse = m_row + log_s
        entropy = log_s - t_row / s_row  # -sum p log p, shifted by the row max for accuracy

        top = tl.max(best, axis=0)
        argmax = tl.min(tl.where(best == top, best_col, 2147483647), axis=0)  # first maximum
        z_target = tl.load(logits_row + tl.where(labelled, target, 0)).to(tl.float32)
        tl.store(loss_ptr + row, tl.where(labelled, lse - z_target, 0.0))
        tl.store(entropy_ptr + row, entropy)
        tl.store(hit_ptr + row, (labelled & (argmax == target)).to(tl.int32))

        # Pass 2: dL/dz = (softmax - onehot) * coef, in place. Each element is read before it
        # is written by the same program, so overwriting the logits is safe.
        if write_grad != 0:
            coef = tl.where(labelled, tl.load(coef_ptr), 0.0)
            for start in range(0, V, BLOCK_V):
                cols = start + lanes
                valid = cols < V
                z = tl.load(logits_row + cols, mask=valid, other=0.0).to(tl.float32)
                p = tl.exp(z - lse)
                g = (p - tl.where(cols == target, 1.0, 0.0)) * coef
                tl.store(logits_row + cols, g.to(logits_ptr.dtype.element_ty), mask=valid)


@dataclass
class CEStats:
    """Per-row results that are not differentiated (for the trainer's metrics and rewind)."""

    token_loss: torch.Tensor  # [N] fp32, 0 on unlabelled rows
    token_entropy: torch.Tensor  # [N] fp32
    hits: torch.Tensor  # [N] int32
    n_labelled: int


def _as_scale(grad_scale, device) -> torch.Tensor:
    if grad_scale is None:
        return torch.ones(1, dtype=torch.float32, device=device)
    if torch.is_tensor(grad_scale):
        return grad_scale.detach().to(device=device, dtype=torch.float32).reshape(1).clone()
    return torch.full((1,), float(grad_scale), dtype=torch.float32, device=device)


def _denominator(num_items, n_labelled: int, device) -> torch.Tensor:
    if num_items is None:
        return torch.full((), max(n_labelled, 1), dtype=torch.float32, device=device)
    if torch.is_tensor(num_items):
        return num_items.to(device).reshape(()).to(torch.float32)
    return torch.full((), float(num_items), dtype=torch.float32, device=device)


class ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    """loss = sum_rows CE(hidden @ W^T, target) / num_items, gradient taken chunk by chunk.

    forward(hidden [N, H] (fp32 or fp16), weight16 [V, H] fp16, targets [N] int64, num_items,
    grad_scale, chunk) -> (loss, token_loss, token_entropy, hits). The weight is frozen (tied
    embedding): no gradient flows to it.
    """

    @staticmethod
    def forward(ctx, hidden, weight16, targets, num_items, grad_scale, chunk, n_labelled):
        device = hidden.device
        n_rows, hidden_size = hidden.shape
        vocab = weight16.shape[0]
        with_grad = bool(ctx.needs_input_grad[0])
        with torch.autocast(device_type=device.type, enabled=False):
            labelled = targets != IGNORE_INDEX
            # Labelled rows first (stable), so the gradient GEMM skips the unlabelled ones.
            order = torch.argsort((~labelled).to(torch.int8), stable=True)
            h16 = hidden.index_select(0, order).to(torch.float16)
            tgt = targets.index_select(0, order).contiguous()
            token_loss = torch.empty(n_rows, dtype=torch.float32, device=device)
            token_entropy = torch.empty(n_rows, dtype=torch.float32, device=device)
            hits = torch.empty(n_rows, dtype=torch.int32, device=device)
            scale = _as_scale(grad_scale, device)
            den = _denominator(num_items, n_labelled, device)
            coef = (scale / den).reshape(1).contiguous()
            grad16 = (torch.empty((n_labelled, hidden_size), dtype=torch.float16, device=device)
                      if with_grad else None)
            buf = torch.empty((min(chunk, max(n_rows, 1)), vocab), dtype=torch.float16,
                              device=device)
            for lo, hi, grad in ((0, n_labelled, with_grad), (n_labelled, n_rows, False)):
                for start in range(lo, hi, chunk):
                    stop = min(start + chunk, hi)
                    logits = buf[: stop - start]
                    torch.mm(h16[start:stop], weight16.t(), out=logits)
                    _launch_ce_kernel(logits, tgt, token_loss, token_entropy, hits, coef, start,
                                      grad)
                    if grad:
                        torch.mm(logits, weight16, out=grad16[start:stop])
            del buf, h16
            loss_sum = token_loss.sum(dtype=torch.float64).to(torch.float32)
            loss = loss_sum / den
            # Back to the caller's row order.
            unsort = torch.empty_like(order)
            unsort[order] = torch.arange(n_rows, device=device)
            token_loss = token_loss.index_select(0, unsort)
            token_entropy = token_entropy.index_select(0, unsort)
            hits = hits.index_select(0, unsort)
        ctx.mark_non_differentiable(token_loss, token_entropy, hits)
        ctx.n_rows, ctx.hidden_size = n_rows, hidden_size
        ctx.hidden_dtype = hidden.dtype
        if with_grad:
            ctx.save_for_backward(grad16, order[:n_labelled], scale)
        ctx.n_labelled = n_labelled
        return loss, token_loss, token_entropy, hits

    @staticmethod
    def backward(ctx, grad_loss, _grad_token_loss, _grad_entropy, _grad_hits):
        grad16, rows, scale = ctx.saved_tensors
        # In training grad_loss is exactly the scale used in forward, so factor == 1.0 and the
        # fp16 gradient is passed on unchanged, as lm_head's backward does in base.
        factor = (grad_loss.to(torch.float32).reshape(1) / scale).to(torch.float32)
        grad_hidden = torch.zeros((ctx.n_rows, ctx.hidden_size), dtype=ctx.hidden_dtype,
                                  device=grad16.device)
        if ctx.n_labelled:
            grad_hidden.index_copy_(0, rows, (grad16.to(torch.float32) * factor).to(ctx.hidden_dtype))
        return grad_hidden, None, None, None, None, None, None


def _launch_ce_kernel(logits, targets, token_loss, token_entropy, hits, coef, row_start, grad):
    n = logits.shape[0]
    if n == 0:
        return
    if triton is not None and _runtime.device_ok(logits):
        _cross_entropy_fwd_bwd_inplace_kernel[(n,)](
            logits, logits.stride(0), targets, token_loss, token_entropy, hits, coef,
            row_start, logits.shape[1], int(bool(grad)), BLOCK_V=BLOCK_V, num_warps=NUM_WARPS)
        return
    # Plain PyTorch, same math (no Triton / CPU without the interpreter).
    z = logits.float()
    tgt = targets[row_start:row_start + n]
    lab = tgt != IGNORE_INDEX
    lse = torch.logsumexp(z, dim=-1)
    p = torch.exp(z - lse[:, None])
    token_entropy[row_start:row_start + n] = lse - (p * z).sum(-1)
    zt = z.gather(1, tgt.clamp(min=0)[:, None]).squeeze(1)
    token_loss[row_start:row_start + n] = torch.where(lab, lse - zt, torch.zeros_like(lse))
    hits[row_start:row_start + n] = ((z.argmax(-1) == tgt) & lab).to(torch.int32)
    if grad:
        onehot = torch.zeros_like(p).scatter_(1, tgt.clamp(min=0)[:, None], 1.0)
        g = (p - onehot) * coef * lab[:, None].to(p.dtype)
        logits.copy_(g.to(logits.dtype))


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight16: torch.Tensor,
    targets: torch.Tensor,
    num_items: Optional[Union[torch.Tensor, int]] = None,
    grad_scale: Optional[Union[torch.Tensor, float]] = None,
    chunk: Optional[int] = None,
) -> tuple[torch.Tensor, CEStats]:
    """Mean (or sum / num_items) next-token cross-entropy of `hidden @ weight16^T`.

    `hidden` [N, H] are the final hidden states of real tokens, `targets` [N] the label each of
    them predicts (IGNORE_INDEX where none). Returns the loss and the per-row statistics.
    """
    chunk = _runtime.CE_CHUNK if chunk is None else chunk
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        raise ValueError("chunk must be a positive integer")
    if hidden.ndim != 2 or weight16.ndim != 2 or targets.ndim != 1:
        raise ValueError("expected hidden [N, H], weight [V, H], targets [N]")
    if hidden.shape[1] != weight16.shape[1] or hidden.shape[0] != targets.shape[0] or weight16.shape[0] == 0:
        raise ValueError("incompatible hidden, weight or target shapes")
    if hidden.device != weight16.device or hidden.device != targets.device:
        raise ValueError("hidden, weight and targets must be on the same device")
    if targets.dtype != torch.int64:
        raise ValueError("targets must have dtype int64")
    if weight16.requires_grad:
        raise ValueError("fused_linear_cross_entropy requires a frozen weight")
    labelled = targets != IGNORE_INDEX
    invalid = labelled & ((targets < 0) | (targets >= weight16.shape[0]))
    # Validate before the Triton target load; an invalid label must never become
    # an out-of-bounds GPU pointer. Transfer both counts in the existing host sync.
    n_labelled, n_invalid = torch.stack((labelled.sum(), invalid.sum())).tolist()
    if n_invalid:
        raise ValueError("targets must be -100 or an index in [0, vocab_size)")
    if weight16.dtype != torch.float16:
        weight16 = weight16.to(torch.float16)  # transient: the gradient is taken in forward
    loss, token_loss, token_entropy, hits = ChunkedLinearCrossEntropyFunction.apply(
        hidden, weight16, targets, num_items, grad_scale, chunk, n_labelled)
    return loss, CEStats(token_loss, token_entropy, hits, n_labelled)


def reference_linear_cross_entropy(hidden, weight, targets, num_items=None):
    """What base computes from the full logits (for the tests): loss, per-row CE, entropy, hits."""
    logits = (hidden.to(torch.float16) @ weight.to(torch.float16).t()).float()
    lab = targets != IGNORE_INDEX
    ce = torch.nn.functional.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX,
                                           reduction="none")
    den = lab.sum() if num_items is None else num_items
    loss = ce.sum() / den
    logp = torch.log_softmax(logits, dim=-1)
    entropy = -(logp.exp() * logp).sum(-1)
    hits = ((logits.argmax(-1) == targets) & lab).to(torch.int32)
    return loss, ce, entropy, hits
