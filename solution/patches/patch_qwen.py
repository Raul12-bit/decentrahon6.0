"""How the kernels get into `soup train`: a fused training forward for Soup's 4-bit Qwen2 + LoRA.

Entry points (installed by `solution.apply()`):

* `Qwen2ForCausalLM.forward` (class patch). When a training step asks for it (see below) and the
  model is the 4-bit NF4 Qwen2 + LoRA that Soup builds for this task, the decoder runs here:
    - pad tokens are dropped: every per-token op (norms, projections, LoRA, MLP, lm_head, loss)
      sees only the real tokens of the batch (the README rules allow removing padding inside a
      batch). Attention: on a right-padded batch each row runs causal SDPA over its own real
      tokens (exactly what base's mask allows them to see, see `_varlen_attention`); otherwise
      it runs on the padded [B, H, S, D] layout through transformers' own SDPA call with base's
      mask;
    - RMSNorm + residual add, LoRA add + RoPE, SwiGLU, NF4 unpacking and the lm_head + loss are
      the Triton kernels of solution/kernels (rmsnorm, rope, swiglu, nf4, cross_entropy);
    - no [B, S, V] logits: the output carries the loss and no logits.
  Anything else (eval, generation, another model, a config the fast path does not cover) goes to
  the original forward untouched.
* `trl.SFTTrainer.compute_loss` (class patch; Soup's RewindSFTTrainer inherits it). It asks the
  model for the fast path, hands it the GradScaler scale, and logs `entropy`, `num_tokens` and
  `mean_token_accuracy` with TRL's formulas from the per-token values the loss kernel returns,
  and leaves the per-token losses where Soup's rewind recorder reads them for a step without
  logits (`_soup_chunked_step`, the hook Soup itself uses for its chunked loss, #1693).
* `SFTTrainerWrapper.setup` / `.train` (Soup): start compiling the kernels while the model loads;
  `apply_t4_triton_patches(model)` before training, undone when training ends.

Numerics: every rounding point of base Soup is kept (fp32 residual stream, fp16 GEMM inputs,
fp32 LoRA add, fp16 SDPA inputs, fp16 dL/dz at the GradScaler scale). LoRA dropout masks are drawn
on the padded [B * S, H] layout, so the random stream is the one base uses. What differs is the
order of some fp32 sums and the M of the GEMMs (fewer rows), i.e. rounding, not math.
"""

from __future__ import annotations

import types
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .. import _runtime
from ..kernels.cross_entropy import IGNORE_INDEX, fused_linear_cross_entropy
from ..kernels.nf4 import (add_norm_gate_up_swiglu, gate_up_swiglu, is_linear4bit, nf4_eligible,
                           nf4_linear)
from ..kernels.lora import enabled as lora_remat_enabled
from ..kernels.lora import lora_remat
from ..kernels.rmsnorm import add_rmsnorm, add_rmsnorm_rstd
from ..kernels.rope import rope_lora_scatter
from ..kernels.swiglu import swiglu

STEP_ATTR = "_soup_chunked_step"  # soup_cli.utils.stream_loss_chunk.STEP_ATTR
_ORIGINALS: dict[str, Any] = {}
_REQUEST: Optional["_StepRequest"] = None


class _StepRequest:
    """Set by the trainer around one training forward."""

    def __init__(self, grad_scale):
        self.grad_scale = grad_scale
        self.result = None  # (CEStats, dst, batch, seq) when the fast path ran


class _LayerCtx:
    def __init__(self, batch, seq, dst, cos, sin, cos_row, mask, block_heads, row_lengths=None):
        self.batch, self.seq = batch, seq
        self.dst = dst  # [T] flat index b * S + s of each real token
        self.n_tok = dst.numel()
        self.full = self.n_tok == batch * seq
        self.cos, self.sin, self.cos_row = cos, sin, cos_row
        self.mask = mask
        self.block_heads = block_heads
        # Right-padded batch with pads: the real length of each row (rows with no token left
        # out), and the identity scatter index of the token-major layout [1, H, T, D]. None:
        # attention runs on the padded layout with `mask`.
        self.row_lengths = row_lengths
        self.ident = (torch.arange(self.n_tok, device=dst.device)
                      if row_lengths is not None else None)


# --------------------------------------------------------------------------- module helpers


def _lora_parts(module):
    """(base_layer, (lora_A, lora_B, dropout, scaling) or None), or None if unsupported."""
    try:
        from peft.tuners.lora import LoraLayer
    except Exception:  # noqa: BLE001
        LoraLayer = ()  # noqa: N806
    if not LoraLayer or not isinstance(module, LoraLayer):
        return module, None
    if getattr(module, "merged", False):
        return None
    base = module.get_base_layer()
    if getattr(module, "disable_adapters", False):
        return base, None
    active = [name for name in module.active_adapters if name in module.lora_A]
    if len(active) != 1:
        return None if active else (base, None)
    name = active[0]
    if name in getattr(module, "lora_variant", {}) or getattr(module, "use_dora", {}).get(name):
        return None
    return base, (module.lora_A[name], module.lora_B[name], module.lora_dropout[name],
                  module.scaling[name])


def _embed_fp32(embedding, ids):
    weight = embedding.weight
    out = F.embedding(ids, weight, embedding.padding_idx, embedding.max_norm,
                      embedding.norm_type, embedding.scale_grad_by_freq, embedding.sparse)
    return out.to(torch.float32)


class _PaddedInput:
    """The attention input on the padded [B * S, H] layout (zeros at the pads), built at most
    once per layer and shared by the q and v LoRA dropouts."""

    def __init__(self, x32, ctx: _LayerCtx, norm_state=None):
        self.x32, self.ctx = x32, ctx
        self.norm_state = norm_state  # (resid, rstd, weight, eps) of the norm that made x32
        self.full, self.dst = ctx.full, ctx.dst
        self._with_grad = None
        self._detached = None

    def get(self):
        """The padded copy inside the autograd graph (peft's LoRA path)."""
        if self._with_grad is None:
            ctx = self.ctx
            padded = self.x32.new_zeros((ctx.batch * ctx.seq, self.x32.shape[1]))
            self._with_grad = padded.index_copy(0, ctx.dst, self.x32)
        return self._with_grad

    def padded(self):
        """The padded copy without autograd: the layout interface of kernels/lora.py."""
        if self._detached is None:
            ctx = self.ctx
            with torch.no_grad():
                self._detached = self.x32.new_zeros((ctx.batch * ctx.seq, self.x32.shape[1]))
                self._detached.index_copy_(0, ctx.dst, self.x32.detach())
        return self._detached


def _dropout_like_base(dropout, x32, ctx: _LayerCtx, padded: Optional[_PaddedInput] = None):
    """peft's lora_dropout(x) on the rows of the padded batch base would pass it.

    Dropout's random mask depends on the number of elements, not on their values, so drawing
    it over the [B * S, H] layout (zeros at the pads) gives every real token the mask it gets in
    base, and the same RNG offset for every later dropout.
    """
    if isinstance(dropout, torch.nn.Identity) or not dropout.training or not dropout.p:
        return x32
    if ctx.full:
        return dropout(x32)
    if padded is None:
        padded = _PaddedInput(x32, ctx)
    return dropout(padded.get()).index_select(0, ctx.dst)


def _linear16(module, x16):
    """A frozen 4-bit (or plain) projection of fp16 input, fp16 output."""
    if nf4_eligible(module):
        return nf4_linear(module, x16)
    return module(x16)


def _proj_parts(module, x32, ctx, padded: Optional[_PaddedInput] = None):
    """(base16, lora16 or None) of q/k/v: the two terms peft adds in fp32 in base.

    Each projection casts x32 to fp16 on its own (as bitsandbytes does inside each Linear4bit),
    so autograd adds the q/k/v input gradients in fp32, as in base.
    """
    base, lora = _lora_parts(module)
    base16 = _linear16(base, x32.to(torch.float16))
    if lora is None:
        return base16, None
    lora_a, lora_b, dropout, scaling = lora
    p = _remat_dropout_p(lora_a, lora_b, dropout, padded)
    if p is not None:
        resid, rstd, norm_weight, eps = padded.norm_state
        return base16, lora_remat(x32, lora_a.weight, lora_b.weight, resid, rstd, norm_weight,
                                  eps, scaling, p, padded)
    dropped = _dropout_like_base(dropout, x32, ctx, padded)
    out = lora_b(lora_a(dropped)) * scaling  # fp16 under autocast
    return base16, out


def _remat_dropout_p(lora_a, lora_b, dropout, padded) -> Optional[float]:
    """The dropout probability if this LoRA branch can take kernels/lora.py (plain bias-free
    Linear A/B, nn.Dropout or none, the input norm's state at hand), else None."""
    if padded is None or padded.norm_state is None or not lora_remat_enabled():
        return None
    if type(lora_a) is not torch.nn.Linear or type(lora_b) is not torch.nn.Linear \
            or lora_a.bias is not None or lora_b.bias is not None:
        return None
    if isinstance(dropout, torch.nn.Identity):
        return 0.0
    if type(dropout) is not torch.nn.Dropout:
        return None
    return float(dropout.p) if dropout.training and 0.0 < dropout.p < 1.0 else 0.0


def _attention(attn, x32, ctx: _LayerCtx, norm_state=None):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    config = attn.config
    n_heads, n_kv, head_dim = config.num_attention_heads, config.num_key_value_heads, attn.head_dim
    padded = _PaddedInput(x32, ctx, norm_state)
    # The order of q, k, v (and so of their dropout masks) is base's.
    q_base, q_lora = _proj_parts(attn.q_proj, x32, ctx, padded)
    k_base, k_lora = _proj_parts(attn.k_proj, x32, ctx, padded)
    v_base, v_lora = _proj_parts(attn.v_proj, x32, ctx, padded)
    del padded
    if ctx.row_lengths is not None and not (attn.training and attn.attention_dropout):
        return _linear16(attn.o_proj, _varlen_attention(attn, ctx, q_base, q_lora, k_base,
                                                        k_lora, v_base, v_lora))
    common = (ctx.cos, ctx.sin, ctx.cos_row, ctx.dst, ctx.batch, ctx.seq)
    q = rope_lora_scatter(q_base, q_lora, *common, n_heads, True, ctx.block_heads)
    k = rope_lora_scatter(k_base, k_lora, *common, n_kv, True, ctx.block_heads)
    v = rope_lora_scatter(v_base, v_lora, *common, n_kv, False, ctx.block_heads)
    interface = mq.ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation,
                                                         mq.eager_attention_forward)
    out, _ = interface(attn, q, k, v, ctx.mask,
                       dropout=0.0 if not attn.training else attn.attention_dropout,
                       scaling=attn.scaling, sliding_window=attn.sliding_window)
    out = out.reshape(ctx.batch * ctx.seq, n_heads * head_dim)
    if not ctx.full:
        out = out.index_select(0, ctx.dst)
    return _linear16(attn.o_proj, out)


def _varlen_attention(attn, ctx: _LayerCtx, q_base, q_lora, k_base, k_lora, v_base, v_lora):
    """Causal attention of each row over its own real tokens only (right-padded batch).

    With right padding, base's mask lets the query at position s of row b see exactly the keys
    0..s of the same row, all of them real: that is plain causal attention over the row's
    first L_b tokens. So each row runs SDPA with `is_causal=True` and no mask on its L_b tokens,
    which skips every pad query and pad key (attention work sum(L_b^2) instead of B * S^2),
    stores q/k/v and the output for real tokens only, and needs no zero-filled padded tensors.

    The RoPE kernel writes q/k/v token-major, [1, heads, T, D] (batch 1, "sequence" = the T real
    tokens in row order, each rotated at its own position s via cos_row); `split` hands each row
    a view without a copy, and its backward concatenates the row gradients in one pass. k/v are
    expanded to the query heads with transformers' repeat_kv, as the masked SDPA path of base
    does. The per-row outputs are concatenated in row order: the [T, H * D] rows o_proj reads.
    """
    import torch.nn.functional as F
    from transformers.models.qwen2 import modeling_qwen2 as mq

    config = attn.config
    n_heads, n_kv, head_dim = config.num_attention_heads, config.num_key_value_heads, attn.head_dim
    common = (ctx.cos, ctx.sin, ctx.cos_row, ctx.ident, 1, ctx.n_tok)
    q = rope_lora_scatter(q_base, q_lora, *common, n_heads, True, ctx.block_heads)
    k = rope_lora_scatter(k_base, k_lora, *common, n_kv, True, ctx.block_heads)
    v = rope_lora_scatter(v_base, v_lora, *common, n_kv, False, ctx.block_heads)
    groups = n_heads // n_kv
    outs = []
    for qi, ki, vi in zip(q.split(ctx.row_lengths, dim=2), k.split(ctx.row_lengths, dim=2),
                          v.split(ctx.row_lengths, dim=2)):
        if groups > 1:
            ki, vi = mq.repeat_kv(ki, groups), mq.repeat_kv(vi, groups)
        oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0, is_causal=True,
                                            scale=attn.scaling)
        outs.append(oi.transpose(1, 2).reshape(qi.shape[2], n_heads * head_dim))
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


def _post_attention(layer, resid, attn_out):
    """(new residual, MLP output) of the second half of a decoder layer."""
    norm, mlp = layer.post_attention_layernorm, layer.mlp
    if _runtime.FUSED_MLP_NORM and nf4_eligible(mlp.gate_proj) and nf4_eligible(mlp.up_proj):
        resid, p = add_norm_gate_up_swiglu(resid, attn_out, norm.weight, norm.variance_epsilon,
                                           mlp.gate_proj, mlp.up_proj,
                                           recompute=_runtime.MLP_RECOMPUTE)
        return resid, _linear16(mlp.down_proj, p)
    resid, x = add_rmsnorm(resid, attn_out, norm.weight, norm.variance_epsilon)
    return resid, _mlp(mlp, x)


def _mlp(mlp, x32):
    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    if nf4_eligible(gate) and nf4_eligible(up):
        p = gate_up_swiglu(x32, gate, up, recompute=_runtime.MLP_RECOMPUTE)
    else:  # two separate casts: their gradients are added in fp32, as in base
        p = swiglu(_linear16(gate, x32.to(torch.float16)), _linear16(up, x32.to(torch.float16)),
                   inplace=True)  # the two GEMM outputs are read by nothing else
    return _linear16(down, p)


# ------------------------------------------------------------------------------ eligibility


def _model_reason(lm) -> Optional[str]:
    """None if the fast path covers this model, else why not. Cached on the model."""
    cached = getattr(lm, "_soup_fast_reason", False)
    if cached is None or isinstance(cached, str):
        return cached
    reason = _compute_model_reason(lm)
    try:
        object.__setattr__(lm, "_soup_fast_reason", reason)
    except Exception:  # noqa: BLE001
        pass
    if reason is not None:
        _runtime.log(f"fast path off for this model: {reason}")
    return reason


def _compute_model_reason(lm) -> Optional[str]:
    config = lm.config
    model = lm.model
    if getattr(config, "_attn_implementation", None) != "sdpa":
        return "attention backend is not sdpa"
    if getattr(config, "hidden_act", "silu") != "silu":
        return "hidden_act is not silu"
    if any(t != "full_attention" for t in getattr(config, "layer_types", ["full_attention"])):
        return "sliding-window layers"
    if getattr(model, "gradient_checkpointing", False):
        return "gradient checkpointing is on"
    head = lm.lm_head
    if not isinstance(head, torch.nn.Linear) or head.bias is not None or head.weight.requires_grad:
        return "lm_head is not a frozen bias-free Linear"
    emb = model.embed_tokens
    if not isinstance(emb, torch.nn.Embedding) or emb.weight.requires_grad:
        return "embedding is not a frozen nn.Embedding"
    for i, layer in enumerate(model.layers[: config.num_hidden_layers]):
        attn, mlp = layer.self_attn, layer.mlp
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            if norm.weight.requires_grad or norm.weight.dtype != torch.float32:
                return f"layer {i}: norm weight is trainable or not fp32"
        # bitsandbytes returns a projection in its input's dtype: that is what makes feeding it
        # fp16 equal to base's fp32 -> fp16 cast inside it. Plain nn.Linear would differ.
        for name in ("q_proj", "k_proj", "v_proj"):
            parts = _lora_parts(getattr(attn, name))
            if parts is None or not is_linear4bit(parts[0]):
                return f"layer {i}: {name} is not a 4-bit Linear (+ plain LoRA)"
        for owner, name in ((attn, "o_proj"), (mlp, "gate_proj"), (mlp, "up_proj"),
                            (mlp, "down_proj")):
            if not is_linear4bit(getattr(owner, name)):
                return f"layer {i}: {name} is not a plain 4-bit Linear"
    if model.norm.weight.requires_grad:
        return "final norm is trainable"
    return None


def _step_reason(lm, input_ids, inputs_embeds, labels, past_key_values, use_cache,
                 logits_to_keep, position_ids, kwargs) -> Optional[str]:
    if not lm.training:
        return "not training"
    if labels is None or input_ids is None or inputs_embeds is not None:
        return "no labels / input_ids"
    if past_key_values is not None or use_cache:
        return "cache in use"
    if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
        return "logits_to_keep"
    if position_ids is not None:
        return "explicit position_ids"
    if kwargs.get("output_attentions") or kwargs.get("output_hidden_states") \
            or kwargs.get("return_dict") is False or kwargs.get("shift_labels") is not None:
        return "extra outputs requested"
    if not _runtime.device_ok(input_ids):
        return "no Triton-capable device"
    device_type = input_ids.device.type
    if not torch.is_autocast_enabled(device_type) or \
            torch.get_autocast_dtype(device_type) != torch.float16:
        return "not under fp16 autocast"
    return _model_reason(lm)


# ------------------------------------------------------------------------------ the forward


def _right_padded_lengths(attention_mask, n_tok: int) -> Optional[list[int]]:
    """Per-row real lengths when the batch has pads and every row is [1 ... 1, 0 ... 0] (TRL's
    collator pads on the right), so per-row attention applies; else None. One host sync, only
    for batches that have pads."""
    batch, seq = attention_mask.shape
    if not (_runtime.UNPAD and _runtime.VARLEN_ATTN) or n_tok == batch * seq:
        return None
    real = attention_mask != 0
    lengths = real.sum(dim=1)
    right = (real == (torch.arange(seq, device=real.device) < lengths[:, None])).all()
    info = torch.cat((lengths, right.reshape(1).to(lengths.dtype))).tolist()
    if not info[-1]:
        return None
    return [int(n) for n in info[:-1] if n > 0]


def _fast_forward(lm, input_ids, attention_mask, labels, num_items_in_batch, request):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    model, config = lm.model, lm.config
    batch, seq = input_ids.shape
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq, device=device).unsqueeze(0)  # Qwen2Model's default
    if _runtime.UNPAD:
        dst = (attention_mask.reshape(-1) != 0).nonzero(as_tuple=False).squeeze(1)
    else:
        dst = torch.arange(batch * seq, device=device)
    cos_row = dst % seq  # one rotary table for the batch: row = position s

    hidden = _embed_fp32(model.embed_tokens, input_ids.reshape(-1).index_select(0, dst))
    probe = hidden.new_empty((1, 1, 1))
    cos, sin = model.rotary_emb(probe, position_ids)  # [1, S, D] fp32, as base builds them
    row_lengths = _right_padded_lengths(attention_mask, dst.numel())
    mask = None
    if row_lengths is None:  # the padded path needs base's 4-D mask (None if no padding)
        mask = mq.create_causal_mask(config=config,
                                     inputs_embeds=probe.expand(batch, seq, config.hidden_size),
                                     attention_mask=attention_mask, past_key_values=None,
                                     position_ids=position_ids)
    ctx = _LayerCtx(batch, seq, dst, cos[0], sin[0], cos_row, mask,
                    _runtime.next_pow2(config.num_attention_heads), row_lengths)

    resid, delta = hidden, None
    for layer in model.layers[: config.num_hidden_layers]:
        norm = layer.input_layernorm
        resid, x, rstd = add_rmsnorm_rstd(resid, delta, norm.weight, norm.variance_epsilon)
        attn_out = _attention(layer.self_attn, x, ctx,
                              (resid, rstd, norm.weight, norm.variance_epsilon))
        resid, delta = _post_attention(layer, resid, attn_out)
    resid, hidden = add_rmsnorm(resid, delta, model.norm.weight, model.norm.variance_epsilon)

    # ForCausalLMLoss: position t is scored against labels[t + 1]
    targets = F.pad(labels, (0, 1), value=IGNORE_INDEX)[:, 1:].reshape(-1).index_select(0, dst)
    loss, stats = fused_linear_cross_entropy(hidden, lm.lm_head.weight, targets,
                                             num_items=num_items_in_batch,
                                             grad_scale=request.grad_scale)
    request.result = (stats, dst, batch, seq)
    return mq.CausalLMOutputWithPast(loss=loss, logits=None)


def _patched_causal_lm_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                               past_key_values=None, inputs_embeds=None, labels=None,
                               use_cache=None, logits_to_keep=0, **kwargs):
    request = _REQUEST
    if request is not None and request.result is None:
        reason = _step_reason(self, input_ids, inputs_embeds, labels, past_key_values,
                              use_cache, logits_to_keep, position_ids, kwargs)
        if reason is None:
            # A late kernel failure can happen after LoRA dropout consumed RNG.
            # Retry base with the same masks, preserving the successful path's RNG advance.
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(input_ids.device) if input_ids.is_cuda else None
            try:
                return _fast_forward(self, input_ids, attention_mask, labels,
                                     kwargs.get("num_items_in_batch"), request)
            except Exception as exc:  # noqa: BLE001 - never crash a run: plain path from now on
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, input_ids.device)
                request.result = None
                object.__setattr__(self, "_soup_fast_reason", f"fast forward raised {exc!r}")
                _runtime.log(f"fast path off: {exc!r}")
        else:
            _runtime.debug(f"plain forward: {reason}")
    return _ORIGINALS["forward"](self, input_ids=input_ids, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_values=past_key_values,
                                 inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                                 logits_to_keep=logits_to_keep, **kwargs)


# ------------------------------------------------------------------------------ the trainer


def _causal_lm(model):
    seen = 0
    while seen < 8:
        seen += 1
        inner = getattr(model, "module", None)  # DDP / accelerate wrappers
        if isinstance(inner, torch.nn.Module):
            model = inner
            continue
        get_base = getattr(model, "get_base_model", None)
        if callable(get_base):
            base = get_base()
            if base is not model:
                model = base
                continue
        break
    return model


def _grad_scale(trainer):
    """The scale accelerate's GradScaler will multiply this step's loss by (None: no scaler)."""
    scaler = getattr(getattr(trainer, "accelerator", None), "scaler", None)
    if scaler is None:
        return None
    try:
        if not scaler.is_enabled():
            return None
    except Exception:  # noqa: BLE001
        return None
    scale = getattr(scaler, "_scale", None)
    if scale is not None:
        return scale
    return float(getattr(scaler, "_init_scale", 2.0 ** 16))  # initialised on the first scale()


def _trainer_reason(trainer, model, inputs) -> Optional[str]:
    from transformers.models.qwen2 import modeling_qwen2 as mq

    if not getattr(getattr(trainer, "model", None), "training", False) or \
            not getattr(model, "training", False):
        return "not training"
    if not isinstance(inputs, dict) or not {"labels", "attention_mask", "input_ids"} <= set(inputs):
        return "inputs"
    if any(key in inputs for key in ("shift_labels", "position_ids", "inputs_embeds",
                                      "pixel_values")):
        return "inputs"
    args = getattr(trainer, "args", None)
    if getattr(args, "use_liger_kernel", False) or getattr(trainer, "num_virtual_tokens", 0) \
            or getattr(trainer, "aux_loss_enabled", False) \
            or getattr(trainer, "compute_loss_func", None) is not None \
            or getattr(trainer, "label_smoother", None) is not None:
        return "trainer options"
    if not _runtime.device_ok(inputs["input_ids"]):
        return "device"
    accelerator = getattr(trainer, "accelerator", None)
    lm = _causal_lm(accelerator.unwrap_model(model) if accelerator is not None else model)
    if not isinstance(lm, mq.Qwen2ForCausalLM):
        return f"model {type(lm).__name__}"
    return _model_reason(lm)


def _gather(trainer, value):
    accelerator = getattr(trainer, "accelerator", None)
    return accelerator.gather_for_metrics(value) if accelerator is not None else value


def _log_from_stats(trainer, mode, inputs, result):
    """TRL's entropy / num_tokens / mean_token_accuracy, from the loss kernel's per-token values."""
    stats, dst, batch, seq = result
    attention_mask = inputs["attention_mask"]
    real = attention_mask.reshape(-1).index_select(0, dst).to(torch.float32)
    entropy = torch.sum(stats.token_entropy * real) / attention_mask.sum()
    trainer._metrics[mode]["entropy"].append(_gather(trainer, entropy).mean().item())
    if mode == "train":
        trainer._total_train_tokens += _gather(trainer, attention_mask.sum()).sum().item()
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    correct = _gather(trainer, stats.hits.sum())
    total = _gather(trainer, torch.tensor(stats.n_labelled, device=stats.hits.device))
    total_sum = total.sum()
    accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
    trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)
    # Soup's rewind recorder: [B, S] per-token cross-entropy, position t scoring labels[t + 1].
    token_loss = torch.zeros(batch * seq, dtype=torch.float32, device=dst.device)
    token_loss.index_copy_(0, dst, stats.token_loss)
    setattr(trainer, STEP_ATTR, types.SimpleNamespace(
        token_loss=token_loss.view(batch, seq), token_entropy=None, correct=None, labelled=None))


def _log_from_logits(trainer, mode, inputs, labels, outputs):
    """TRL 0.29's own metric code, for a step that came back with logits after all."""
    from trl.trainer.utils import entropy_from_logits

    with torch.no_grad():
        per_token_entropy = entropy_from_logits(outputs.logits)
        attention_mask = inputs["attention_mask"]
        entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
        trainer._metrics[mode]["entropy"].append(_gather(trainer, entropy).mean().item())
    if mode == "train":
        trainer._total_train_tokens += _gather(trainer, attention_mask.sum()).sum().item()
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    with torch.no_grad():
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        predictions = shift_logits.argmax(dim=-1)
        mask = shift_labels != IGNORE_INDEX
        correct = _gather(trainer, ((predictions == shift_labels) & mask).sum())
        total = _gather(trainer, mask.sum())
        total_sum = total.sum()
        accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
        trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)


def _patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    global _REQUEST
    reason = _trainer_reason(self, model, inputs)
    if reason is not None:
        _runtime.debug(f"plain compute_loss: {reason}")
        if hasattr(self, STEP_ATTR):
            setattr(self, STEP_ATTR, None)
        return _ORIGINALS["compute_loss"](self, model, inputs, return_outputs=return_outputs,
                                          num_items_in_batch=num_items_in_batch)
    from transformers import Trainer

    mode = "train" if self.model.training else "eval"
    inputs.pop("_prediction_loss_only", None)  # as TRL does before delegating
    labels = inputs["labels"]
    inputs["use_cache"] = False
    request = _StepRequest(_grad_scale(self))
    _REQUEST = request
    try:
        loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                             num_items_in_batch=num_items_in_batch)
    finally:
        _REQUEST = None
    if request.result is not None:
        _log_from_stats(self, mode, inputs, request.result)
    else:
        setattr(self, STEP_ATTR, None)
        _log_from_logits(self, mode, inputs, labels, outputs)
    return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------- model preparation (fp16 copy)


class _Fp32OutputEmbedding(torch.nn.Embedding):
    """An embedding stored in fp16 whose rows come out in fp32, as the fp32 table gave them."""

    def forward(self, input):  # noqa: A002 - nn.Embedding's signature
        return super().forward(input).to(torch.float32)


class _Fp16Head(torch.nn.Linear):
    """lm_head on the fp16 table: its input is cast to the table's dtype (autocast does the
    same in training)."""

    def forward(self, input):  # noqa: A002
        return F.linear(input.to(self.weight.dtype), self.weight, self.bias)


def apply_t4_triton_patches(model) -> dict:
    """Prepare a built Soup model (PEFT over 4-bit Qwen2) for the fast path. Idempotent.

    * Stores the tied embedding / lm_head table once in fp16 (467 MB) instead of fp32 (933 MB),
      only if fp16 -> fp32 gives back exactly the fp32 table (it does: Soup loads the model in
      fp16 on a T4 and prepare_model_for_kbit_training only upcasts). Base casts that table to
      fp16 for the lm_head GEMM at every step anyway; the embedding rows still come out in fp32.
    * Checks once whether the fast forward covers the model (the reason is logged if not).
    Returns a small report. `restore_t4_triton_patches(model)` undoes the table change.
    """
    lm = _causal_lm(model)
    report = {"model": type(lm).__name__, "fp16_table": False, "fast_path": False}
    try:
        from transformers.models.qwen2 import modeling_qwen2 as mq
    except Exception:  # noqa: BLE001
        return report
    if not isinstance(lm, mq.Qwen2ForCausalLM):
        return report
    if not _runtime.ENABLED or not _runtime.device_ok(lm.lm_head.weight):
        return report
    if hasattr(lm, "_soup_fast_reason"):
        del lm._soup_fast_reason
    if _model_reason(lm) is not None:
        return report
    # Eligibility must precede any mutation: unsupported models keep their original
    # head dtype and forward even when manual preparation was requested.
    if _runtime.FP16_EMBED and getattr(lm, "_soup_fp16_table", None) is None:
        report["fp16_table"] = _store_table_fp16(lm)
    else:
        report["fp16_table"] = getattr(lm, "_soup_fp16_table", None) is not None
    report["fast_path"] = _model_reason(lm) is None
    if report["fast_path"]:
        # Compiled and checked against PyTorch on this GPU (normally already done by the
        # warm-up thread while the model loaded); any problem -> plain Soup, not a crash.
        from ..warmup import ensure_checked

        problems = ensure_checked(lm.config, lm.lm_head.weight.device)
        if not problems:
            _check_lora_dropout(model, lm.lm_head.weight.device)
        if problems:
            restore_t4_triton_patches(model)
            report["fp16_table"] = False
            reason = "kernel self-check: " + "; ".join(problems)
            object.__setattr__(lm, "_soup_fast_reason", reason)
            report["fast_path"] = False
            _runtime.log(f"fast path off: {reason}")
    _runtime.log(f"prepared {report}")
    return report


def _check_lora_dropout(model, device) -> None:
    """Before training, on this device: does kernels/lora.py draw and apply the LoRA dropout
    exactly like F.dropout? If not, its rematerialization stays off (peft's branch is used)."""
    from ..kernels import lora

    if not _runtime.LORA_REMAT or lora.STATE["checked"]:
        return
    rates = {float(m.p) for m in model.modules()
             if type(m) is torch.nn.Dropout and 0.0 < m.p < 1.0}
    # The hidden size: the mask-bit kernels then run the binaries the warm-up compiled.
    hidden = int(getattr(_causal_lm(model).config, "hidden_size", 4096))
    for p in sorted(rates):
        if lora.dropout_self_check(device, p, hidden) is not None:
            return


def _store_table_fp16(lm) -> bool:
    emb, head = lm.get_input_embeddings(), lm.get_output_embeddings()
    if type(emb) is not torch.nn.Embedding or type(head) is not torch.nn.Linear:
        return False
    weight = emb.weight
    if head.weight is not weight or weight.dtype != torch.float32 or weight.requires_grad:
        return False
    if not weight.is_cuda and not _runtime.INTERPRET:
        return False
    with torch.no_grad():
        half = weight.detach().to(torch.float16)
        # fp16 -> fp32 must give back the fp32 table exactly. Compared in row chunks: a full
        # fp32 copy (933 MB) would exist for a few milliseconds only, a spike the harness's
        # 100 ms nvidia-smi poller can miss while PyTorch's reserved peak records it (check 9
        # compares the two; seen on a T4: reserved 3164 MiB vs nvidia-smi 2831 MiB).
        rows = half.shape[0]
        for start in range(0, rows, 4096):
            stop = min(start + 4096, rows)
            if not torch.equal(half[start:stop].to(torch.float32), weight.detach()[start:stop]):
                return False
    table = torch.nn.Parameter(half, requires_grad=False)
    emb.weight = table
    head.weight = table
    emb.__class__ = _Fp32OutputEmbedding
    head.__class__ = _Fp16Head
    object.__setattr__(lm, "_soup_fp16_table", (emb, head))
    # No torch.cuda.empty_cache() here (phase 4): the freed fp32 table stays reserved and the
    # training step reuses it. Handing it back to the driver would make the few milliseconds
    # when both tables exist a reserved peak that nvidia-smi (sampled every 100 ms) can miss
    # once the training peak is lower than it: the mismatch check 9 rejects. This way our code
    # never shrinks PyTorch's reserved memory, so a level it reaches stays mapped (and visible
    # to nvidia-smi) for the rest of the run.
    del weight
    return True


def restore_t4_triton_patches(model) -> None:
    """Put the fp32 table back (after training, before Soup saves and checks the model)."""
    lm = _causal_lm(model)
    stored = getattr(lm, "_soup_fp16_table", None)
    if stored is None:
        return
    emb, head = stored
    with torch.no_grad():
        table = torch.nn.Parameter(emb.weight.detach().to(torch.float32), requires_grad=False)
    emb.__class__ = torch.nn.Embedding
    head.__class__ = torch.nn.Linear
    emb.weight = table
    head.weight = table
    object.__setattr__(lm, "_soup_fp16_table", None)


def _restore_callback(model):
    from transformers import TrainerCallback

    class RestoreTable(TrainerCallback):
        def on_train_end(self, args, state, control, **kwargs):
            restore_t4_triton_patches(model)

    return RestoreTable()


# ------------------------------------------------------------------------------- installing


def install() -> list[str]:
    """Class-level patches. Looked up at call time, so the order w.r.t. Soup's setup is free."""
    installed = []
    from transformers.models.qwen2 import modeling_qwen2 as mq

    if "forward" not in _ORIGINALS:
        _ORIGINALS["forward"] = mq.Qwen2ForCausalLM.forward
        mq.Qwen2ForCausalLM.forward = _patched_causal_lm_forward
        installed.append("Qwen2ForCausalLM.forward")
    try:
        import trl

        if "compute_loss" not in _ORIGINALS:
            _ORIGINALS["compute_loss"] = trl.SFTTrainer.compute_loss
            trl.SFTTrainer.compute_loss = _patched_compute_loss
            installed.append("trl.SFTTrainer.compute_loss")
    except Exception as exc:  # noqa: BLE001 - without TRL the model patch alone never triggers
        _runtime.log(f"trl not patched: {exc!r}")
    try:
        from soup_cli.trainer import sft

        wrapper = sft.SFTTrainerWrapper
        if "setup" not in _ORIGINALS:
            _ORIGINALS["setup"], _ORIGINALS["train"] = wrapper.setup, wrapper.train
            wrapper.setup = _wrapped_setup
            wrapper.train = _wrapped_train
            installed.append("SFTTrainerWrapper.setup/train")
    except Exception as exc:  # noqa: BLE001
        _runtime.log(f"Soup wrapper not patched: {exc!r}")
    return installed


def _wrapped_setup(self, *args, **kwargs):
    if _runtime.WARMUP:
        from ..warmup import start_background_warmup

        start_background_warmup(getattr(getattr(self, "config", None), "base", None))
    return _ORIGINALS["setup"](self, *args, **kwargs)


def _wrapped_train(self, *args, **kwargs):
    model = getattr(self, "model", None)
    prepared = False
    if isinstance(model, torch.nn.Module):
        try:
            apply_t4_triton_patches(model)
            prepared = True
            trainer = getattr(self, "trainer", None)
            if trainer is not None and hasattr(trainer, "add_callback"):
                trainer.add_callback(_restore_callback(model))
        except Exception as exc:  # noqa: BLE001 - never stop a run over the optional prep
            _runtime.log(f"model preparation skipped: {exc!r}")
    try:
        return _ORIGINALS["train"](self, *args, **kwargs)
    finally:
        if prepared:
            restore_t4_triton_patches(model)
