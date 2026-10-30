"""Correctness of our kernels against plain PyTorch / bitsandbytes, forward and backward.

Run on a GPU: `python -m pytest solution -q` (the Kaggle notebook does). The last test runs a
tiny 4-bit Qwen2 + LoRA through base transformers and through the fast path and compares the
loss, the logged metrics and every LoRA gradient.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    """The starter kit's example kernel (kept as shipped; it is not used by apply())."""
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


# ----------------------------------------------------------------------------- Module 1: loss


@pytest.mark.parametrize("rows,vocab,chunk", [(300, 151936, 128), (37, 5003, 16)])
def test_fused_linear_cross_entropy(rows, vocab, chunk):
    from solution.kernels.cross_entropy import (fused_linear_cross_entropy,
                                               reference_linear_cross_entropy)

    torch.manual_seed(0)
    hidden = torch.randn(rows, 256, device="cuda") * 0.5
    weight = (torch.randn(vocab, 256, device="cuda") * 0.2).half()
    targets = torch.randint(0, vocab, (rows,), device="cuda")
    targets[::3] = -100
    num_items = (targets != -100).sum()
    scale = 65536.0  # the GradScaler's initial scale, as in training

    h = hidden.clone().requires_grad_(True)
    loss, stats = fused_linear_cross_entropy(h, weight, targets, num_items=num_items,
                                             grad_scale=scale, chunk=chunk)
    (loss * scale).backward()
    ref_h = hidden.clone().requires_grad_(True)
    ref_loss, ref_ce, ref_entropy, ref_hits = reference_linear_cross_entropy(
        ref_h, weight, targets, num_items)
    (ref_loss * scale).backward()

    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-5)
    labelled = targets != -100
    torch.testing.assert_close(stats.token_loss, torch.where(labelled, ref_ce, 0.0),
                               rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(stats.token_entropy, ref_entropy, rtol=1e-3, atol=1e-3)
    assert (stats.hits == ref_hits).float().mean() > 0.99  # fp16 GEMM ties may flip one
    assert stats.n_labelled == int(num_items)
    # dL/dz is rounded to fp16 at the GradScaler scale in both, so a rare last-bit flip of one
    # logit's gradient moves a whole row a little: compare the norm of the difference.
    rel = (h.grad - ref_h.grad).norm() / ref_h.grad.norm()
    assert rel < 1e-2, float(rel)


# ------------------------------------------------------------------------- Module 2: backbone


def test_add_rmsnorm_forward_backward():
    from solution.kernels.rmsnorm import add_rmsnorm, ref_add_rmsnorm

    torch.manual_seed(0)
    resid = torch.randn(333, 1536, device="cuda", requires_grad=True)
    delta = torch.randn(333, 1536, device="cuda").half().requires_grad_(True)
    weight = torch.rand(1536, device="cuda") + 0.5
    new_resid, y = add_rmsnorm(resid, delta, weight, 1e-6)
    grads = [torch.randn_like(y), torch.randn_like(y)]
    torch.autograd.backward([y, new_resid], grads)

    r2 = resid.detach().clone().requires_grad_(True)
    d2 = delta.detach().clone().requires_grad_(True)
    ref_resid, ref_y = ref_add_rmsnorm(r2, d2, weight, 1e-6)
    torch.autograd.backward([ref_y, ref_resid], grads)
    torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(new_resid, ref_resid, rtol=0, atol=0)
    torch.testing.assert_close(resid.grad, r2.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(delta.grad.float(), d2.grad.float(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("rotate,with_lora", [(True, True), (True, False), (False, True)])
def test_rope_lora_scatter(rotate, with_lora):
    from solution.kernels.rope import ref_rope, rope_lora_scatter

    torch.manual_seed(0)
    batch, seq, heads, dim = 4, 37, 12, 128
    mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
    mask[1, 20:] = 0
    mask[3, 5:] = 0
    dst = mask.reshape(-1).nonzero().squeeze(1)
    n_tok = dst.numel()
    inv = 1.0 / (1e6 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
    freqs = torch.arange(seq, device="cuda").float()[:, None] * inv[None]
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos(), emb.sin()
    base = torch.randn(n_tok, heads * dim, device="cuda").half().requires_grad_(True)
    lora = torch.randn(n_tok, heads * dim, device="cuda").half().requires_grad_(True)
    out = rope_lora_scatter(base, lora if with_lora else None, cos, sin, dst % seq, dst,
                            batch, seq, heads, rotate, 16)
    gout = torch.randn_like(out)
    out.backward(gout)

    b2 = base.detach().clone().requires_grad_(True)
    l2 = lora.detach().clone().requires_grad_(True)
    x = (b2.float() + (l2.float() if with_lora else 0.0)).view(n_tok, heads, dim)
    if rotate:
        x = ref_rope(x, cos[dst % seq][:, None], sin[dst % seq][:, None])
    padded = torch.zeros(batch * seq, heads, dim, device="cuda")
    padded = padded.index_copy(0, dst, x).view(batch, seq, heads, dim).permute(0, 2, 1, 3)
    padded.backward(gout)
    torch.testing.assert_close(out, padded, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(base.grad.float(), b2.grad.float(), rtol=2e-3, atol=2e-3)
    if with_lora:
        torch.testing.assert_close(lora.grad.float(), l2.grad.float(), rtol=2e-3, atol=2e-3)


def test_fp16_swiglu():
    from solution.kernels.swiglu import swiglu

    torch.manual_seed(0)
    g = (torch.randn(513, 8960, device="cuda") * 3).half().requires_grad_(True)
    u = torch.randn(513, 8960, device="cuda").half().requires_grad_(True)
    dp = torch.randn(513, 8960, device="cuda").half()
    g2 = g.detach().float().requires_grad_(True)
    u2 = u.detach().float().requires_grad_(True)
    p = swiglu(g, u)
    p.backward(dp)
    ref = torch.nn.functional.silu(g2) * u2
    ref.backward(dp.float())
    torch.testing.assert_close(p.float(), ref.half().float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(g.grad.float(), g2.grad.half().float(), rtol=2e-3, atol=4e-3)
    torch.testing.assert_close(u.grad.float(), u2.grad.half().float(), rtol=2e-3, atol=4e-3)


# ----------------------------------------------------------------------------- Module 3: NF4


def _linear4bit(out_features, in_features, bias=False):
    import bitsandbytes as bnb

    torch.manual_seed(0)
    dense = torch.randn(out_features, in_features, dtype=torch.float16) * 0.05
    layer = bnb.nn.Linear4bit(in_features, out_features, bias=bias, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4")
    layer.weight = bnb.nn.Params4bit(dense, requires_grad=False, compress_statistics=True,
                                     quant_type="nf4")
    if bias:
        layer.bias = torch.nn.Parameter(torch.randn(out_features) * 0.1, requires_grad=False)
    return layer.cuda()


@pytest.mark.parametrize("shape", [(8960, 1536), (256, 1536), (1536, 8960)])
def test_nf4_dequant_is_bitsandbytes_bit_for_bit(shape):
    pytest.importorskip("bitsandbytes")
    from solution.kernels.nf4 import dequantize_nf4, nf4_eligible, reference_dequantize

    layer = _linear4bit(*shape)
    assert nf4_eligible(layer)
    assert torch.equal(dequantize_nf4(layer), reference_dequantize(layer))


def test_nf4_linear_matches_bitsandbytes_forward_backward():
    pytest.importorskip("bitsandbytes")
    from solution.kernels.nf4 import nf4_linear

    layer = _linear4bit(256, 1536, bias=True)
    x = torch.randn(700, 1536, device="cuda").half().requires_grad_(True)
    out = nf4_linear(layer, x)
    grad = torch.randn_like(out)
    out.backward(grad)
    x2 = x.detach().clone().requires_grad_(True)
    ref = layer(x2)  # bitsandbytes' own path (fp16 input -> fp16 output)
    ref.backward(grad)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, x2.grad, rtol=0, atol=0)


# ------------------------------------------------------------------ end to end, tiny model


def test_fast_path_matches_base_on_tiny_4bit_qwen2():
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("peft")
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import solution
    from solution.patches import patch_qwen as patch_model

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=1000, hidden_size=256, intermediate_size=640,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256, tie_word_embeddings=True)
    cfg._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(cfg).half()
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                lin = getattr(parent, name)
                q = bnb.nn.Linear4bit(lin.in_features, lin.out_features,
                                      bias=lin.bias is not None, compute_dtype=torch.float16,
                                      compress_statistics=True, quant_type="nf4")
                q.weight = bnb.nn.Params4bit(lin.weight.data * 4, requires_grad=False,
                                             compress_statistics=True, quant_type="nf4")
                if lin.bias is not None:
                    q.bias = torch.nn.Parameter(lin.bias.data + 0.1, requires_grad=False)
                setattr(parent, name, q)
    model = model.cuda()
    model.is_loaded_in_4bit = True
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    for name, p in model.named_parameters():
        if "lora_" in name:
            with torch.no_grad():
                p.normal_(0, 0.05)
    model.train()
    solution.apply()
    assert solution.apply_t4_triton_patches(model)["fast_path"]

    batch, seq = 4, 40
    ids = torch.randint(0, cfg.vocab_size, (batch, seq), device="cuda")
    attn = torch.ones(batch, seq, dtype=torch.long, device="cuda")
    attn[1, 25:] = 0
    attn[2, 9:] = 0
    labels = ids.clone()
    labels[:, :6] = -100
    labels[attn == 0] = -100
    num_items = (labels[:, 1:] != -100).sum()
    params = [p for p in model.parameters() if p.requires_grad]

    def run(fast):
        for p in params:
            p.grad = None
        torch.manual_seed(1234)  # the same LoRA dropout masks
        request = patch_model._StepRequest(None) if fast else None
        patch_model._REQUEST = request
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=ids, attention_mask=attn, labels=labels,
                            num_items_in_batch=num_items, use_cache=False)
        finally:
            patch_model._REQUEST = None
        out.loss.backward()
        return out, request, [p.grad.detach().clone() for p in params]

    base_out, _, base_grads = run(False)
    fast_out, request, fast_grads = run(True)
    assert fast_out.logits is None and request.result is not None
    torch.testing.assert_close(fast_out.loss, base_out.loss, rtol=2e-3, atol=2e-3)
    stats = request.result[0]
    logits = base_out.logits.float()
    logp = torch.log_softmax(logits, -1)
    entropy = (-(logp.exp() * logp).sum(-1) * attn).sum() / attn.sum()
    torch.testing.assert_close(stats.token_entropy.sum() / attn.sum(), entropy,
                               rtol=2e-3, atol=2e-3)
    for g_base, g_fast in zip(base_grads, fast_grads):
        rel = (g_base - g_fast).norm() / g_base.norm().clamp(min=1e-12)
        assert rel < 2e-2, float(rel)
    patch_model.restore_t4_triton_patches(model)
