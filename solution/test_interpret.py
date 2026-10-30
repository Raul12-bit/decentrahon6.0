"""The Triton kernels themselves, run by Triton's interpreter on the CPU (no GPU needed).

    TRITON_INTERPRET=1 python -m pytest solution/test_interpret.py -q

`TRITON_INTERPRET=1` must be set before Triton is imported; without it these tests skip. They
execute the actual `@triton.jit` code of solution/kernels (masks, index math, reductions,
in-place writes) on the real model dimensions (hidden 1536, 12/2 heads x 128, vocab 151936,
intermediate 8960) with fewer rows, and compare against plain PyTorch. The GPU tests in
test_kernels.py check the same on a T4 with compiled kernels.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1",
                                reason="set TRITON_INTERPRET=1 to run the kernels on the CPU")

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, VOCAB, INTER = 1536, 12, 2, 128, 151936, 8960


@pytest.fixture(autouse=True)
def _cpu_kernels():
    from solution import _runtime

    assert _runtime.INTERPRET and _runtime.triton_available()
    torch.manual_seed(0)


def test_add_rmsnorm_kernels():
    from solution.kernels.rmsnorm import add_rmsnorm, ref_add_rmsnorm

    resid = torch.randn(7, HIDDEN, requires_grad=True)
    delta = torch.randn(7, HIDDEN).half().requires_grad_(True)
    weight = torch.rand(HIDDEN) + 0.5
    new_resid, y = add_rmsnorm(resid, delta, weight, 1e-6)
    grads = [torch.randn_like(y), torch.randn_like(y)]
    torch.autograd.backward([y, new_resid], grads)
    r2 = resid.detach().clone().requires_grad_(True)
    d2 = delta.detach().clone().requires_grad_(True)
    ref_resid, ref_y = ref_add_rmsnorm(r2, d2, weight, 1e-6)
    torch.autograd.backward([ref_y, ref_resid], grads)
    torch.testing.assert_close(new_resid, ref_resid, rtol=0, atol=0)
    torch.testing.assert_close(y, ref_y, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(resid.grad, r2.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(delta.grad.float(), d2.grad.float(), rtol=2e-3, atol=2e-3)

    # first layer: no residual add
    x = torch.randn(5, HIDDEN, requires_grad=True)
    same, y = add_rmsnorm(x, None, weight, 1e-6)
    assert same is x
    gy = torch.randn_like(y)
    y.backward(gy)
    x2 = x.detach().clone().requires_grad_(True)
    ref_add_rmsnorm(x2, None, weight, 1e-6)[1].backward(gy)
    torch.testing.assert_close(x.grad, x2.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("heads,rotate,with_lora", [(HEADS, True, True), (KV_HEADS, True, False),
                                                    (KV_HEADS, False, True)])
def test_rope_lora_scatter_kernels(heads, rotate, with_lora):
    from solution.kernels.rope import ref_rope, rope_lora_scatter

    batch, seq = 3, 11
    mask = torch.ones(batch, seq, dtype=torch.long)
    mask[1, 7:] = 0
    mask[2, 2:] = 0
    dst = mask.reshape(-1).nonzero().squeeze(1)
    n_tok = dst.numel()
    inv = 1.0 / (1e6 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    freqs = torch.arange(seq).float()[:, None] * inv[None]
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos(), emb.sin()
    base = torch.randn(n_tok, heads * HEAD_DIM).half().requires_grad_(True)
    lora = torch.randn(n_tok, heads * HEAD_DIM).half().requires_grad_(True)
    out = rope_lora_scatter(base, lora if with_lora else None, cos, sin, dst % seq, dst,
                            batch, seq, heads, rotate, 16)
    assert out.dtype == torch.float16 and out.shape == (batch, heads, seq, HEAD_DIM)
    gout = torch.randn_like(out)
    out.backward(gout)

    b2 = base.detach().clone().requires_grad_(True)
    l2 = lora.detach().clone().requires_grad_(True)
    x = (b2.float() + (l2.float() if with_lora else 0.0)).view(n_tok, heads, HEAD_DIM)
    if rotate:
        x = ref_rope(x, cos[dst % seq][:, None], sin[dst % seq][:, None])
    padded = torch.zeros(batch * seq, heads, HEAD_DIM)
    padded = padded.index_copy(0, dst, x).view(batch, seq, heads, HEAD_DIM).permute(0, 2, 1, 3)
    padded = padded.half()
    padded.backward(gout)
    torch.testing.assert_close(out, padded, rtol=1e-3, atol=1e-3)
    assert out.reshape(batch, heads, seq * HEAD_DIM)[2, :, 2 * HEAD_DIM:].abs().sum() == 0
    torch.testing.assert_close(base.grad.float(), b2.grad.float(), rtol=2e-3, atol=2e-3)
    if with_lora:
        torch.testing.assert_close(lora.grad.float(), l2.grad.float(), rtol=2e-3, atol=2e-3)


def test_swiglu_kernels():
    from solution.kernels.swiglu import swiglu

    g = (torch.randn(3, INTER) * 3).half().requires_grad_(True)
    u = torch.randn(3, INTER).half().requires_grad_(True)
    dp = torch.randn(3, INTER).half()
    g2 = g.detach().float().requires_grad_(True)
    u2 = u.detach().float().requires_grad_(True)
    p = swiglu(g, u)
    p.backward(dp)
    ref = torch.nn.functional.silu(g2) * u2
    ref.backward(dp.float())
    torch.testing.assert_close(p.float(), ref.half().float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(g.grad.float(), g2.grad.half().float(), rtol=2e-3, atol=4e-3)
    torch.testing.assert_close(u.grad.float(), u2.grad.half().float(), rtol=2e-3, atol=4e-3)


@pytest.mark.parametrize("rows,vocab,chunk", [(9, VOCAB, 4), (13, 5003, 16)])
def test_cross_entropy_kernel(rows, vocab, chunk):
    from solution.kernels.cross_entropy import (fused_linear_cross_entropy,
                                               reference_linear_cross_entropy)

    hidden = torch.randn(rows, 64) * 0.5
    weight = (torch.randn(vocab, 64) * 0.3).half()
    targets = torch.randint(0, vocab, (rows,))
    targets[::3] = -100
    num_items = (targets != -100).sum()
    scale = 1024.0
    h = hidden.clone().requires_grad_(True)
    loss, stats = fused_linear_cross_entropy(h, weight, targets, num_items=num_items,
                                             grad_scale=scale, chunk=chunk)
    (loss * scale).backward()
    ref_h = hidden.clone().requires_grad_(True)
    ref_loss, ref_ce, ref_entropy, ref_hits = reference_linear_cross_entropy(
        ref_h, weight, targets, num_items)
    (ref_loss * scale).backward()
    labelled = targets != -100
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(stats.token_loss, torch.where(labelled, ref_ce, 0.0),
                               rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(stats.token_entropy, ref_entropy, rtol=1e-4, atol=1e-4)
    assert torch.equal(stats.hits, ref_hits)
    assert stats.n_labelled == int(num_items)
    rel = (h.grad - ref_h.grad).norm() / ref_h.grad.norm()
    assert rel < 1e-2, float(rel)


def test_cross_entropy_argmax_ties_take_the_first_column():
    """torch.argmax returns the first maximum; so must the kernel (TRL's token accuracy)."""
    from solution.kernels.cross_entropy import _launch_ce_kernel

    vocab = 5000
    logits = torch.zeros(3, vocab, dtype=torch.float16)
    logits[0, [17, 4000]] = 5.0  # tie across vocabulary tiles
    logits[1, [2047, 2048]] = 3.0  # tie across lanes of one tile and the next
    logits[2, :] = -1.0  # all equal
    targets = torch.tensor([4000, 2047, 0])
    loss, entropy = torch.empty(3), torch.empty(3)
    hits = torch.empty(3, dtype=torch.int32)
    _launch_ce_kernel(logits, targets, loss, entropy, hits, torch.ones(1), 0, False)
    assert hits.tolist() == [0, 1, 1]


def test_fast_path_matches_base_on_tiny_4bit_qwen2_cpu():
    """The whole patched training forward + backward (every kernel above, wired by
    patches/patch_qwen.py) against base transformers + peft + bitsandbytes, on a padded batch
    with LoRA dropout: same loss, entropy, dropout RNG stream and LoRA gradients."""
    bnb = pytest.importorskip("bitsandbytes")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import solution
    from solution.kernels.nf4 import nf4_eligible
    from solution.patches import patch_qwen as pm

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
    model = model.to("cpu")  # Params4bit quantizes on its first move
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
    assert nf4_eligible(model.base_model.model.model.layers[0].mlp.gate_proj)
    assert solution.apply_t4_triton_patches(model)["fast_path"]

    batch, seq = 4, 40
    ids = torch.randint(0, cfg.vocab_size, (batch, seq))
    attn = torch.ones(batch, seq, dtype=torch.long)
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
        request = pm._StepRequest(None) if fast else None
        pm._REQUEST = request
        try:
            with torch.autocast("cpu", dtype=torch.float16):
                out = model(input_ids=ids, attention_mask=attn, labels=labels,
                            num_items_in_batch=num_items, use_cache=False)
        finally:
            pm._REQUEST = None
        out.loss.backward()
        return out, request, [p.grad.detach().clone() for p in params], torch.randn(4)

    try:
        base_out, _, base_grads, base_rng = run(False)
        fast_out, request, fast_grads, fast_rng = run(True)
    finally:
        pm.restore_t4_triton_patches(model)
    assert fast_out.logits is None and request.result is not None, "the fast path did not run"
    torch.testing.assert_close(fast_out.loss, base_out.loss, rtol=1e-4, atol=1e-4)
    stats = request.result[0]
    logp = torch.log_softmax(base_out.logits.float(), -1)
    entropy = (-(logp.exp() * logp).sum(-1) * attn).sum() / attn.sum()
    torch.testing.assert_close(stats.token_entropy.sum() / attn.sum(), entropy,
                               rtol=1e-4, atol=1e-4)
    assert torch.equal(base_rng, fast_rng), "LoRA dropout consumed the RNG differently"
    for g_base, g_fast in zip(base_grads, fast_grads):
        rel = (g_base - g_fast).norm() / g_base.norm().clamp(min=1e-12)
        assert rel < 1e-2, float(rel)


def test_nf4_dequant_kernel_matches_reference_math():
    from solution.kernels.nf4 import BLOCK_BYTES, DEQUANT_WARPS, _nf4_dequant_nested_kernel

    n_bytes = 3 * BLOCK_BYTES + 96  # a partial last program
    packed = torch.randint(0, 256, (n_bytes,), dtype=torch.uint8)
    n_blocks = 2 * n_bytes // 64
    absmax8 = torch.randint(0, 256, (n_blocks,), dtype=torch.uint8)
    code2 = torch.linspace(-1.0, 1.0, 256)
    absmax2 = torch.rand((n_blocks + 255) // 256) + 0.5
    offset = torch.tensor([0.0371])
    nf4 = torch.linspace(-1.0, 1.0, 16)
    out = torch.empty(2 * n_bytes, dtype=torch.float16)
    grid = ((n_bytes + BLOCK_BYTES - 1) // BLOCK_BYTES,)
    _nf4_dequant_nested_kernel[grid](packed, absmax8, code2, absmax2, offset, nf4, out, n_bytes,
                                     BLOCK_BYTES=BLOCK_BYTES, QBLOCK=64, QBLOCK2=256,
                                     num_warps=DEQUANT_WARPS)
    block = torch.arange(n_blocks)
    scale = code2[absmax8.long()] * absmax2[block // 256] + offset[0]
    codes = torch.stack(((packed >> 4).long(), (packed & 15).long()), 1).reshape(-1)
    want = (nf4[codes] * scale.repeat_interleave(64)).to(torch.float16)
    assert torch.equal(out, want)
