"""Small reproducible CPU checks; these do not validate Triton/CUDA execution."""
from types import SimpleNamespace

import pytest
import torch

from solution.kernels.cross_entropy import fused_linear_cross_entropy, reference_linear_cross_entropy
from solution.kernels.rmsnorm import add_rmsnorm, ref_add_rmsnorm
from solution.kernels.rope import ref_rope, rope_lora_scatter
from solution.kernels.swiglu import swiglu


@pytest.mark.parametrize("rows,vocab,chunk", [(1, 11, 1), (17, 67, 4), (7, 2053, 3)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scale", [1., 128., 65536.])
def test_ce_loss_statistics_and_gradient(rows, vocab, chunk, dtype, scale):
    torch.manual_seed(41)
    x = (torch.randn(rows, 16) * .2).to(dtype).requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    weight = (torch.randn(vocab, 16) * .1).half()
    labels = torch.randint(vocab, (rows,))
    if rows > 1:
        labels[::3] = -100
    den = int((labels != -100).sum())
    loss, stats = fused_linear_cross_entropy(x, weight, labels, den, scale, chunk)
    ref, ce, entropy, hits = reference_linear_cross_entropy(ref_x, weight, labels, den)
    (loss * scale).backward()
    (ref * scale).backward()
    torch.testing.assert_close(loss, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(stats.token_loss, ce, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(stats.token_entropy, entropy, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(stats.hits, hits)
    torch.testing.assert_close(x.grad, ref_x.grad, atol=1e-2, rtol=1e-2)
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("rows", [0, 1, 7])
def test_ce_no_label_has_zero_gradient(rows):
    x = torch.ones(rows, 8, requires_grad=True)
    loss, stats = fused_linear_cross_entropy(x, torch.ones(11, 8).half(),
                                            torch.full((rows,), -100), num_items=1)
    loss.backward()
    assert loss.item() == 0 and stats.n_labelled == 0
    assert torch.equal(x.grad, torch.zeros_like(x))


@pytest.mark.parametrize("delta_present", [True, False])
@pytest.mark.parametrize("shape", [(1, 8), (7, 33), (0, 8)])
def test_norm_output_and_input_gradients(delta_present, shape):
    torch.manual_seed(42)
    r = torch.randn(shape, requires_grad=True)
    d = torch.randn(shape).half().requires_grad_() if delta_present else None
    w = torch.rand(shape[-1]) + .5
    r2 = r.detach().clone().requires_grad_()
    d2 = d.detach().clone().requires_grad_() if d is not None else None
    resid, y = add_rmsnorm(r, d, w, 1e-6)
    rr, yy = ref_add_rmsnorm(r2, d2, w, 1e-6)
    (y.square().sum() + resid.sum()).backward()
    (yy.square().sum() + rr.sum()).backward()
    torch.testing.assert_close(y, yy)
    torch.testing.assert_close(r.grad, r2.grad, atol=1e-4, rtol=1e-4)
    if d is not None:
        torch.testing.assert_close(d.grad, d2.grad, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("rotate", [True, False])
@pytest.mark.parametrize("heads", [1, 3])
def test_rope_custom_positions_and_gradients(rotate, heads):
    torch.manual_seed(43)
    dim, batch, seq = 8, 2, 4
    dst = torch.tensor([0, 1, 2, 4, 5])
    positions = torch.tensor([3, 1, 0, 2, 1])
    cos, sin = torch.randn(seq, dim), torch.randn(seq, dim)
    b = torch.randn(5, heads * dim).half().requires_grad_()
    l = torch.randn_like(b).requires_grad_()
    b2, l2 = b.detach().clone().requires_grad_(), l.detach().clone().requires_grad_()
    out = rope_lora_scatter(b, l, cos, sin, positions, dst, batch, seq, heads, rotate)
    x = (b2.float() + l2.float()).view(5, heads, dim)
    if rotate:
        x = ref_rope(x, cos[positions, None], sin[positions, None])
    ref = torch.zeros(batch * seq, heads, dim).index_copy(0, dst, x)
    ref = ref.view(batch, seq, heads, dim).permute(0, 2, 1, 3)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(b.grad, b2.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(l.grad, l2.grad, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", [(0, 8), (1, 8), (7, 33)])
def test_swiglu_forward_backward(shape):
    torch.manual_seed(44)
    g = torch.randn(shape).half().requires_grad_()
    u = torch.randn(shape).half().requires_grad_()
    g2, u2 = g.detach().clone().requires_grad_(), u.detach().clone().requires_grad_()
    out = swiglu(g, u)
    ref = (torch.nn.functional.silu(g2.float()) * u2.float()).half()
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(g.grad, g2.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(u.grad, u2.grad, atol=1e-2, rtol=1e-2)


def test_apply_without_triton_is_idempotent(monkeypatch):
    import sys
    import solution
    from solution import _runtime
    # Simulate a machine without Triton, whatever this one has installed.
    monkeypatch.setattr(sys.modules["solution.apply"], "_APPLIED", [])
    monkeypatch.setattr(_runtime, "triton_available", lambda: False)
    assert solution.apply() == []
    assert solution.apply() == []


def test_benchmark_rejects_unequal_curves():
    from benchmarks.verify_t4_speedup import _compare_curves
    entry = dict(loss=1., entropy=2., mean_token_accuracy=.5, num_tokens=10)
    assert _compare_curves({"logs": [entry, entry]}, {"logs": [entry]})


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_benchmark_rejects_invalid_metrics(bad):
    from benchmarks.verify_t4_speedup import _compare_curves
    entry = dict(loss=1., entropy=2., mean_token_accuracy=.5, num_tokens=10)
    assert _compare_curves({"logs": [entry]}, {"logs": [{**entry, "loss": bad}]})


def test_benchmark_rejects_empty_curves():
    from benchmarks.verify_t4_speedup import _compare_curves
    assert _compare_curves({"logs": []}, {"logs": []})


@pytest.mark.parametrize("target", [-1, 11])
def test_ce_rejects_out_of_range_labels(target):
    with pytest.raises((ValueError, IndexError)):
        fused_linear_cross_entropy(torch.ones(1, 8), torch.ones(11, 8).half(),
                                   torch.tensor([target]))


@pytest.mark.parametrize("chunk", [-1, 0])
def test_ce_rejects_nonpositive_chunk(chunk):
    with pytest.raises(ValueError):
        fused_linear_cross_entropy(torch.ones(1, 8), torch.ones(11, 8).half(),
                                   torch.tensor([1]), chunk=chunk)


def test_fallback_restores_rng_after_failed_fast_forward(monkeypatch):
    from solution.patches import patch_qwen as pm
    # Fault injection only: verify that a failure after real dropout consumption is transactional.
    def fail(*args):
        torch.nn.functional.dropout(torch.ones(20), p=.5, training=True)
        raise RuntimeError("injected late kernel failure")
    def original(*args, **kwargs):
        return torch.nn.functional.dropout(torch.ones(20), p=.5, training=True)
    monkeypatch.setattr(pm, "_step_reason", lambda *a: None)
    monkeypatch.setattr(pm, "_fast_forward", fail)
    monkeypatch.setattr(pm, "_REQUEST", pm._StepRequest(None))
    monkeypatch.setitem(pm._ORIGINALS, "forward", original)
    torch.manual_seed(45)
    expected = original()
    expected_state = torch.get_rng_state()
    torch.manual_seed(45)
    got = pm._patched_causal_lm_forward(SimpleNamespace(), input_ids=torch.ones(1, 4).long())
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), expected_state)


def test_fallback_disables_failed_path_for_later_calls(monkeypatch):
    from solution.patches import patch_qwen as pm
    lm = SimpleNamespace()
    calls = []
    monkeypatch.setattr(pm, "_step_reason", lambda *a: getattr(lm, "_soup_fast_reason", None))
    def fail(*args):
        calls.append(1)
        raise RuntimeError("injected failure")
    monkeypatch.setattr(pm, "_fast_forward", fail)
    monkeypatch.setitem(pm._ORIGINALS, "forward", lambda *a, **kw: 123)
    monkeypatch.setattr(pm, "_REQUEST", pm._StepRequest(None))
    for _ in range(2):
        assert pm._patched_causal_lm_forward(lm, input_ids=torch.ones(1, 4).long()) == 123
    assert len(calls) == 1


@pytest.mark.parametrize("disabled", [True, False])
def test_manual_prepare_does_not_mutate_unsupported_model(monkeypatch, disabled):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from solution import _runtime
    from solution.patches import patch_qwen as pm
    config = Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config).half().float().requires_grad_(False)
    original = model.lm_head.weight
    monkeypatch.setattr(_runtime, "INTERPRET", True)
    monkeypatch.setattr(_runtime, "ENABLED", not disabled)
    monkeypatch.setattr(_runtime, "device_ok", lambda tensor: True)
    # Deliberately not a quantized eligible model, even if interpreter mode was requested.
    report = pm.apply_t4_triton_patches(model)
    assert not report["fast_path"] and not report["fp16_table"]
    assert model.lm_head.weight is original
    assert type(model.lm_head) is torch.nn.Linear


def test_block_document_mask_prevents_cross_document_influence():
    from soup_cli.utils.cross_doc_attn import build_cross_doc_mask, compute_doc_boundaries
    boundaries = compute_doc_boundaries([3, 2])
    mask = torch.from_numpy(build_cross_doc_mask(boundaries, 5)).bool()
    assert not mask[3:, :3].any() and not mask.triu(1).any()
    torch.manual_seed(46)
    q, k, v = [torch.randn(1, 2, 5, 8) for _ in range(3)]
    result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, :3] += 10
    v2[:, :, :3] -= 20
    changed = torch.nn.functional.scaled_dot_product_attention(q, k2, v2, attn_mask=mask)
    torch.testing.assert_close(result[:, :, 3:], changed[:, :, 3:], atol=0, rtol=0)


@pytest.mark.parametrize("boundaries", [[], [1, 5], [0, 3], [0, 4, 3, 5]])
def test_bad_document_boundaries_are_rejected(boundaries):
    from soup_cli.utils.cross_doc_attn import build_cross_doc_mask
    with pytest.raises(ValueError):
        build_cross_doc_mask(boundaries, 5)
