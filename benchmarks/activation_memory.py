"""What one decoder layer keeps for backward, measured, per code path (no GPU needed).

    TRITON_INTERPRET=1 python benchmarks/activation_memory.py

Builds Qwen2.5-1.5B-shaped decoder layers (hidden 1536, MLP 8960, 12 query / 2 key-value heads
of 128) in 4-bit NF4 with LoRA r=16 on q/v (dropout 0.05), runs one training forward on a padded
batch, and sums the bytes of every tensor autograd saves for backward (torch's saved-tensor
hooks; each storage counted once; model weights and LoRA parameters excluded). The per-layer
number is the difference between a 2-layer and a 1-layer model, so the embedding, final norm and
loss cancel out. Divided by the real tokens of the batch it is the activation cost that the
largest batch (1960 of 2048 tokens on this task) multiplies by 28 layers at the memory peak.

On a CPU the attention backend is PyTorch's CPU flash kernel, which saves the same inputs and
output as the T4's memory-efficient kernel (q, k, v, output, logsumexp).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LENGTHS = (64, 40, 17, 52)  # a right-padded batch of 4 rows
PATHS = {
    "phase1 (masked attention, separate MLP norm)": {"VARLEN_ATTN": False, "FUSED_MLP_NORM": False,
                                                     "LORA_REMAT": False, "PACKED_MASKS": False,
                                                     "RESID_DELTA": False, "GQA_ATTN": False},
    "phase2 (per-row attention, fused MLP norm)": {"VARLEN_ATTN": True, "FUSED_MLP_NORM": True,
                                                   "LORA_REMAT": False, "PACKED_MASKS": False,
                                                   "RESID_DELTA": False, "GQA_ATTN": False},
    "phase3 (+ LoRA input rematerialized)": {"VARLEN_ATTN": True, "FUSED_MLP_NORM": True,
                                             "LORA_REMAT": True, "PACKED_MASKS": False,
                                             "RESID_DELTA": False, "GQA_ATTN": False},
    "phase4 (+ mask bits, fp16 attention delta)": {"VARLEN_ATTN": True, "FUSED_MLP_NORM": True,
                                                   "LORA_REMAT": True, "PACKED_MASKS": True,
                                                   "RESID_DELTA": True, "GQA_ATTN": False},
    "defaults (+ k/v not expanded, enable_gqa)": {"VARLEN_ATTN": True, "FUSED_MLP_NORM": True,
                                                  "LORA_REMAT": True, "PACKED_MASKS": True,
                                                  "RESID_DELTA": True, "GQA_ATTN": True},
}


def build(n_layers: int):
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=512, hidden_size=1536, intermediate_size=8960,
                      num_hidden_layers=n_layers, num_attention_heads=12, num_key_value_heads=2,
                      tie_word_embeddings=True)
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
                q.weight = bnb.nn.Params4bit(lin.weight.data, requires_grad=False,
                                             compress_statistics=True, quant_type="nf4")
                if lin.bias is not None:
                    q.bias = torch.nn.Parameter(lin.bias.data, requires_grad=False)
                setattr(parent, name, q)
    model = model.to("cpu")
    model.is_loaded_in_4bit = True
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    model.train()
    return model


def saved_bytes(model, fast: bool, breakdown: dict = None) -> int:
    import torch

    from solution.patches import patch_qwen as pm

    seq = max(LENGTHS)
    ids = torch.randint(0, 512, (len(LENGTHS), seq))
    attn = torch.zeros(len(LENGTHS), seq, dtype=torch.long)
    for row, n in enumerate(LENGTHS):
        attn[row, :n] = 1
    labels = ids.clone()
    labels[attn == 0] = -100
    weights = {p.untyped_storage().data_ptr() for p in model.parameters()}
    weights |= {b.untyped_storage().data_ptr() for b in model.buffers()}
    seen: dict[int, int] = {}

    def pack(t):
        storage = t.untyped_storage()
        if storage.data_ptr() not in weights:
            if breakdown is not None and storage.data_ptr() not in seen:
                key = (str(t.dtype).replace("torch.", ""),
                       "x".join(map(str, t.shape)) or "scalar")
                count, size = breakdown.get(key, (0, 0))
                breakdown[key] = (count + 1, size + storage.nbytes())
            seen[storage.data_ptr()] = storage.nbytes()
        return t

    request = pm._StepRequest(None) if fast else None
    pm._REQUEST = request
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t), \
                torch.autocast("cpu", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=attn, labels=labels, use_cache=False)
    finally:
        pm._REQUEST = None
    assert (request.result is not None) if fast else out.logits is not None
    del out
    return sum(seen.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--breakdown", action="store_true",
                        help="also list what one layer saves, by dtype and shape")
    args = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") != "1" and not __import__("torch").cuda.is_available():
        sys.exit("set TRITON_INTERPRET=1 (no GPU): the solution's kernels run in the interpreter")
    import solution
    from solution import _runtime
    from solution.patches import patch_qwen as pm

    solution.apply()
    models = {n: build(n) for n in (1, 2)}
    for model in models.values():
        assert solution.apply_t4_triton_patches(model)["fast_path"]
    real, padded = sum(LENGTHS), len(LENGTHS) * max(LENGTHS)
    print(f"batch: {len(LENGTHS)} rows, {real} real tokens of {padded} positions")
    rows = {"base Soup": None, **PATHS}
    for label, switches in rows.items():
        for key, value in (switches or {}).items():
            setattr(_runtime, key, value)
        two, one = {}, {}
        per_layer = (saved_bytes(models[2], switches is not None, two)
                     - saved_bytes(models[1], switches is not None, one))
        print(f"{label:46s} {per_layer / 2**20:7.2f} MiB per layer = "
              f"{per_layer / real / 1024:5.1f} KiB per real token per layer "
              f"(x 1960 tokens x 28 layers = {per_layer / real * 1960 * 28 / 2**30:4.2f} GiB)")
        if args.breakdown:  # what the second layer added: the per-layer tensors
            layer = {key: (two[key][0] - one.get(key, (0, 0))[0],
                           two[key][1] - one.get(key, (0, 0))[1]) for key in two}
            for (dtype, shape), (count, size) in sorted(layer.items(), key=lambda kv: -kv[1][1]):
                if count > 0 and size > 0:
                    print(f"      {count:2d} x {dtype:8s} {shape:22s} {size / real / 1024:6.2f} "
                          "KiB/token")
    for model in models.values():
        pm.restore_t4_triton_patches(model)


if __name__ == "__main__":
    main()
