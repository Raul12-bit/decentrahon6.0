# AGENTS.md — Soup T4 Triton & Memory Optimization Guide

## 1. Project Context & Mission
You are working on **Soup (`soup-cli`)**, a unified YAML-driven LLM fine-tuning framework (SFT, DPO, GRPO, QLoRA, and layer-streaming runtime).
**Primary Objective:** Accelerate a fixed Soup training run on **Kaggle NVIDIA Tesla T4 (`sm_75`, Turing, 16 GB GDDR6)** and/or significantly reduce peak VRAM consumption **at identical training quality (numerical equivalence)**.

---

## 2. Target Hardware Constraints: NVIDIA Tesla T4 (`sm_75`)
Every Triton kernel and PyTorch modification MUST strictly obey T4 hardware limits:
1. **NO Native `bfloat16` or `tf32`:** T4 Tensor Cores (`mma.sync.aligned.m16n8k8`) only accelerate `float16` (with `fp16` or `fp32` accumulation) and `int8`. Never cast activations or weights to `tl.bfloat16` or `torch.bfloat16`. Always use `tl.float16` I/O and `tl.float32` internal accumulation.
2. **Strict 64 KB Shared Memory (SRAM) per SM Limit:** Unlike A100 (164 KB) or H100 (228 KB), T4 has at most 64 KB of shared memory per SM.
   - Keep tile sizes conservative: `BLOCK_M=64`, `BLOCK_N=64`, `BLOCK_K=32` (or `BLOCK_SIZE=1024..2048` for 1D reductions).
   - Keep `num_stages=1` or `num_stages=2` (higher stages cause `triton.runtime.OutOfResources` or severe register spilling).
   - Default to `num_warps=4` (or `8` for wide row reductions like vocabulary softmax).
3. **Memory Bandwidth Bound (300 GB/s GDDR6 & PCIe Gen3):** T4 is heavily memory-bandwidth constrained. Fuse elementwise, normalization, rotary, activation, and dequantization operations to minimize global HBM round-trips.

---

## 3. Repository Architecture Map
- `src/soup_cli/utils/layer_stream_runtime.py`: Core engine for `stream_layers: true`. Contains `LayerStreamRuntime`, `StreamedDecoderLayer`, `StreamedLargeLayer`, NF4 weight packing/unpacking, and dual CUDA stream prefetching (`_prefetch_stream`).
- `src/soup_cli/trainer/`: Training loops (`sft.py`, `dpo.py`, `grpo.py`, etc.), model initialization, PEFT/LoRA wrapping, and loss computation.
- `src/soup_cli/utils/gpu.py`: Pre-flight VRAM estimation (`estimate_logits_bytes`, activation memory budgets).
- `src/soup_cli/data/`: Dataset tokenization, data collators, and `multipack` sequence packing.
- `src/soup_cli/kernels/`: Dedicated directory for custom Triton kernels, custom `torch.autograd.Function` wrappers, and model monkey-patching utilities.

---

## 4. High-Priority Optimization Targets
1. **Fused Chunked `lm_head` + Online Cross-Entropy (Fixes Issue #1693):**
   - Eliminate materialization of the full `[B, S, V]` logits tensor in both standard training and `StreamedLargeLayer`.
   - Process hidden states in token chunks (`CHUNK_N = 256` or `512`), compute chunk logits in `fp16`, run a Triton online logsumexp + cross-entropy kernel in `fp32`, and overwrite the chunk logits buffer **in-place** with `dL/dz` gradients.
2. **Fused Non-Matmul Backbone Ops (`RMSNorm`, `RoPE`, `SwiGLU`):**
   - Fuse residual addition + RMSNorm in a single Triton pass (store `rstd` for backward).
   - Apply Grouped-Query Attention (GQA) Rotary Position Embeddings (`RoPE`) **in-place** for `Q` and `K` in one kernel launch, supporting custom `position_ids` for `multipack`. Use conjugate rotation (`-sin`) in backward to save 0 activation memory.
   - Implement activation-recomputing `SwiGLU` backward so `silu(gate) * up` does not need to be stored alongside `gate` and `up`.
3. **Fast NF4 Double-Dequantization + LoRA Fusion:**
   - Avoid double global-memory dequantization of `bitsandbytes` `Params4bit` across forward and backward passes, or fuse NF4 dequantization with `lora_B @ lora_A` adapter addition inside SRAM before GEMM.
4. **Turing `sm_75` FlashAttention with Block-Diagonal `doc_ids` Skipping:**
   - Provide a Triton FlashAttention kernel tuned for `sm_75` (`64x64` tiles) that accepts 1D per-token document IDs (`doc_ids`) for `multipack`, skipping tiles where `doc_max_k < doc_min_q` without allocating a dense `[B, 1, S, S]` mask.

---

## 5. Quality & Verification Gates (Non-Negotiable)
- **Numerical Equivalence:** Never approximate layers in ways that degrade training loss.
  - All reductions (`sum`, `max`, `logsumexp`, `rstd`) MUST execute in `tl.float32`.
  - Loss reductions across chunks MUST accumulate in `torch.float64` or `torch.float32` and divide by exact valid token count (`labels != -100`).
- **Unit Testing Requirement:** Every new Triton kernel MUST have a corresponding test in `tests/test_triton_kernels.py` verifying forward output and backward gradients (`dX`, `dW`) against reference PyTorch implementations using `torch.testing.assert_close(atol=1e-2, rtol=1e-2)` for `fp16`.
- **Commands to run tests:**
  - `pytest tests/test_triton_kernels.py -v`
  - `pytest tests/ -k "not slow"`

---

## 6. Engineering Rules for AI Agents (Codex & Claude Code)
1. **No Breaking Changes:** Never break existing YAML configuration keys in `soup.yaml`. All Triton optimizations must be toggleable via config flags (e.g., `training.triton_kernels: true`) or safely auto-detected.
2. **Graceful Fallbacks:** If Triton is not installed or the GPU compute capability is less than `sm_75` (e.g., CPU or older GPU), the code must automatically fall back to standard PyTorch/Hugging Face implementations without crashing.
3. **Clean Code & Typing:** Write clean, modular Python code with type hints (`mypy` compatible) and clear docstrings explaining the math and memory layout.