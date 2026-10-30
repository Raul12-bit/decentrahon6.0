"""Our Triton kernels, one module per fused operation (all fp32 math, fp16 storage on a T4).

    rmsnorm.py        residual add + RMSNorm, forward and backward
    rope.py           LoRA add + RoPE + scatter to SDPA's padded layout, forward and backward
    swiglu.py         silu(gate) * up in fp16, backward in place (not counted as our own)
    cross_entropy.py  lm_head + cross-entropy + TRL/rewind metrics in token chunks (no logits)
    nf4.py            NF4 double-quant unpacking in one pass (+ the 4-bit gate/up/down matmuls)

`solution.patches.patch_qwen` wires them into Soup's training forward; `solution.apply()`
installs it.
"""
