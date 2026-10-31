# Submission: <team name>

## Links
- Repository, final commit: the last commit of `main`,
  https://github.com/Raul12-bit/decentrahon6.0/commits/main (the same sha is in `COMMIT` of the
  notebook's first cell, and the notebook prints it). Its solution code is the code measured
  officially in https://github.com/Raul12-bit/decentrahon6.0/commit/7817156; only README.md and
  this file changed after it.
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  the README):
  https://www.kaggle.com/code/ff3923/soup-kernel-challenge-t4-submission
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `_cross_entropy_fwd_bwd_inplace_kernel` | `solution/kernels/cross_entropy.py` | `[B,S,V]` logits, ForCausalLMLoss (fp32 upcast + log_softmax + backward), TRL entropy / accuracy, rewind row losses | together with the rows below: **17.2%** (check 4, lowest of 2 official runs); without SwiGLU ~14% (benchmark matrix) |
| `_nf4_dequant_nested_kernel` | `solution/kernels/nf4.py` | bitsandbytes absmax dequant + offset add + NF4 dequant (3 kernels), forward and backward | (in the total above) |
| `_add_rmsnorm_fwd_kernel`, `_add_rmsnorm_bwd_kernel` | `solution/kernels/rmsnorm.py` | residual add + Qwen2RMSNorm (fp32 eager chain); fp16 output fused into the MLP; rebuilds the LoRA input in backward | (in the total above) |
| `_rope_lora_scatter_fwd_kernel`, `_rope_gather_bwd_kernel` | `solution/kernels/rope.py` | peft LoRA add + apply_rotary_pos_emb (fp32 eager chain) + fp16 cast + pad scatter / gather, per-row layout | (in the total above) |
| `_swiglu_fp16_fwd_kernel`, `_swiglu_fp16_bwd_inplace_kernel` | `solution/kernels/swiglu.py` | fp32 silu * up; a variant of the starter kit's example (check_rules: 0.81 similar), **not counted** by us | (counted by the harness; ~3% of the 17.2%) |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Online softmax: Milakov & Gimelshein 2018, https://arxiv.org/abs/1805.02867
- Chunked lm_head + loss without full logits: Cut Cross-Entropy, https://arxiv.org/abs/2411.09009,
  and Soup's own `stream_loss_chunk` (#1693)
- NF4 / double quantization: QLoRA, https://arxiv.org/abs/2305.14314 (bitsandbytes format)

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json`)
`python harness/run_task.py --repeats 2` (base, solution, solution, base), `check_rules.py`,
`score.py`; Kaggle GPU T4 x2, commit 7817156, public rows.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s (median of 2) | 83.08 | 59.58 | speedup x1.394 |
| Peak GPU memory, MiB (nvidia-smi, max of 2) | 14345 | 2847 | base/ours x5.039 |
| Eval loss (public rows) | 2.04967 | 2.04967 | +0.000% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed (`ALL CHECKS PASSED`; loss curve: first
  step 0.03%, mean 0.01%; own kernels 17.2% of the training time; 79 Soup tests passed)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations you sped up and why, how the kernels work, results against the base (speed,
memory, quality), limits and next steps.

## Limitations
See README.md, "Ограничения и что дальше".
