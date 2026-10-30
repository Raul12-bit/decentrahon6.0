# Submission: <team name>

## Links
- Repository, final commit: https://github.com/<you>/<repo>/commit/<sha>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link must
  be in your README):
  https://www.kaggle.com/code/<you>/<notebook>?scriptVersionId=<version>
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `_cross_entropy_fwd_bwd_inplace_kernel` | `solution/kernels/cross_entropy.py` | `[B,S,V]` logits, ForCausalLMLoss (fp32 upcast + log_softmax + backward), TRL entropy / accuracy, rewind row losses | % |
| `_nf4_dequant_nested_kernel` | `solution/kernels/nf4.py` | bitsandbytes absmax dequant + offset add + NF4 dequant (3 kernels), forward and backward | % |
| `_add_rmsnorm_fwd_kernel`, `_add_rmsnorm_bwd_kernel` | `solution/kernels/rmsnorm.py` | residual add + Qwen2RMSNorm (fp32 eager chain) | % |
| `_rope_lora_scatter_fwd_kernel`, `_rope_gather_bwd_kernel` | `solution/kernels/rope.py` | peft LoRA add + apply_rotary_pos_emb (fp32 eager chain) + pad scatter / gather | % |
| `_swiglu_fp16_fwd_kernel`, `_swiglu_fp16_bwd_inplace_kernel` | `solution/kernels/swiglu.py` | fp32 silu * up; a variant of the starter kit's example, **not counted** as our own | (not counted) |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Online softmax: Milakov & Gimelshein 2018, https://arxiv.org/abs/1805.02867
- Chunked lm_head + loss without full logits: Cut Cross-Entropy, https://arxiv.org/abs/2411.09009,
  and Soup's own `stream_loss_chunk` (#1693)
- NF4 / double quantization: QLoRA, https://arxiv.org/abs/2305.14314 (bitsandbytes format)

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | | | speedup x |
| Peak GPU memory, MiB | | | base/ours x |
| Eval loss (public rows) | | | % vs base |

- [ ] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations you sped up and why, how the kernels work, results against the base (speed,
memory, quality), limits and next steps.

## Limitations
See README.md, "Ограничения и что дальше".
