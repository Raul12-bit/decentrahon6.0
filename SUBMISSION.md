# Submission: DoStar

## Links
- Repository: https://github.com/Raul12-bit/decentrahon6.0 (public).
- Final commit: the last commit of `main`,
  https://github.com/Raul12-bit/decentrahon6.0/commits/main. Its solution code is the code measured
  on a Kaggle T4 in https://github.com/Raul12-bit/decentrahon6.0/commit/4e000cd: later commits change
  documentation and `assets/` only, `solution/` is byte-identical (sha256 of `solution/`, computed
  like `solution_sha256` in `runs/checks.json`:
  `4028fbb20de57b0ac25f91f80deb91b3cd14da4042b7946e24b751272b23efed`).
- Public Kaggle notebook, the saved version of `templates/kaggle_notebook.ipynb` with `COMMIT` =
  `b55de11` (later commits change documentation only; the same link is in the README):
  https://www.kaggle.com/code/sayatsaytli/soup-kernel-challenge-t4-final. The earlier public
  notebook of the phase 3 code (commit `b21090d`):
  https://www.kaggle.com/code/ff3923/soup-kernel-challenge-t4-submission.
- Demo video (up to 3 minutes, 2:50, Russian narration with subtitles), plays in the browser:
  https://cdn.jsdelivr.net/gh/Raul12-bit/decentrahon6.0@211a969/assets/soup_demo.mp4 (the same
  file in the repository: `assets/soup_demo.mp4`, subtitles `assets/soup_demo.srt`)
- Presentation (PDF): https://drive.google.com/file/d/1cyQWeTlgyoH4tdQYNCG_2kvRQGc-Tc1I/view?usp=sharing

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (Telegram) |
|---|---|---|---|---|
| Bayrzhan Tanatar | Captain | [baurzhanx22](https://github.com/baurzhanx22) | | — |
| Almen Magzhan | AI Engineer | [magasigmaff39](https://github.com/magasigmaff39) | | [@magzhan2311](https://t.me/magzhan2311) |
| Sayat Mukhamedzhanov | Machine Learning | [lisayatsayt-byte](https://github.com/lisayatsayt-byte) | [sayatsaytli](https://www.kaggle.com/sayatsaytli) | [@sosossyat](https://t.me/sosossyat) |
| Ni Raul | Software Engineer | [Raul12-bit](https://github.com/Raul12-bit) | | [@rnsq34](https://t.me/rnsq34) |

The captain has no Telegram or e-mail: please reach the team through the members' Telegram above.

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `_cross_entropy_fwd_bwd_inplace_kernel` | `solution/kernels/cross_entropy.py` | `[B,S,V]` logits, ForCausalLMLoss (fp32 upcast + log_softmax + backward), TRL entropy / accuracy, rewind row losses | together with the rows below: **18.1%** (check 4, lowest of 2 runs); without SwiGLU ~14% |
| `_nf4_dequant_nested_kernel` | `solution/kernels/nf4.py` | bitsandbytes absmax dequant + offset add + NF4 dequant (3 kernels), forward and backward; bit-exact | (in the total above) |
| `_add_rmsnorm_fwd_kernel`, `_add_rmsnorm_bwd_kernel` | `solution/kernels/rmsnorm.py` | residual add + Qwen2RMSNorm (fp32 eager chain); fp16 output fused into the MLP; rebuilds the LoRA input in backward | (in the total above) |
| `_rope_lora_scatter_fwd_kernel`, `_rope_gather_bwd_kernel` | `solution/kernels/rope.py` | peft LoRA add + apply_rotary_pos_emb (fp32 eager chain) + fp16 cast + pad scatter / gather, per-row layout | (in the total above) |
| `_dropout_mask_pack_kernel`, `_dropout_bits_bwd_kernel` | `solution/kernels/lora.py` | LoRA dropout keep masks stored as bits (8 per byte) instead of bool, unpacked with dropout's scale in backward (phase 4) | (in the total above) |
| `_swiglu_fp16_fwd_kernel`, `_swiglu_fp16_bwd_inplace_kernel` | `solution/kernels/swiglu.py` | fp32 silu * up; a variant of the starter kit's example (check_rules: 0.81 similar), **not counted** by us | (counted by the harness: about 3.8 of the 18.1 points) |

Other changes in `solution/patches/patch_qwen.py` (no kernel of their own): pad tokens removed
inside the batch, causal SDPA per row on the real tokens, LoRA input rebuilt in backward
(`kernels/lora.py`), gate/up recomputed in backward, the MLP keeps the fp16 attention output instead
of the fp32 residual, the tied embedding table stored once in fp16 (and kept in fp16 after
training), kernels compiled while the model loads (`warmup.py`). Per-row attention can read k/v
without expanding them (`enable_gqa`) on sm_80+ GPUs; on a T4 a check on the device turns that off,
because PyTorch's only fused SDPA kernel there (memory-efficient) does not take grouped k/v heads.

Ideas taken from other work, with links (we wrote the code ourselves):
- Online softmax: Milakov & Gimelshein 2018, https://arxiv.org/abs/1805.02867
- Chunked lm_head + loss without full logits: Cut Cross-Entropy, https://arxiv.org/abs/2411.09009,
  and Soup's own `stream_loss_chunk` (#1693)
- NF4 / double quantization: QLoRA, https://arxiv.org/abs/2305.14314 (bitsandbytes format)
- Attention without pad tokens (varlen): FlashAttention-2, https://arxiv.org/abs/2307.08691 (we
  call PyTorch's SDPA per row)
- Recompute instead of store: Chen et al. 2016, https://arxiv.org/abs/1604.06174 (Soup's built-in
  `gradient_checkpointing` stays off)
- Fused residual + RMSNorm, RoPE backward by the conjugate rotation, fused linear cross-entropy:
  similar techniques exist in Liger-Kernel (https://github.com/linkedin/Liger-Kernel) and Unsloth
  (https://github.com/unslothai/unsloth); none of their code is used or imported

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json`)
Final code, as the jury runs it: `python harness/run_task.py --repeats 2` (base, solution,
solution, base), `check_rules.py`, `score.py`; Kaggle GPU T4 x2, commit 4e000cd, public rows,
2026-10-31 (a private team notebook made from the same template).

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s (median of 2) | 84.83 | 60.69 | speedup x1.398 |
| Peak GPU memory, MiB (nvidia-smi, max of 2) | 14345 | 2479 | base/ours x5.787 |
| Eval loss (public rows) | 2.04967 | 2.04965 | -0.001% vs base |

The public notebook above (official template, `run_task.py` with 1 repeat, another Kaggle session,
commit b55de11 with the same `solution/`): base 83.58 s / 14345 MiB, ours 64.63 s / 2477 MiB,
speedup x1.293, memory x5.791, eval loss -0.002% vs base; all 10 checks passed (own kernels 17.9%).

Earlier measurements of the phase 3 code (commit 7817156, the same `solution/` as b21090d):

| | Protocol | Base, s | Ours, s | Speedup | Peak, MiB (base / ours) | Memory | Eval loss vs base |
|---|---|---|---|---|---|---|---|
| Official run | `--repeats 2` | 83.08 | 59.58 | x1.394 | 14345 / 2847 | x5.039 | +0.000% |
| Public notebook (b21090d) | 1 repeat | 88.49 | 61.96 | x1.428 | 14345 / 2847 | x5.039 | -0.001% |

- [x] All 10 checks of `harness/check_rules.py` passed on the final code (`ALL CHECKS PASSED`; loss
  curve: first step 0.03%, mean 0.01%; own kernels 18.1% of the training time; time x0.715 and
  memory x0.173 of the base; 79 Soup tests passed; `pytest solution` on the T4: 23 passed), and in
  both phase 3 measurements
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why, how the kernels work, results against the base (speed,
memory, quality), limits and next steps.

## Limitations
- The whole command speeds up less than a training step (x1.40 against x1.81): ~25 s of Python
  start, model loading / 4-bit quantization and adapter saving are not touched.
- A step is now bound by cuBLAS GEMMs, which we did not replace.
- On a T4, k/v of the per-row attention stay expanded to the query heads (6 KiB per token per layer
  instead of 1); keeping them unexpanded there would need our own wrapper around memory-efficient
  attention.
- The fast path covers this task's configuration (Qwen2, NF4 + double quant, plain LoRA, SDPA, fp16
  autocast); anything else falls back to plain Soup.
- The SwiGLU kernel is a variant of the example and is not counted as ours.
