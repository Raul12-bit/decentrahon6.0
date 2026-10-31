"""Quick A/B on a T4: the first N steps of the task, base Soup vs our solution.

    python task/prepare_data.py                       # once: data/train.jsonl
    python benchmarks/verify_t4_speedup.py            # 15 steps each side
    python benchmarks/verify_t4_speedup.py --steps 30 --out runs_bench
    python benchmarks/verify_t4_speedup.py --matrix   # base + every solution variant below

With --matrix (or --variant NAME=ENV=VALUE,ENV=VALUE, repeatable) the solution side runs once
per variant, each in its own process with those SOUP_KERNELS_* switches, and the script prints
the experiment table: speed, peak memory, loss deviation from base and the share of our own
kernels in steps 3+ (timed with CUDA events by the harness's own TritonWatch from
harness/_train.py, imported, not copied, the way check_rules.py reads it).

Each side runs in its own process (the patches are process-wide), with the harness's
environment (GPU 0, `expandable_segments` allocator, offline Hugging Face), the task config and
data, stopped after N optimizer steps. It reports, per side:

  - steady-state speed: tokens/s over steps 3..N (the first two compile and warm up);
  - peak memory: torch.cuda.max_memory_allocated / max_memory_reserved inside the process and
    the nvidia-smi peak sampled from outside every 100 ms (what the jury scores);
  - the loss curve, entropy and token accuracy that Soup logs at every step;

then checks the curves: torch.testing.assert_close per step, and the harness's own criteria
(first step within 1%, mean relative gap within 3%). This is a development tool; the numbers that
count come from harness/run_task.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))

# The experiment matrix of --matrix: one optimization switched off at a time against the
# defaults (all on), plus the previous phase's configuration.
MATRIX = {
    "default": {},  # phase 4: every switch on
    "no_packed_masks": {"SOUP_KERNELS_PACKED_MASKS": "0"},
    "no_resid_delta": {"SOUP_KERNELS_RESID_DELTA": "0"},
    "phase3": {"SOUP_KERNELS_PACKED_MASKS": "0", "SOUP_KERNELS_RESID_DELTA": "0"},
    "no_varlen_attn": {"SOUP_KERNELS_VARLEN_ATTN": "0"},
    "no_fused_mlp_norm": {"SOUP_KERNELS_FUSED_MLP_NORM": "0"},
    "no_lora_remat": {"SOUP_KERNELS_LORA_REMAT": "0"},
    "no_mlp_recompute": {"SOUP_KERNELS_MLP_RECOMPUTE": "0"},
    "phase1": {"SOUP_KERNELS_VARLEN_ATTN": "0", "SOUP_KERNELS_FUSED_MLP_NORM": "0",
               "SOUP_KERNELS_LORA_REMAT": "0", "SOUP_KERNELS_PACKED_MASKS": "0",
               "SOUP_KERNELS_RESID_DELTA": "0"},
}


# ------------------------------------------------------------------------------------- worker


def _worker(args) -> None:
    """One side, inside its own process: `soup train` stopped after args.steps steps."""
    sys.argv = ["soup", "train", "--config", args.config, "--yes"]
    sys.modules["unsloth"] = None  # as harness/_train.py: keep Soup from importing it
    sys.modules["unsloth_zoo"] = None
    import torch

    torch.empty(1, device="cuda")
    import transformers

    from _train import TIMED_FROM_STEP, TritonWatch  # the harness's own launch timer

    watch = TritonWatch(torch)
    applied = None
    if args.side == "solution":
        sys.path.insert(0, str(REPO))
        import solution

        applied = solution.apply()

    record = {"side": args.side, "applied": applied, "step_s": [], "logs": [], "exit_code": 0}

    class Probe(transformers.TrainerCallback):
        def on_step_begin(self, train_args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.start = time.perf_counter()
            watch.timing = len(record["step_s"]) >= TIMED_FROM_STEP

        def on_step_end(self, train_args, state, control, **kwargs):
            torch.cuda.synchronize()
            watch.timing = False
            record["step_s"].append(time.perf_counter() - self.start)
            if state.global_step >= args.steps:
                control.should_training_stop = True

        def on_log(self, train_args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                record["logs"].append({k: logs.get(k) for k in (
                    "loss", "entropy", "mean_token_accuracy", "num_tokens", "grad_norm")})

    original_init = transformers.Trainer.__init__

    def init(self, *init_args, **init_kwargs):
        original_init(self, *init_args, **init_kwargs)
        self.add_callback(Probe())

    transformers.Trainer.__init__ = init
    started = time.perf_counter()
    try:
        from soup_cli.cli import run

        run()
    except SystemExit as exc:
        record["exit_code"] = 0 if exc.code is None else (exc.code if isinstance(exc.code, int) else 1)
    record["wall_s"] = round(time.perf_counter() - started, 2)
    record["max_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    record["max_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    record["own_kernel_ms"] = watch.own_ms()
    record["timed_loop_s"] = round(sum(record["step_s"][TIMED_FROM_STEP:]), 3)
    record["triton_launches"] = dict(watch.launches)
    Path(args.result).write_text(json.dumps(record, indent=2), encoding="utf-8")
    raise SystemExit(record["exit_code"])


# --------------------------------------------------------------------------------------- main


def _make_config(train_file: Path, out_dir: Path, seed: int) -> Path:
    import yaml

    config = yaml.safe_load((REPO / "task" / "task.yaml").read_text(encoding="utf-8"))
    config["data"]["train"] = str(train_file)
    config["training"]["seed"] = seed
    config["output"] = str(out_dir / "adapter")
    path = out_dir / "task.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _run_side(side: str, args, out: Path, name: str = None, extra_env: dict = None) -> dict:
    from common import OFFLINE_ENV, run_logged  # the harness's env and nvidia-smi poller

    import shutil

    name = name or side
    run_dir = out / name
    # A fresh directory, as run_task.py does: a Triton cache left by an earlier run with the
    # same --out would hide the compilation time this run must pay.
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    config = _make_config(Path(args.train_file).resolve(), run_dir, args.seed)
    result = run_dir / "bench.json"
    result.unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", side, "--config",
           str(config), "--steps", str(args.steps), "--result", str(result)]
    # A fresh Triton cache per run, as the harness does: compilation is part of every run.
    # The harness's env copies os.environ, so a SOUP_KERNELS_* switch set in the notebook would
    # leak into every variant (and mislabel "default"): drop them, then apply this variant's.
    env = {key: value for key, value in OFFLINE_ENV.items()
           if not key.startswith("SOUP_KERNELS")}
    env.update(SOUP_DB_PATH=str(run_dir / "experiments.db"),
               TRITON_CACHE_DIR=str(run_dir / "triton_cache"), **(extra_env or {}))
    print(f"[bench] {name}: training {args.steps} steps {extra_env or ''} ...", flush=True)
    code, wall, peak, start = run_logged(cmd, run_dir / "train.log", args.timeout, env)
    data = json.loads(result.read_text(encoding="utf-8")) if result.exists() else {}
    data.update(process_exit=code, process_wall_s=wall, nvidia_smi_peak_mib=peak,
                gpu_start_mib=start)
    if code != 0 or data.get("exit_code") != 0 or not data.get("logs"):
        raise SystemExit(f"[bench] {name} failed (exit {code}), see {run_dir / 'train.log'}")
    if len(data.get("step_s", [])) != args.steps or len(data["logs"]) != args.steps:
        raise SystemExit(f"[bench] {name} did not complete and log all {args.steps} requested "
                         f"steps; see {run_dir / 'train.log'}")
    return data


def _kernel_share(data: dict):
    """(share of all our kernels, share without the SwiGLU ones) of steps 3+, as check 4 reads
    it: kernel ms / (1000 * timed loop s)."""
    loop = data.get("timed_loop_s") or 0
    own = data.get("own_kernel_ms") or {}
    if not loop or not own:
        return None, None
    total = sum(own.values())
    no_swiglu = sum(ms for name, ms in own.items() if "swiglu" not in name)
    return round(100 * total / (1000 * loop), 2), round(100 * no_swiglu / (1000 * loop), 2)


def _summary(data: dict) -> dict:
    timed = data["step_s"][2:]
    tokens = [e["num_tokens"] for e in data["logs"]]
    per_step = [b - a for a, b in zip([0.0] + tokens[:-1], tokens)]
    timed_tokens = sum(per_step[2:len(data["step_s"])])
    share, share_no_swiglu = _kernel_share(data)
    return {
        "own_kernel_share_pct": share,
        "share_without_swiglu_pct": share_no_swiglu,
        "steps": len(data["step_s"]),
        "tokens_per_s": round(timed_tokens / sum(timed), 1) if timed else None,
        "median_step_s": round(sorted(timed)[len(timed) // 2], 4) if timed else None,
        "max_allocated_mib": data["max_allocated_mib"],
        "max_reserved_mib": data["max_reserved_mib"],
        "nvidia_smi_peak_mib": data["nvidia_smi_peak_mib"],
        "process_wall_s": data["process_wall_s"],
    }


def _compare_curves(base: dict, sol: dict) -> list[str]:
    import torch

    problems = []
    if not base.get("logs") or not sol.get("logs"):
        return ["empty training curves"]
    if len(base["logs"]) != len(sol["logs"]):
        return ["training curve lengths differ: the runs did not complete the same steps"]
    n = len(base["logs"])
    for side, data in (("base", base), ("solution", sol)):
        for row in data["logs"]:
            for key in ("loss", "entropy", "mean_token_accuracy", "num_tokens"):
                value = row.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    problems.append(f"{side}: missing or non-finite {key}")
    if problems:
        return problems
    for key, rtol in (("loss", 1e-2), ("entropy", 3e-2), ("mean_token_accuracy", 3e-2)):
        b = torch.tensor([e[key] for e in base["logs"][:n]], dtype=torch.float64)
        s = torch.tensor([e[key] for e in sol["logs"][:n]], dtype=torch.float64)
        try:
            torch.testing.assert_close(s, b, rtol=rtol, atol=1e-6)
        except AssertionError as exc:
            problems.append(f"{key}: {str(exc).splitlines()[0]}")
        gap = ((s - b).abs() / b.abs().clamp(min=1e-12)).mean().item()
        print(f"  {key:20s} mean relative gap {100 * gap:6.3f}%  first step "
              f"{100 * abs(s[0] - b[0]).item() / max(abs(b[0].item()), 1e-12):6.3f}%")
        if key == "loss" and (abs(s[0] - b[0]) > 0.01 * abs(b[0]) or gap > 0.03):
            problems.append("loss curve outside the harness limits (1% first step, 3% mean)")
    tokens_b = [e["num_tokens"] for e in base["logs"][:n]]
    tokens_s = [e["num_tokens"] for e in sol["logs"][:n]]
    if tokens_b != tokens_s:
        problems.append("num_tokens differ: the batches are not the same")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--train-file", default=str(REPO / "data" / "train.jsonl"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None, help="directory for logs (default: a temp dir)")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--matrix", action="store_true",
                        help="run every solution variant of MATRIX (one switch off at a time)")
    parser.add_argument("--variant", action="append",
                        help="NAME=ENV=VALUE[,ENV=VALUE...]: a solution run with these switches")
    parser.add_argument("--worker", choices=("base", "solution"), help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        args.side = args.worker
        _worker(args)
        return
    if args.steps < 3:
        parser.error("--steps must be at least 3 (two warmup steps and one timed step)")
    if not Path(args.train_file).exists():
        raise SystemExit(f"{args.train_file} is missing: run `python task/prepare_data.py` first")
    variants = dict(MATRIX) if args.matrix else {}
    for spec in args.variant or []:
        name, _, assignments = spec.partition("=")
        variants[name] = dict(item.split("=", 1) for item in assignments.split(",") if item)
    if not variants:
        variants = {"solution": {}}

    out = Path(args.out or tempfile.mkdtemp(prefix="soup_bench_")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # The runs are offline (HF_HUB_OFFLINE=1, as in the harness): download the model and read
    # the packages once first, with the harness's own warm-up, so no run fails or pays for it.
    from run_task import warm_up

    warm_up(out)
    runs = {"base": _run_side("base", args, out)}
    for name, env in variants.items():
        runs[name] = _run_side("solution", args, out, name, env)
    rows = {name: _summary(data) for name, data in runs.items()}
    problems = {name: _compare_curves(runs["base"], data)
                for name, data in runs.items() if name != "base"}

    b = rows["base"]
    print(f"\n=== first {args.steps} steps (speed: tokens/s over steps 3+; memory: nvidia-smi peak) ===")
    header = (f"{'run':20s} {'tok/s':>8s} {'speed':>7s} {'peak MiB':>9s} {'mem x':>7s} "
              f"{'alloc MiB':>10s} {'wall s':>7s} {'loss gap':>9s} {'1st gap':>8s} "
              f"{'own %':>6s} {'-swiglu':>8s}  curves")
    print(header)
    for name, row in rows.items():
        gaps = _loss_gaps(runs["base"], runs[name])
        speed = (row["tokens_per_s"] / b["tokens_per_s"]
                 if row["tokens_per_s"] and b["tokens_per_s"] else float("nan"))
        mem = b["nvidia_smi_peak_mib"] / row["nvidia_smi_peak_mib"]
        verdict = "-" if name == "base" else ("ok" if not problems[name] else "DIFFER")
        print(f"{name:20s} {row['tokens_per_s'] or 0:8.1f} {speed:7.3f} "
              f"{row['nvidia_smi_peak_mib']:9.0f} {mem:7.3f} {row['max_allocated_mib']:10.0f} "
              f"{row['process_wall_s']:7.1f} {gaps[0]:8.3f}% {gaps[1]:7.3f}% "
              f"{_fmt(row['own_kernel_share_pct']):>6s} {_fmt(row['share_without_swiglu_pct']):>8s}"
              f"  {verdict}")
    for name, data in runs.items():
        if name != "base":
            print(f"{name}: patches {data.get('applied')}")
    (out / "summary.json").write_text(json.dumps({"variants": variants, "rows": rows,
                                                  "problems": problems, "runs": runs},
                                                 indent=2), encoding="utf-8")
    print(f"written: {out / 'summary.json'}")
    failing = {name: p for name, p in problems.items() if p}
    if failing:
        for name, p in failing.items():
            print(f"CURVES DIFFER ({name}):\n  " + "\n  ".join(p))
        raise SystemExit(1)
    print("curves match (assert_close + harness limits)")


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.1f}"


def _loss_gaps(base: dict, run: dict) -> tuple:
    """(mean relative loss gap %, first-step gap %) against base, as check 6 computes them."""
    b = [e["loss"] for e in base["logs"]]
    s = [e["loss"] for e in run["logs"]]
    if not b or len(b) != len(s):
        return float("nan"), float("nan")
    mean = sum(abs(x - y) / max(abs(y), 1e-12) for x, y in zip(s, b)) / len(b)
    return 100 * mean, 100 * abs(s[0] - b[0]) / max(abs(b[0]), 1e-12)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
