"""Quick A/B on a T4: the first N steps of the task, base Soup vs our solution.

    python task/prepare_data.py                       # once: data/train.jsonl
    python benchmarks/verify_t4_speedup.py            # 15 steps each side
    python benchmarks/verify_t4_speedup.py --steps 30 --out runs_bench

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
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))


# ------------------------------------------------------------------------------------- worker


def _worker(args) -> None:
    """One side, inside its own process: `soup train` stopped after args.steps steps."""
    sys.argv = ["soup", "train", "--config", args.config, "--yes"]
    sys.modules["unsloth"] = None  # as harness/_train.py: keep Soup from importing it
    sys.modules["unsloth_zoo"] = None
    import torch

    torch.empty(1, device="cuda")
    import transformers

    applied = None
    if args.side == "solution":
        sys.path.insert(0, str(REPO))
        import solution

        applied = solution.apply()

    record = {"side": args.side, "applied": applied, "step_s": [], "logs": []}

    class Probe(transformers.TrainerCallback):
        def on_step_begin(self, train_args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.start = time.perf_counter()

        def on_step_end(self, train_args, state, control, **kwargs):
            torch.cuda.synchronize()
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
        record["exit_code"] = 0 if exc.code is None else exc.code
    record["wall_s"] = round(time.perf_counter() - started, 2)
    record["max_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    record["max_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    Path(args.result).write_text(json.dumps(record, indent=2), encoding="utf-8")


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


def _run_side(side: str, args, out: Path) -> dict:
    from common import OFFLINE_ENV, run_logged  # the harness's env and nvidia-smi poller

    run_dir = out / side
    run_dir.mkdir(parents=True, exist_ok=True)
    config = _make_config(Path(args.train_file).resolve(), run_dir, args.seed)
    result = run_dir / "bench.json"
    result.unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", side, "--config",
           str(config), "--steps", str(args.steps), "--result", str(result)]
    env = dict(OFFLINE_ENV, SOUP_DB_PATH=str(run_dir / "experiments.db"),
               TRITON_CACHE_DIR=str(run_dir / "triton_cache"))
    print(f"[bench] {side}: training {args.steps} steps ...", flush=True)
    code, wall, peak, start = run_logged(cmd, run_dir / "train.log", args.timeout, env)
    data = json.loads(result.read_text(encoding="utf-8")) if result.exists() else {}
    data.update(process_exit=code, process_wall_s=wall, nvidia_smi_peak_mib=peak,
                gpu_start_mib=start)
    if code != 0 or not data.get("logs"):
        raise SystemExit(f"[bench] {side} failed (exit {code}), see {run_dir / 'train.log'}")
    return data


def _summary(data: dict) -> dict:
    timed = data["step_s"][2:]
    tokens = [e["num_tokens"] for e in data["logs"]]
    per_step = [b - a for a, b in zip([0.0] + tokens[:-1], tokens)]
    timed_tokens = sum(per_step[2:len(data["step_s"])])
    return {
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
    n = min(len(base["logs"]), len(sol["logs"]))
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
    parser.add_argument("--worker", choices=("base", "solution"), help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        args.side = args.worker
        _worker(args)
        return
    if not Path(args.train_file).exists():
        raise SystemExit(f"{args.train_file} is missing: run `python task/prepare_data.py` first")
    out = Path(args.out or tempfile.mkdtemp(prefix="soup_bench_")).resolve()
    runs = {side: _run_side(side, args, out) for side in ("base", "solution")}
    rows = {side: _summary(data) for side, data in runs.items()}
    print("\n=== first", args.steps, "steps ===")
    for side, row in rows.items():
        print(f"{side:9s} " + "  ".join(f"{k} {v}" for k, v in row.items()))
    b, s = rows["base"], rows["solution"]
    if b["tokens_per_s"] and s["tokens_per_s"]:
        print(f"speed (tokens/s, steps 3+): {s['tokens_per_s'] / b['tokens_per_s']:.3f}x")
    print(f"peak memory, nvidia-smi: {b['nvidia_smi_peak_mib'] / s['nvidia_smi_peak_mib']:.3f}x "
          f"lower; torch max allocated: {b['max_allocated_mib'] / s['max_allocated_mib']:.3f}x")
    print("solution patches:", runs["solution"].get("applied"))
    problems = _compare_curves(runs["base"], runs["solution"])
    (out / "summary.json").write_text(json.dumps({"rows": rows, "problems": problems,
                                                  "runs": runs}, indent=2), encoding="utf-8")
    print(f"written: {out / 'summary.json'}")
    if problems:
        print("CURVES DIFFER:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    print("curves match (assert_close + harness limits)")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
