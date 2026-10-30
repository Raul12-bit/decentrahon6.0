"""Run one bounded audit command and preserve its real output and exit code."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "reports" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
name, *command = sys.argv[1:]
started = time.perf_counter()
env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
           TOKENIZERS_PARALLELISM="false", SOUP_NO_AUDIT_LOG="1",
           PYTHONPATH=str(ROOT))
try:
    with (LOGS / f"{name}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=900, check=False)
    code = result.returncode
    output = (LOGS / f"{name}.log").read_text(encoding="utf-8", errors="replace")
except subprocess.TimeoutExpired as exc:
    code = 124
    output = (LOGS / f"{name}.log").read_text(encoding="utf-8", errors="replace") + "\nAUDIT TIMEOUT 900s\n"
record = {"name": name, "command": command, "exit_code": code,
          "wall_seconds": time.perf_counter() - started}
(LOGS / f"{name}.log").write_text(output, encoding="utf-8")
(LOGS / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record))
print(output[-6000:])
sys.exit(code)
