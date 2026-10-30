"""Dependency-free inventory and static checks, with machine-readable evidence."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "harness"))
from check_rules import source_kernels

files = sorted(root.rglob("*.py"))
issues = []
for file in files:
    try:
        compile(file.read_text(encoding="utf-8"), str(file), "exec")
    except Exception as exc:
        issues.append(f"{file.relative_to(root)}: {exc}")
own, copies, lookalikes, banned, broken = source_kernels(root / "solution")
result = {
    "platform": platform.platform(), "python": sys.version,
    "python_files_compiled": len(files), "syntax_errors": issues,
    "source_kernel_candidates": own, "example_kernels": copies,
    "lookalikes": lookalikes, "banned_imports": banned, "broken_sources": broken,
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "protected_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for folder in ("task", "harness", "tests")
                         for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                         and p.suffix != ".pyc"},
}
(root / "reports" / "inventory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
assert not issues and not banned and not broken
