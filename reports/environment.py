"""Capture the actual runtime, separate from the NVIDIA driver's advertised CUDA version."""
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import psutil
import torch

names = ("soup-cli", "torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes",
         "triton", "datasets", "pytest", "ruff", "mypy")
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,memory.free,driver_version",
                      "--format=csv"], capture_output=True, text=True, check=False)
result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
          "versions": versions, "torch_cuda_runtime": torch.version.cuda,
          "torch_cuda_available": torch.cuda.is_available(),
          "nvidia_smi_exit": gpu.returncode, "nvidia_smi": gpu.stdout,
          "ram_total_bytes": psutil.virtual_memory().total,
          "ram_available_bytes": psutil.virtual_memory().available,
          "cpu": platform.processor(), "cpu_logical_cores": psutil.cpu_count()}
Path("reports/environment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
