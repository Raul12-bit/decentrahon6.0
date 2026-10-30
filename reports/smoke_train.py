"""Bounded real Soup CLI training + checkpoint resume on a local tiny Qwen2.

CPU/fp32 fallback only. No downloaded model, synthetic performance claims or CUDA claims.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "reports" / "artifacts" / "cpu_smoke"
ART.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(2026)
started = time.perf_counter()
words = ["[PAD]", "[UNK]", "[EOS]", "user", "assistant", ":", "say", "hello", "world", "yes"]
tok = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
tok.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                   eos_token="[EOS]", model_max_length=64)
tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ' : ' + m['content'] + ' ' }}{% endfor %}{{ eos_token }}"
base = ART / "tiny_qwen2"
tokenizer.save_pretrained(base)
cfg = Qwen2Config(vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                  tie_word_embeddings=True, pad_token_id=0, eos_token_id=2)
model = Qwen2ForCausalLM(cfg)
model.save_pretrained(base)
model_setup_s = time.perf_counter() - started
del model
data = ART / "train.jsonl"
data.write_text("".join(json.dumps({"instruction": "say hello", "input": "",
                                  "output": "hello world" if i % 2 else "yes hello"}) + "\n"
                        for i in range(8)), encoding="utf-8")
output = ART / "adapter"
config = {
    "base": str(base), "task": "sft",
    "data": {"train": str(data), "format": "alpaca", "val_split": 0., "max_length": 64},
    "training": {"epochs": 1, "lr": .001, "batch_size": 2, "gradient_accumulation_steps": 1,
                 "seed": 2026, "logging_steps": 1, "save_steps": 2, "quantization": "none",
                 "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]}},
    "output": str(output),
}
path = ART / "soup.yaml"
path.write_text(yaml.safe_dump(config), encoding="utf-8")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
           SOUP_DB_PATH=str(ART / "experiments.db"), SOUP_NO_AUDIT_LOG="1", PYTHONUTF8="1",
           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
# The same bootstrap as the challenge: apply() then the genuine Soup CLI in the same process.
bootstrap = "import solution; solution.apply(); from soup_cli.cli import run; run()"
command = [sys.executable, "-c", bootstrap, "train", "--config", str(path), "--yes"]

def run(name, extra=(), expected=0):
    t = time.perf_counter()
    p = subprocess.run(command + list(extra), cwd=ROOT, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=240)
    (ROOT / "reports" / "logs" / f"{name}.log").write_text(p.stdout + p.stderr, encoding="utf-8")
    item = {"command": command + list(extra), "exit_code": p.returncode,
            "wall_seconds": time.perf_counter() - t}
    print(name, json.dumps(item), flush=True)
    assert p.returncode == expected, (name, (p.stdout + p.stderr)[-5000:])
    return item

results = {"scope": "CPU tiny Qwen2 + LoRA; solution fallback; NOT the fixed challenge",
           "model_setup_s": model_setup_s, "torch": torch.__version__}
results["train"] = run("smoke_train")
baseline = load_file(str(output / "adapter_model.safetensors"))
state = json.loads((output / "checkpoint-4" / "trainer_state.json").read_text())
losses = [e["loss"] for e in state["log_history"] if "loss" in e]
grad_norms = [e["grad_norm"] for e in state["log_history"] if "grad_norm" in e]
assert len(losses) == 4 and all(math.isfinite(x) for x in losses)
assert all(math.isfinite(x) and x > 0 for x in grad_norms)
assert any(torch.count_nonzero(v) for k, v in baseline.items() if "lora_B" in k)
results["losses"] = losses
results["grad_norms"] = grad_norms
results["global_step"] = state["global_step"]
results["resume"] = run("smoke_resume", ["--resume", str(output / "checkpoint-2")])
restored = load_file(str(output / "adapter_model.safetensors"))
assert baseline.keys() == restored.keys()
for name in baseline:
    torch.testing.assert_close(restored[name], baseline[name], atol=0, rtol=0)
results["resume_adapter_bit_equal"] = True
results["adapter_tensors"] = len(baseline)
results["invalid_checkpoint"] = run("smoke_bad_checkpoint", ["--resume", str(ART / "missing")], expected=1)
results["rewind_exists"] = (output / "rewind.jsonl").is_file()
assert results["rewind_exists"]
(ROOT / "reports" / "smoke_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
