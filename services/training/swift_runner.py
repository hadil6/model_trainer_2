"""Run SWIFT 4.x fine-tuning via subprocess and parse progress/metrics.

SWIFT 4.x CLI:
  swift sft --model <hf_model_id> --tuner_type lora --dataset <file.jsonl> ...

Key changes from SWIFT 3.x → 4.x:
  - Binary: swift.exe (not `python -m swift`)
  - --sft_type  replaced by --tuner_type
  - --quant_bits / --quant_method for quantization
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Keys that are metadata — never forwarded to the CLI
_SKIP_KEYS = {
    "swift_model_type", "quantization_required", "swift_cli_command",
    "sft_type",          # legacy key from model_selection output
    "_model_size_b", "_effective_batch", "_n_pairs",
    "lora_alpha_ratio",  # Optuna internal dimension — derived into lora_alpha, never passed to CLI
}

# Keys whose values are booleans → pass as "true"/"false" strings
_BOOL_KEYS = {"gradient_checkpointing", "load_best_model_at_end", "greater_is_better"}

# Best-checkpoint defaults applied unless the caller overrides them via hparams.
#
# Note: SWIFT 4.x CLI does NOT expose `--early_stopping_patience` /
# `--early_stopping_threshold` (HuggingFace's EarlyStoppingCallback must be
# attached programmatically, not via the CLI parser). Passing these flags causes
# SWIFT to fail with `ValueError: remaining_argv`. They are therefore omitted.
#
# Overfitting protection comes from two complementary mechanisms instead:
#   1. `load_best_model_at_end=True` + `metric_for_best_model=eval_loss` +
#      `greater_is_better=False` → SWIFT saves and loads the checkpoint with the
#      LOWEST eval_loss, regardless of how training ends. Late-training
#      overfitting therefore never reaches the exported artifact.
#   2. The orchestrator's `_detect_overfitting()` runs post-hoc on the
#      improvement_log and triggers a corrective re-training when needed.
_EARLY_STOPPING_DEFAULTS: dict = {
    "metric_for_best_model":   "eval_loss",
    "greater_is_better":       False,
    "load_best_model_at_end":  True,
}

def _tuner_type_for(peft_method: str) -> str:
    """Resolve the SWIFT tuner_type for a given peft_method dynamically."""
    from data.model_zoo.model_cards import get_peft_method_map
    cfg = get_peft_method_map().get(peft_method)
    if cfg:
        return cfg["tuner_type"]
    return "lora"  # safe fallback for unknown methods


def _swift_binary() -> str:
    """Locate the swift CLI binary installed in the current venv."""
    # Prefer the binary in the same Scripts/ dir as the current Python
    scripts = Path(sys.executable).parent
    for name in ("swift.exe", "swift.cmd", "swift"):
        candidate = scripts / name
        if candidate.exists():
            return str(candidate)
    # Fallback: PATH
    found = shutil.which("swift")
    if found:
        return found
    raise FileNotFoundError(
        "swift CLI not found. Install with: pip install ms-swift"
    )


def build_command(
    model_id: str,
    peft_method: str,
    hparams: dict,
    train_dataset: Path,
    val_dataset: Path | None,
    output_dir: Path,
) -> list[str]:
    """Return the full swift sft CLI argument list for SWIFT 4.x."""
    tuner_type = _tuner_type_for(peft_method)

    cmd: list[str] = [
        _swift_binary(), "sft",
        "--model",       model_id,
        "--tuner_type",  tuner_type,
        "--dataset",     str(train_dataset),
        "--output_dir",  str(output_dir),
    ]

    if val_dataset and val_dataset.exists():
        cmd += ["--val_dataset", str(val_dataset)]

    # Quantization — read from the dynamic PEFT method map
    from data.model_zoo.model_cards import get_peft_method_map
    peft_cfg = get_peft_method_map().get(peft_method, {})
    if "quant_bits" in peft_cfg:
        cmd += ["--quant_bits", str(peft_cfg["quant_bits"]),
                "--quant_method", peft_cfg.get("quant_method", "bnb")]

    # Merge early-stopping defaults — caller-supplied hparams take precedence
    # so the LLM hparam advisor can still override patience or the watched metric.
    merged_hparams: dict = {**_EARLY_STOPPING_DEFAULTS, **hparams}

    # Hyperparameters
    for key, val in merged_hparams.items():
        if key in _SKIP_KEYS:
            continue
        if key in _BOOL_KEYS:
            cmd += [f"--{key}", "true" if val else "false"]
        else:
            cmd += [f"--{key}", str(val)]

    return cmd


def run(
    model_id: str,
    peft_method: str,
    hparams: dict,
    train_dataset: Path,
    val_dataset: Path | None,
    output_dir: Path,
    log_callback: Callable[[str], None] | None = None,
) -> dict:
    """Execute `swift sft`, stream logs, parse metrics, return result dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(model_id, peft_method, hparams, train_dataset, val_dataset, output_dir)
    cmd_str = " ".join(cmd)
    logger.info("swift_run cmd=%s", cmd_str)
    if log_callback:
        log_callback(f"[swift] {cmd_str}")

    metrics: dict = {}
    tail: list[str] = []  # keep last 40 lines to surface errors

    import os
    swift_env = os.environ.copy()
    swift_env["USE_HF"] = "1"                              # force HuggingFace instead of ModelScope
    swift_env["HF_HUB_DISABLE_SYMLINKS_IN_WINDOWS"] = "1" # fix WinError 1314 on Windows
    swift_env["HF_HUB_DISABLE_SYMLINKS"] = "1"            # compatibility with older hub versions
    swift_env["TRUST_REMOTE_CODE"] = "1"                   # allow custom architectures

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=swift_env,
        )
        assert proc.stdout
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            logger.info("swift | %s", line)   # INFO so it appears in server logs
            if log_callback:
                log_callback(line)

            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)

            # SWIFT emits metric dicts as JSON lines: {"loss": 1.2, "epoch": 1.0, ...}
            if line.startswith("{") and "loss" in line:
                try:
                    metrics.update(json.loads(line))
                except json.JSONDecodeError:
                    pass

            # Fallback regex for eval_loss in formatted output
            m = re.search(r"'eval_loss':\s*([\d.eE+\-]+)", line)
            if m:
                metrics["eval_loss"] = float(m.group(1))

        proc.wait()
        exit_code = proc.returncode

    except FileNotFoundError:
        msg = "SWIFT not installed — run: pip install ms-swift"
        logger.error(msg)
        return {"success": False, "error": msg, "metrics": {}}
    except Exception as exc:
        logger.error("swift_run_exception: %s", exc)
        return {"success": False, "error": str(exc), "metrics": metrics}

    if exit_code != 0:
        error_lines = "\n".join(tail[-15:])  # last 15 lines = the actual error
        logger.error("swift_run_failed exit_code=%d\n%s", exit_code, error_lines)
        return {
            "success":    False,
            "error":      f"swift sft exited with code {exit_code}",
            "swift_error": error_lines,
            "metrics":    metrics,
            "output_dir": str(output_dir),
            "command":    cmd_str,
        }

    logger.info("swift_run_done metrics=%s", metrics)
    return {
        "success":    True,
        "output_dir": str(output_dir),
        "metrics":    metrics,
        "command":    cmd_str,
    }
