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

    # ── Early stopping state ──────────────────────────────────────────────────
    _ES_DIVERGE_THRESHOLD = 0.15   # eval_loss rises this much above best → stop immediately
    _es_best_eval    = float("inf")
    _es_eval_history: list[float] = []   # all eval_loss values seen so far
    _es_prev_train   = float("inf")
    _es_last_eval    = None              # detect when a new eval checkpoint appears
    _es_triggered    = False
    _es_reason       = ""

    import os
    swift_env = os.environ.copy()
    swift_env["USE_HF"] = "1"
    swift_env["HF_HUB_DISABLE_SYMLINKS_IN_WINDOWS"] = "1"
    swift_env["HF_HUB_DISABLE_SYMLINKS"] = "1"
    swift_env["TRUST_REMOTE_CODE"] = "1"

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
            logger.info("swift | %s", line)
            if log_callback:
                log_callback(line)

            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)

            # Parse metric lines
            line_metrics: dict = {}
            if line.startswith("{") and "loss" in line:
                try:
                    line_metrics = json.loads(line)
                    metrics.update(line_metrics)
                except json.JSONDecodeError:
                    pass

            m = re.search(r"'eval_loss':\s*([\d.eE+\-]+)", line)
            if m:
                val = float(m.group(1))
                metrics["eval_loss"] = val
                line_metrics.setdefault("eval_loss", val)

            # ── Early stopping check (fires only when a new eval_loss appears) ──
            cur_eval  = line_metrics.get("eval_loss")
            cur_train = line_metrics.get("loss") or metrics.get("loss", _es_prev_train)

            if cur_eval is not None and cur_eval != _es_last_eval:
                _es_last_eval = cur_eval
                _es_eval_history.append(cur_eval)
                history_str = [f"{v:.4f}" for v in _es_eval_history]

                # Rule 1 — Divergence: eval spikes while train still falls
                if (_es_best_eval < float("inf")
                        and cur_eval > _es_best_eval + _ES_DIVERGE_THRESHOLD
                        and cur_train < _es_prev_train):
                    _es_triggered = True
                    _es_reason = (
                        f"divergence — eval_loss={cur_eval:.4f} > "
                        f"best+{_ES_DIVERGE_THRESHOLD}={_es_best_eval + _ES_DIVERGE_THRESHOLD:.4f}, "
                        f"train_loss still falling ({_es_prev_train:.4f}→{cur_train:.4f}) "
                        f"history={history_str}"
                    )
                    logger.info("early_stop DIVERGENCE → %s", _es_reason)
                    if log_callback:
                        log_callback(f"[early_stop] DIVERGENCE: {_es_reason}")
                    proc.terminate()
                    break

                # Rule 2 — Dynamic overfitting: stop as soon as eval rises above best.
                # The first eval always sets the baseline (best=inf → any value improves it).
                # From the second eval onward: any rise above the best triggers stop — no
                # fixed patience count, the history itself determines the optimal point.
                if cur_eval < _es_best_eval:
                    _es_best_eval = cur_eval
                    logger.info(
                        "early_stop new_best=%.4f  eval_history=%s",
                        _es_best_eval, history_str,
                    )
                    if log_callback:
                        log_callback(
                            f"[early_stop] new best={_es_best_eval:.4f}  "
                            f"history={history_str}"
                        )
                elif _es_best_eval < float("inf"):
                    # eval_loss rose above the best — optimal point already passed
                    _es_triggered = True
                    _es_reason = (
                        f"overfitting — eval_loss={cur_eval:.4f} > best={_es_best_eval:.4f} "
                        f"after {len(_es_eval_history)} evals: {history_str}"
                    )
                    logger.info("early_stop OVERFITTING → %s", _es_reason)
                    if log_callback:
                        log_callback(f"[early_stop] OVERFITTING: {_es_reason}")
                    proc.terminate()
                    break

                _es_prev_train = cur_train

        proc.wait()
        exit_code = proc.returncode

    except FileNotFoundError:
        msg = "SWIFT not installed — run: pip install ms-swift"
        logger.error(msg)
        return {"success": False, "error": msg, "metrics": {}}
    except Exception as exc:
        logger.error("swift_run_exception: %s", exc)
        return {"success": False, "error": str(exc), "metrics": metrics}

    # Early-stopped trials are treated as success (best checkpoint already saved)
    if _es_triggered:
        logger.info("swift_run_early_stopped reason=%s metrics=%s", _es_reason, metrics)
        return {
            "success":       True,
            "early_stopped": True,
            "early_stop_reason": _es_reason,
            "output_dir":    str(output_dir),
            "metrics":       metrics,
            "command":       cmd_str,
        }

    if exit_code != 0:
        error_lines = "\n".join(tail[-15:])
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
        "success":       True,
        "early_stopped": False,
        "output_dir":    str(output_dir),
        "metrics":       metrics,
        "command":       cmd_str,
    }
