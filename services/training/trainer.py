"""High-level training coordinator.

Flow
----
run_training():
  1. auto_peft   — LLM confirms/adjusts PEFT method from hardware + model analysis
  2. hparam_advisor — LLM recommends specific hyperparameter values
  3. swift_runner.run() — first training run
  4. training_report.json written

run_hpo() [single-phase HPO]:
  1. hparam_advisor — LLM recommends initial hparams (seed for trial 0)
  2. Optuna seeds trial 0 with LLM values, runs n_trials full training runs
  3. Each trial uses FULL epochs from the search space (no probe cap)
  4. Best trial IS the final model — training_report.json written from best trial
  No separate run_training() needed after run_hpo().

Public API
----------
run_training(job_id, model_id, peft_method, swift_config, n_pairs, vram_gb, task) -> dict
run_hpo(job_id, model_id, peft_method, n_pairs, vram_gb, task, n_trials)          -> dict
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from agents.llm_client import get_llm_client
from data.artifact_store import (
    train_path,
    val_path,
    training_output_path,
    training_report_path,
    write_json,
)
from services.training.auto_peft import recommend_peft
from services.training.hparam_advisor import recommend_hparams
from services.training.hparams import adaptive_max_length
from services.training import swift_runner

logger = logging.getLogger(__name__)


def run_training(
    job_id: str,
    model_id: str,
    peft_method: str = "qlora",
    swift_config: dict | None = None,
    n_pairs: int = 0,
    vram_gb: int = 4,
    task: str = "question-answering",
    hparam_overrides: dict | None = None,
    log_callback: Callable[[str], None] | None = None,
    objective: str = "balanced",
) -> dict:
    """Full training pipeline: auto_peft → hparam_advisor → swift sft."""
    train_file = train_path(job_id)
    val_file   = val_path(job_id)

    if not train_file.exists():
        raise FileNotFoundError(
            f"train.jsonl not found for job {job_id}. "
            "The data agent must call tool_finish before training."
        )

    actual_n = n_pairs or sum(1 for _ in train_file.open(encoding="utf-8"))
    llm      = get_llm_client()

    # ── Step 1: auto_peft — LLM confirms/overrides PEFT method ───────────────
    peft_rec = recommend_peft(model_id, vram_gb, actual_n, task, llm, objective=objective)
    final_peft = peft_rec["peft_method"]
    if final_peft != peft_method:
        logger.info(
            "auto_peft overrides selection: %s → %s | %s",
            peft_method, final_peft, peft_rec.get("rationale", ""),
        )
    if peft_rec.get("warning"):
        logger.warning("auto_peft warning: %s", peft_rec["warning"])

    # ── Step 2: hparam_advisor — LLM recommends hyperparameters ──────────────
    hparams = recommend_hparams(model_id, final_peft, actual_n, vram_gb, task, llm)
    source  = hparams.pop("_source", "unknown")
    rationale = hparams.pop("_rationale", "")

    # Adaptive max_length — computed from actual dataset token stats.
    # Overrides the static 2048 default only if the caller did not already
    # provide an explicit max_length override.
    if not hparam_overrides or "max_length" not in hparam_overrides:
        hparams["max_length"] = adaptive_max_length(train_file, task)
        logger.info("adaptive_max_length=%d job=%s", hparams["max_length"], job_id)

    if hparam_overrides:
        hparams.update(hparam_overrides)

    logger.info(
        "hparams [%s] job=%s rank=%d alpha=%d lr=%s epochs=%d max_length=%d | %s",
        source, job_id,
        hparams["lora_rank"], hparams["lora_alpha"],
        hparams["learning_rate"], hparams["num_train_epochs"],
        hparams["max_length"],
        rationale,
    )

    # ── Step 3: Free GPU before handing it to SWIFT ──────────────────────────
    def _log_vram(label: str) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                used = total - free
                logger.info(
                    "VRAM [%s] — free: %.2f GB / total: %.2f GB / used: %.2f GB",
                    label, free / 1024**3, total / 1024**3, used / 1024**3,
                )
        except Exception as e:
            logger.warning("VRAM measure failed [%s]: %s", label, e)

    _log_vram("before_release_gpu")

    try:
        import urllib.request, urllib.error
        from core.config import get_settings as _gs
        _mcp_url = f"http://localhost:{_gs().mcp_port}/release_gpu"
        _payload = json.dumps({"job_id": job_id}).encode()
        _req = urllib.request.Request(
            _mcp_url, data=_payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(_req, timeout=30) as _resp:
            _result = _resp.read().decode()
            logger.info("mcp release_gpu response: %s", _result[:300])
    except Exception as _mcp_exc:
        logger.warning("mcp release_gpu call failed (non-fatal): %s", _mcp_exc)

    _log_vram("after_release_gpu")

    # Also flush local CUDA cache in the uvicorn process
    try:
        from services.vram_cleanup import cleanup_vram
        cleanup_vram(label="before_swift_training")
    except Exception as _vram_exc:
        logger.warning("pre-training VRAM cleanup failed (non-fatal): %s", _vram_exc)

    _log_vram("after_full_cleanup")

    # ── Step 4: Run SWIFT ─────────────────────────────────────────────────────
    result = swift_runner.run(
        model_id=model_id,
        peft_method=final_peft,
        hparams=hparams,
        train_dataset=train_file,
        val_dataset=val_file if val_file.exists() else None,
        output_dir=training_output_path(job_id),
        log_callback=log_callback,
    )

    # ── Step 4: Write report ──────────────────────────────────────────────────
    public_hparams = {k: v for k, v in hparams.items() if not k.startswith("_")}
    report = {
        "job_id":       job_id,
        "model_id":     model_id,
        "peft_method":  final_peft,
        "peft_source":  peft_rec["source"],
        "peft_rationale": peft_rec.get("rationale", ""),
        "hparam_source": source,
        "hparam_rationale": rationale,
        "task":         task,
        "n_pairs":      actual_n,
        "hparams":      public_hparams,
        "result":       result,
    }
    write_json(training_report_path(job_id), report)
    logger.info("training_done job=%s success=%s", job_id, result.get("success"))
    return report


# ── Phase 2: Optuna HPO ───────────────────────────────────────────────────────

def run_hpo(
    job_id: str,
    model_id: str,
    peft_method: str = "qlora",
    n_pairs: int = 0,
    vram_gb: int = 4,
    task: str = "question-answering",
    n_trials: int = 5,
    log_callback: Callable[[str], None] | None = None,
    objective: str = "balanced",
) -> dict:
    """Single-phase Optuna HPO — each trial is a full training run.

    Trial 0 = LLM recommendation (seeded via enqueue_trial).
    Trials 1..n = Optuna explores search space using TPE.
    Each trial uses FULL epochs from the search space — no probe cap.
    Best trial IS the final model. Writes training_report.json.
    No run_training() needed after this call.
    """
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna not installed (pip install optuna) — falling back to run_training()")
        return run_training(
            job_id=job_id, model_id=model_id, peft_method=peft_method,
            n_pairs=n_pairs, vram_gb=vram_gb, task=task,
            log_callback=log_callback,
        )

    from services.training.hparams import optuna_search_space

    llm        = get_llm_client()
    final_peft = peft_method

    llm_hparams = recommend_hparams(model_id, final_peft, n_pairs, vram_gb, task, llm)
    llm_hparams.pop("_source", None)
    llm_hparams.pop("_rationale", None)

    train_file = train_path(job_id)
    val_file   = val_path(job_id)

    if not train_file.exists():
        raise FileNotFoundError(
            f"train.jsonl not found for job {job_id}. "
            "The data agent must finish before training."
        )

    actual_n = n_pairs or sum(1 for _ in train_file.open(encoding="utf-8"))

    # Adaptive max_length — shared across all HPO trials (same dataset)
    llm_hparams["max_length"] = adaptive_max_length(train_file, task)
    logger.info("adaptive_max_length=%d job=%s (hpo)", llm_hparams["max_length"], job_id)
    hpo_base = training_output_path(job_id) / "hpo"

    # Track per-trial results for the report
    trial_results: dict[int, dict] = {}

    def _objective(trial) -> float:
        search  = optuna_search_space(trial, peft_method=final_peft, n_pairs=actual_n)
        hparams = dict(llm_hparams)
        hparams.update(search)
        # lora_alpha derived from ratio × rank (not an independent dimension)
        hparams["lora_alpha"] = search["lora_alpha"]

        # Kill orphan SWIFT processes + flush CUDA cache before each trial
        try:
            from services.vram_cleanup import cleanup_vram
            cleanup_vram(label=f"before_trial_{trial.number}")
        except Exception as _ce:
            logger.warning("pre-trial VRAM cleanup failed (non-fatal): %s", _ce)

        completed = [
            t.value for t in study.trials
            if t.value is not None and t.value != float("inf")
            and t.number != trial.number
        ]
        global_best = min(completed) if completed else float("inf")

        result = swift_runner.run(
            model_id=model_id,
            peft_method=final_peft,
            hparams=hparams,
            train_dataset=train_file,
            val_dataset=val_file if val_file.exists() else None,
            output_dir=hpo_base / f"trial_{trial.number}",
            log_callback=log_callback,
            cross_trial_best=global_best,
        )
        trial_results[trial.number] = {"hparams": hparams, "result": result}

        if not result.get("success"):
            logger.warning(
                "hpo_trial_failed job=%s trial=%d error=%s",
                job_id, trial.number, result.get("error", "unknown"),
            )
        return result.get("metrics", {}).get("eval_loss", float("inf"))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")

    # Seed trial 0 with LLM values — Optuna explores around this starting point
    _rank  = llm_hparams.get("lora_rank", 16)
    _alpha = llm_hparams.get("lora_alpha", _rank * 2)
    _ratio = max(1, round(_alpha / _rank)) if _rank else 2
    _ratio = min([1, 2, 4], key=lambda x: abs(x - _ratio))

    seed_params = {
        k: llm_hparams[k]
        for k in ("lora_rank", "lora_dropout",
                  "learning_rate", "warmup_ratio", "weight_decay", "num_train_epochs")
        if k in llm_hparams
    }
    seed_params["lora_alpha_ratio"] = _ratio

    # Stop the study early if no improvement after 2 consecutive trials
    _no_improve_streak = [0]
    _best_so_far       = [float("inf")]

    def _early_stop_callback(study, trial):
        val = trial.value
        if val is None or val == float("inf"):
            _no_improve_streak[0] += 1
        elif val < _best_so_far[0]:
            _best_so_far[0]       = val
            _no_improve_streak[0] = 0
        else:
            _no_improve_streak[0] += 1

        if _no_improve_streak[0] >= 2 and len(study.trials) >= 2:
            logger.info(
                "hpo_early_stop: no improvement for %d consecutive trials — stopping study",
                _no_improve_streak[0],
            )
            if log_callback:
                log_callback(
                    f"[HPO] Early stop — aucune amélioration depuis "
                    f"{_no_improve_streak[0]} trials consécutifs."
                )
            study.stop()

    study.enqueue_trial(seed_params)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_early_stop_callback])

    best_num    = study.best_trial.number
    best_hparams = trial_results[best_num]["hparams"]
    best_result  = trial_results[best_num]["result"]

    logger.info(
        "hpo_done job=%s best_trial=%d best_loss=%.4f",
        job_id, best_num, study.best_value,
    )

    hpo_report = {
        "peft_method":    final_peft,
        "llm_seed":       seed_params,
        "best_params":    study.best_params,
        "best_eval_loss": study.best_value,
        "best_trial":     best_num,
        "n_trials":       len(study.trials),
        "trials": [
            {"number": t.number, "params": t.params, "value": t.value}
            for t in study.trials
        ],
    }
    write_json(training_output_path(job_id) / "hpo_report.json", hpo_report)

    # Write training_report.json from best trial — replaces run_training() report
    public_hparams = {k: v for k, v in best_hparams.items() if not k.startswith("_")}
    report = {
        "job_id":           job_id,
        "model_id":         model_id,
        "peft_method":      final_peft,
        "peft_source":      "selection",
        "hparam_source":    "optuna",
        "hparam_rationale": (
            f"Best of {len(study.trials)} Optuna trials (TPE), "
            f"trial {best_num}, eval_loss={study.best_value:.4f}"
        ),
        "task":             task,
        "n_pairs":          actual_n,
        "hparams":          public_hparams,
        "result":           best_result,
        "hpo_report":       hpo_report,
    }
    write_json(training_report_path(job_id), report)
    return report
