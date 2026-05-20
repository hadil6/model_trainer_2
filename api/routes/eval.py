"""POST /api/runs/{id}/eval  — trigger evaluation on a completed training job.
GET  /api/runs/{id}/eval  — fetch the saved eval_report.json.

Works from disk artifacts — does NOT require the run to be in the in-memory RunManager,
so it works even after a server restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from data.artifact_store import eval_report_path, training_output_path, training_report_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/runs")


def _require_artifacts(job_id: str) -> dict:
    """Load training_report.json from disk. Raises 404 if not found."""
    path = training_report_path(job_id)
    if not path.exists():
        raise HTTPException(
            404,
            f"No training report found for job {job_id}. "
            "Run training first or check the job_id.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _find_best_checkpoint(output_dir: str) -> str | None:
    if not output_dir:
        return None
    base = Path(output_dir)
    if not base.exists():
        return None

    best_link = base / "best_model"
    if best_link.exists():
        return str(best_link.resolve())

    checkpoints = sorted(
        base.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        if (base / "adapter_config.json").exists():
            return str(base)
        return None

    best_path, best_loss = None, float("inf")
    for cp in checkpoints:
        trainer_state = cp / "trainer_state.json"
        if trainer_state.exists():
            try:
                state_data = json.loads(trainer_state.read_text(encoding="utf-8"))
                for entry in state_data.get("log_history", []):
                    if "eval_loss" in entry and entry["eval_loss"] < best_loss:
                        best_loss = entry["eval_loss"]
                        best_path = str(cp)
            except Exception:
                pass

    return best_path or str(checkpoints[-1])


@router.post("/{job_id}/eval")
async def trigger_eval(
    job_id: str,
    background_tasks: BackgroundTasks,
    n_samples: int = 50,
    use_llm_judge: bool = True,
    judge_n: int = 10,
) -> JSONResponse:
    """Kick off evaluation in the background. Returns immediately with 202."""
    training_report = _require_artifacts(job_id)

    model_id   = training_report.get("model_id") or ""
    output_dir = (
        (training_report.get("result") or {}).get("output_dir")
        or str(training_output_path(job_id))
    )

    # Scan subdirectories too (SWIFT writes to a timestamped subdir)
    adapter_path = _find_best_checkpoint(output_dir)
    if not adapter_path:
        # Try one level deeper (e.g. training_output/v0-20260426-194501/)
        base = Path(output_dir)
        for sub in sorted(base.iterdir()) if base.exists() else []:
            if sub.is_dir():
                candidate = _find_best_checkpoint(str(sub))
                if candidate:
                    adapter_path = candidate
                    break

    if not adapter_path:
        raise HTTPException(
            400,
            f"No checkpoint found under {output_dir}. "
            "Ensure training completed successfully.",
        )
    if not model_id:
        raise HTTPException(400, "model_id not found in training_report.json")

    async def _run_eval():
        from services.training.evaluator import evaluate
        try:
            loop = asyncio.get_event_loop()
            report = await loop.run_in_executor(
                None,
                lambda: evaluate(
                    job_id=job_id,
                    adapter_path=adapter_path,
                    base_model_id=model_id,
                    n_samples=n_samples,
                    use_llm_judge=use_llm_judge,
                    judge_n=judge_n,
                ),
            )
            logger.info(
                "eval_api_done job=%s rouge1=%.3f rougeL=%.3f bleu4=%.3f",
                job_id,
                report.get("metrics", {}).get("rouge1", 0),
                report.get("metrics", {}).get("rougeL", 0),
                report.get("metrics", {}).get("bleu4", 0),
            )
        except Exception as exc:
            logger.error("eval_api_failed job=%s: %s", job_id, exc)

    background_tasks.add_task(_run_eval)
    return JSONResponse(
        {"status": "evaluation_started", "job_id": job_id, "adapter_path": adapter_path},
        status_code=202,
    )


@router.get("/{job_id}/eval")
async def get_eval(job_id: str) -> JSONResponse:
    """Return saved eval_report.json for a job."""
    path = eval_report_path(job_id)
    if not path.exists():
        raise HTTPException(
            404,
            "eval_report.json not found — run POST /api/runs/{job_id}/eval first",
        )
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
