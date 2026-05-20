"""Training report + model export endpoints.

GET  /api/runs/{id}/training-report  — structured training_report.json (works after restart)
GET  /api/runs/{id}/report/download  — download merged JSON report (training + eval)
GET  /api/runs/{id}/model/download   — download LoRA adapter as ZIP
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from data.artifact_store import (
    eval_report_path,
    training_output_path,
    training_report_path,
)
from core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/runs")


def _load_report(job_id: str) -> dict:
    path = training_report_path(job_id)
    if not path.exists():
        raise HTTPException(
            404,
            f"Aucun rapport d'entraînement trouvé pour le job {job_id}. "
            "L'entraînement doit d'abord être terminé.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _find_best_checkpoint(output_dir: str) -> Path | None:
    base = Path(output_dir)
    if not base.exists():
        return None

    best_link = base / "best_model"
    if best_link.exists():
        resolved = best_link.resolve()
        return resolved if resolved.exists() else None

    checkpoints = sorted(
        base.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        if (base / "adapter_config.json").exists():
            return base
        return None

    best_path, best_loss = None, float("inf")
    for cp in checkpoints:
        trainer_state = cp / "trainer_state.json"
        if trainer_state.exists():
            try:
                data = json.loads(trainer_state.read_text(encoding="utf-8"))
                for entry in data.get("log_history", []):
                    if "eval_loss" in entry and entry["eval_loss"] < best_loss:
                        best_loss = entry["eval_loss"]
                        best_path = cp
            except Exception:
                pass

    return best_path or checkpoints[-1]


def _resolve_adapter_dir(training_report: dict) -> Path | None:
    output_dir = (training_report.get("result") or {}).get("output_dir", "")
    cp = _find_best_checkpoint(output_dir)
    if cp:
        return cp
    # Try one level deeper (SWIFT timestamped subdir)
    base = Path(output_dir)
    if base.exists():
        for sub in sorted(base.iterdir()):
            if sub.is_dir():
                cp = _find_best_checkpoint(str(sub))
                if cp:
                    return cp
    return None


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/{job_id}/training-report")
async def get_training_report(job_id: str) -> JSONResponse:
    """Return training_report.json as JSON."""
    return JSONResponse(_load_report(job_id))


@router.get("/{job_id}/report/download")
async def download_report(job_id: str) -> StreamingResponse:
    """Download a merged JSON report (training + eval) as a .json file."""
    training = _load_report(job_id)

    eval_path = eval_report_path(job_id)
    evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None

    final_output_path = Path(get_settings().artifacts_dir) / job_id / "final_output.json"
    decision_journal = None
    if final_output_path.exists():
        try:
            fo = json.loads(final_output_path.read_text(encoding="utf-8"))
            decision_journal = fo.get("decision_journal")
        except Exception:
            pass

    merged = {
        "job_id":            job_id,
        "training":          training,
        "evaluation":        evaluation,
        "decision_journal":  decision_journal,
    }
    content = json.dumps(merged, ensure_ascii=False, indent=2).encode("utf-8")

    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="report_{job_id}.json"'},
    )


@router.get("/{job_id}/model/download")
async def download_model(job_id: str) -> StreamingResponse:
    """Download the best LoRA adapter checkpoint as a ZIP archive."""
    training = _load_report(job_id)

    adapter_dir = _resolve_adapter_dir(training)
    if not adapter_dir:
        raise HTTPException(
            404,
            "Aucun checkpoint trouvé. Vérifiez que l'entraînement s'est terminé avec succès.",
        )

    # Build ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(adapter_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(adapter_dir)
                zf.write(file_path, arcname)
    buf.seek(0)

    model_id = training.get("model_id", "model").split("/")[-1]
    filename = f"{model_id}_lora_{job_id}.zip"
    logger.info("model_download job=%s adapter=%s size=%d bytes", job_id, adapter_dir, buf.getbuffer().nbytes)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
