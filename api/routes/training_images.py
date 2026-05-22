"""Training visualization endpoints.

GET /api/runs/{id}/loss-history   — parsed trainer_state.json → loss curves data
GET /api/runs/{id}/images         — list of available training plot images
GET /api/runs/{id}/images/{name}  — serve a specific training image
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from core.config import get_settings

router = APIRouter(prefix="/api/runs")


def _training_output_dir(job_id: str) -> Path | None:
    """Find the most recent training output directory for a job."""
    base = Path(get_settings().artifacts_dir) / job_id / "training_output"
    if not base.exists():
        return None
    # Versioned dirs: v0-20260428-141745, v1-..., etc.
    versioned = sorted(base.iterdir(), key=lambda p: p.name, reverse=True)
    for d in versioned:
        if d.is_dir():
            return d
    return None


def _collect_loss_history(job_id: str) -> list[dict]:
    """Parse trainer_state.json from the best checkpoint and return loss history."""
    output_dir = _training_output_dir(job_id)
    if not output_dir:
        return []

    # For HPO runs: checkpoints are nested under hpo/trial_N/vN-.../checkpoint-*
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        checkpoints = sorted(
            output_dir.glob("*/*/checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )

    # Merge log_history from all checkpoints (last checkpoint has the full history)
    for cp in reversed(checkpoints):
        ts = cp / "trainer_state.json"
        if ts.exists():
            try:
                data = json.loads(ts.read_text(encoding="utf-8"))
                raw_history = data.get("log_history", [])
                points = []
                for entry in raw_history:
                    step = entry.get("step") or entry.get("epoch")
                    if step is None:
                        continue
                    point: dict = {"step": step}
                    if "loss" in entry:
                        point["train_loss"] = round(float(entry["loss"]), 4)
                    if "eval_loss" in entry:
                        point["eval_loss"] = round(float(entry["eval_loss"]), 4)
                    if "epoch" in entry:
                        point["epoch"] = round(float(entry["epoch"]), 3)
                    if len(point) > 1:   # has at least one loss value
                        points.append(point)
                if points:
                    return points
            except Exception:
                continue

    return []


@router.get("/{job_id}/loss-history")
async def get_loss_history(job_id: str) -> JSONResponse:
    history = _collect_loss_history(job_id)
    if not history:
        raise HTTPException(404, "No training history found for this job")
    return JSONResponse({"job_id": job_id, "points": history})


@router.get("/{job_id}/images")
async def list_training_images(job_id: str) -> JSONResponse:
    output_dir = _training_output_dir(job_id)
    if not output_dir:
        return JSONResponse({"images": []})

    images_dir = output_dir / "images"
    if not images_dir.exists():
        return JSONResponse({"images": []})

    images = [
        {"name": f.name, "url": f"/api/runs/{job_id}/images/{f.name}"}
        for f in sorted(images_dir.iterdir())
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}
    ]
    return JSONResponse({"images": images})


@router.get("/{job_id}/images/{image_name}")
async def serve_training_image(job_id: str, image_name: str) -> FileResponse:
    output_dir = _training_output_dir(job_id)
    if not output_dir:
        raise HTTPException(404, "Training output not found")

    image_path = output_dir / "images" / image_name
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(404, f"Image '{image_name}' not found")

    # Prevent path traversal
    try:
        image_path.resolve().relative_to((output_dir / "images").resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")

    media_type = "image/png" if image_path.suffix == ".png" else "image/jpeg"
    return FileResponse(str(image_path), media_type=media_type)
