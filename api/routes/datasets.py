"""GET /api/runs/{id}/dataset, GET /api/runs/{id}/dataset/download."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from data.artifact_store import dataset_path
from services.run_manager import get_run_manager

router = APIRouter(prefix="/api/runs")


def _get_dataset_path(job_id: str) -> Path:
    # First try canonical path from artifact_store (works after server restart)
    p = dataset_path(job_id)
    if p.exists():
        return p

    # Fall back to in-memory result dict if the run is still tracked
    handle = get_run_manager().get(job_id)
    if handle is not None:
        if handle.status == "running":
            raise HTTPException(202, "run still in progress")
        result = handle.result or {}
        raw = (result.get("dataset") or {}).get("dataset_path") or ""
        if raw:
            pp = Path(raw)
            if pp.exists():
                return pp

    raise HTTPException(404, "dataset file not found")


@router.get("/{job_id}/dataset")
async def get_dataset(job_id: str) -> JSONResponse:
    path = _get_dataset_path(job_id)
    pairs: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    fmt = "sharegpt" if any("conversations" in p for p in pairs[:5]) else "alpaca"
    return JSONResponse({"count": len(pairs), "format": fmt, "pairs": pairs})


@router.get("/{job_id}/dataset/download")
async def download_dataset(job_id: str) -> FileResponse:
    path = _get_dataset_path(job_id)
    return FileResponse(
        str(path),
        filename=path.name,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
    )
