"""GET /api/gpu — returns detected GPU info for the frontend."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from services.gpu_probe import detect_vram

router = APIRouter(prefix="/api/gpu")


@router.get("")
async def gpu_info() -> JSONResponse:
    """Detect the primary GPU and return total/available VRAM."""
    info = detect_vram()
    if info is None:
        return JSONResponse({
            "detected": False,
            "name": None,
            "total_gb": None,
            "available_gb": None,
        })
    return JSONResponse({
        "detected":      True,
        "name":          info.name,
        "total_gb":      info.total_gb,
        "available_gb":  info.available_gb,
    })
