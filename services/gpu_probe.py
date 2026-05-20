"""GPU VRAM detection — tries multiple backends in order of precision.

Public API
----------
detect_vram() -> GpuInfo | None
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GpuInfo:
    name: str
    total_gb: int       # total VRAM on the card
    available_gb: int   # free VRAM right now (total minus driver/OS/other processes)


def detect_vram() -> GpuInfo | None:
    """Return GPU info with total and available VRAM.

    Tries, in order:
      1. PyTorch CUDA  — most precise, accounts for the allocator's reserved pool
      2. pynvml        — NVIDIA Management Library (standalone, no torch needed)
      3. nvidia-smi    — subprocess fallback (always available if driver is installed)

    Returns None if no NVIDIA GPU is detected or all backends fail.
    """
    for probe in (_probe_torch, _probe_pynvml, _probe_nvidia_smi):
        info = probe()
        if info is not None:
            return info

    logger.warning("gpu_probe: no GPU detected — all backends failed")
    return None


# ── Backend 1: PyTorch CUDA ───────────────────────────────────────────────────

def _probe_torch() -> GpuInfo | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        idx   = 0
        props = torch.cuda.get_device_properties(idx)
        total = props.total_memory

        # Currently reserved by the PyTorch memory allocator (includes cached blocks)
        reserved  = torch.cuda.memory_reserved(idx)
        free_raw  = total - reserved
        # Subtract 0.5 GB for CUDA context + driver overhead
        available = max(1, int(free_raw / 1024 ** 3 - 0.5))

        return GpuInfo(
            name=props.name,
            total_gb=max(1, round(total / 1024 ** 3)),
            available_gb=available,
        )
    except Exception as exc:
        logger.debug("torch probe failed: %s", exc)
        return None


# ── Backend 2: pynvml ─────────────────────────────────────────────────────────

def _probe_pynvml() -> GpuInfo | None:
    try:
        import pynvml  # pip install nvidia-ml-py

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name   = pynvml.nvmlDeviceGetName(handle)
        mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()

        if isinstance(name, bytes):
            name = name.decode()

        available = max(1, int(mem.free / 1024 ** 3 - 0.5))
        return GpuInfo(
            name=name,
            total_gb=max(1, round(mem.total / 1024 ** 3)),
            available_gb=available,
        )
    except Exception as exc:
        logger.debug("pynvml probe failed: %s", exc)
        return None


# ── Backend 3: nvidia-smi subprocess ─────────────────────────────────────────

def _probe_nvidia_smi() -> GpuInfo | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if result.returncode != 0:
            return None

        line   = result.stdout.strip().split("\n")[0]
        parts  = [p.strip() for p in line.split(",")]
        name, total_mb, free_mb = parts[0], int(parts[1]), int(parts[2])

        available = max(1, int(free_mb / 1024 - 0.5))
        return GpuInfo(
            name=name,
            total_gb=max(1, round(total_mb / 1024)),
            available_gb=available,
        )
    except Exception as exc:
        logger.debug("nvidia-smi probe failed: %s", exc)
        return None
