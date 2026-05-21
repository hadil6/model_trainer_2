"""VRAM cleanup — kill orphan child processes and flush CUDA cache.

This is a defensive measure for environments where:
  - Windows WDDM is slow to release CUDA pages after a subprocess exits
  - MCP servers from a previous run remain in memory
  - multiprocessing workers spawned by transformers/accelerate are not collected

When called before each new run, it guarantees the next pipeline starts with
the maximum possible free VRAM regardless of what happened during the previous
run (success, crash, or hard error).

Public API
----------
cleanup_vram(label) -> dict
"""

from __future__ import annotations

import gc
import logging
import os
import time

logger = logging.getLogger(__name__)


def _measure_free_vram_mb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.mem_get_info()[0] / 1024 ** 2, 1)
    except Exception as exc:
        logger.debug("vram_cleanup: cannot measure VRAM: %s", exc)
        return None


def _is_target_process(cmdline_str: str) -> tuple[bool, str]:
    """Return (should_kill, reason) for a Python process command line.

    Targets — strictly per-run transient workers spawned by training/eval libs.
    The MCP server (`agents.data_agent.mcp_server`) is INTENTIONALLY excluded:
    it is a persistent infrastructure service that must remain up across runs.
    Killing it would break the next `prepare_data` call.
    """
    if "multiprocessing.spawn" in cmdline_str or "--multiprocessing-fork" in cmdline_str:
        return True, "multiprocessing_worker"
    if "torch.distributed.run" in cmdline_str or "torch.multiprocessing.spawn" in cmdline_str:
        return True, "torch_distributed_worker"
    if "swift/cli/sft.py" in cmdline_str or "swift/cli/train.py" in cmdline_str:
        return True, "swift_sft_orphan"
    return False, ""


def cleanup_vram(label: str = "cleanup") -> dict:
    """Release VRAM from orphaned child processes and flush CUDA cache.

    Steps:
      1. Measure free VRAM (before).
      2. Scan all Python processes; kill orphan MCP servers and multiprocessing
         workers that are NOT children of the current FastAPI process.
         (Children of the current process belong to an active run — keep them.)
      3. Force gc.collect() + torch.cuda.empty_cache() + ipc_collect() in current process.
      4. Sleep 2s to let the WDDM driver release pages.
      5. Measure free VRAM (after) and return a report.

    Safe to call repeatedly. Never kills the current process.

    Parameters
    ----------
    label : str
        A short tag included in log messages (e.g. "before_run", "after_run").
    """
    report: dict = {
        "label":           label,
        "killed":          [],
        "vram_before_mb":  None,
        "vram_after_mb":   None,
        "errors":          [],
    }

    report["vram_before_mb"] = _measure_free_vram_mb()

    current_pid = os.getpid()

    # ── 1. Kill orphan Python processes holding CUDA contexts ────────────────
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "name", "cmdline", "ppid"]):
            try:
                info = proc.info
                pid = info["pid"]
                if pid == current_pid:
                    continue

                name = (info.get("name") or "").lower()
                if "python" not in name:
                    continue

                cmdline_list = info.get("cmdline") or []
                cmdline_str = " ".join(cmdline_list)

                should_kill, reason = _is_target_process(cmdline_str)
                if not should_kill:
                    continue

                # Keep children of the current FastAPI server — they belong to
                # an in-flight run. Only kill TRULY orphaned processes.
                try:
                    if proc.ppid() == current_pid:
                        continue
                except psutil.NoSuchProcess:
                    pass

                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    report["killed"].append({
                        "pid":    pid,
                        "reason": reason,
                        "cmdline": cmdline_str[:120],
                    })
                    logger.info(
                        "vram_cleanup[%s]: killed PID=%d (%s)",
                        label, pid, reason,
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                    report["errors"].append(f"kill_failed pid={pid}: {exc}")

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    except ImportError:
        report["errors"].append("psutil_not_installed")
    except Exception as exc:
        report["errors"].append(f"orphan_scan: {type(exc).__name__}: {exc}")

    # ── 2. Flush CUDA cache in current process ───────────────────────────────
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception as exc:
        report["errors"].append(f"cuda_flush: {type(exc).__name__}: {exc}")

    # ── 3. Release heavy inference models that are no longer needed ──────────
    try:
        from data.model_zoo.embeddings import release_sentence_transformer
        release_sentence_transformer()
    except Exception as exc:
        report["errors"].append(f"st_release: {type(exc).__name__}: {exc}")

    try:
        from data.profilers.pdf_profiler import release_marker_models
        release_marker_models()
    except Exception as exc:
        report["errors"].append(f"marker_release: {type(exc).__name__}: {exc}")

    # ── 4. Wait for the driver (WDDM on Windows is slow) ─────────────────────
    time.sleep(2)

    report["vram_after_mb"] = _measure_free_vram_mb()

    before = report["vram_before_mb"]
    after  = report["vram_after_mb"]
    if before is not None and after is not None:
        report["freed_mb"] = round(after - before, 1)

    logger.info(
        "vram_cleanup[%s]: killed=%d  before=%s MB  after=%s MB  errors=%d",
        label,
        len(report["killed"]),
        report["vram_before_mb"],
        report["vram_after_mb"],
        len(report["errors"]),
    )
    return report
