"""POST /api/runs, GET /api/runs/{id}/events, GET /api/runs/{id}/result."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from agents.orchestrator.agent import OrchestratorState, run_orchestrator
from core.config import get_settings
from data.artifact_store import (
    ensure_job_dirs, input_file_path,
    training_report_path, eval_report_path,
)
from services.run_manager import RunHandle, get_run_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/runs")

_RUN_DONE   = "__RUN_DONE__"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s — %(message)s"
_QUEUE_MAX  = 10_000


class _QueueLogHandler(logging.Handler):
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        super().__init__(level=logging.INFO)
        self._loop  = loop
        self._queue = queue
        self.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, line)


def _safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items() if k != "messages"}
    if isinstance(obj, list):
        return [_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _get_handle(job_id: str) -> RunHandle:
    handle = get_run_manager().get(job_id)
    if not handle:
        raise HTTPException(404, "run not found")
    return handle


def _result_from_disk(job_id: str) -> dict | None:
    """Reconstruct a completed run result from on-disk artifacts after a server restart."""
    report_path = training_report_path(job_id)
    if not report_path.exists():
        return None
    try:
        training_report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    eval_rep = None
    ep = eval_report_path(job_id)
    if ep.exists():
        try:
            eval_rep = json.loads(ep.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "job_id": job_id,
        "final_output": {
            "status": "success" if (training_report.get("result") or {}).get("success") else "failed",
            "training": training_report,
            "evaluation": eval_rep or {},
            "improvement_log": [],
        },
    }


@router.get("")
async def list_runs() -> JSONResponse:
    """Return a summary of all completed runs found in the artifacts directory."""
    settings = get_settings()
    artifacts_dir = Path(settings.artifacts_dir)
    runs: list[dict] = []

    if artifacts_dir.exists():
        for job_dir in sorted(artifacts_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not job_dir.is_dir():
                continue
            final = job_dir / "final_output.json"
            if not final.exists():
                continue
            try:
                data = json.loads(final.read_text(encoding="utf-8"))
                eval_section   = data.get("evaluation") or {}
                eval_metrics   = eval_section.get("metrics") or {}
                selection      = data.get("selected_model") or {}
                dataset        = data.get("dataset") or {}
                n_pairs        = (dataset.get("splits") or {}).get("total") or dataset.get("n_pairs")
                primary        = eval_metrics.get("primary_metric", "rouge1")
                primary_val    = eval_metrics.get(primary)
                runs.append({
                    "job_id":               data.get("job_id", job_dir.name),
                    "status":               data.get("status", "unknown"),
                    "task":                 data.get("task"),
                    "domain":               data.get("domain"),
                    "model_id":             selection.get("model_id"),
                    "peft_method":          selection.get("peft_method"),
                    "primary_metric":       primary,
                    "primary_metric_value": primary_val,
                    "quality_tier":         eval_metrics.get("quality_tier"),
                    "n_pairs":              n_pairs,
                    "created_at":           job_dir.stat().st_mtime,
                    "summary":              data.get("summary", ""),
                })
            except Exception:
                pass

    return JSONResponse({"runs": runs})


@router.post("")
async def create_run(
    files:        list[UploadFile] = File(...),
    goal:         str              = Form(""),
    language:     str              = Form("en"),
    domain:       str              = Form(""),
    target_model: str              = Form(""),
    gpu_vram_gb:  int              = Form(4),
    objective:    str              = Form("balanced"),
    task:         str              = Form(""),
) -> JSONResponse:
    if not files:
        raise HTTPException(400, "at least one file is required")

    # Free VRAM left over from previous runs (orphan MCP servers, multiprocessing
    # workers, cached CUDA pages). Guarantees the new run starts with maximum
    # available VRAM regardless of how the previous one ended.
    from services.vram_cleanup import cleanup_vram
    cleanup_report = cleanup_vram(label="before_run")
    logger.info(
        "vram pre-run cleanup: killed=%d before=%s MB after=%s MB",
        len(cleanup_report.get("killed", [])),
        cleanup_report.get("vram_before_mb"),
        cleanup_report.get("vram_after_mb"),
    )

    # Use available VRAM minus a headroom reserve to account for persistent
    # processes that share the GPU (MCP server, sentence transformer, profiling
    # tools, OS overhead). The reserve is proportional to available VRAM —
    # a fixed 3 GB would eat 75 % of a 4 GB GPU but only ~20 % of a 16 GB one.
    # Cap: at most 3 GB on large GPUs, at least 0.5 GB on tiny ones.
    from services.gpu_probe import detect_vram
    gpu_info = detect_vram()
    if gpu_info is not None:
        reserve = max(0.5, min(3.0, gpu_info.available_gb * 0.20))
        gpu_vram_gb = max(1, int(gpu_info.available_gb - reserve))
        logger.info(
            "GPU auto-detected: %s — total=%.1fGB available=%.1fGB reserved=%.1fGB → budget=%dGB for model selection",
            gpu_info.name, gpu_info.total_gb, gpu_info.available_gb, reserve, gpu_vram_gb,
        )
    else:
        logger.info("No GPU detected — using user-supplied gpu_vram_gb=%d", gpu_vram_gb)

    job_id = uuid.uuid4().hex[:12]
    settings = get_settings()
    ensure_job_dirs(job_id)

    filenames: list[str] = []
    for f in files:
        name = Path(f.filename or "upload.bin").name
        dest = input_file_path(job_id, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        filenames.append(name)

    manager = get_run_manager()
    handle = manager.create(job_id)
    handle.queue = asyncio.Queue(maxsize=_QUEUE_MAX)

    loop = asyncio.get_running_loop()
    handler = _QueueLogHandler(loop, handle.queue)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    orchestrator_state: OrchestratorState = {
        "job_id":        job_id,
        "artifact_dir":  str(Path(settings.artifacts_dir) / job_id),
        "filenames":     filenames,
        "user_goal":     goal,
        "language":      language,
        "domain":        domain or "",
        "objective":     objective if objective in ("speed", "balanced", "performance") else "balanced",
        "target_model":  target_model or None,
        "gpu_vram_gb":   gpu_vram_gb,
        "file_profiles": [],
        "dataset_size":  None,
        "user_intent":   None,
        "feasibility":   None,
        "selection_result": None,
        "data_result":   None,
        "training_result": None,
        "eval_result":   None,
        "eval_strategy": None,
        "improvement_iteration": 0,
        "improvement_log":       [],
        "hparam_overrides":      None,
        "target_n_pairs":        None,
        "attempted_models":      [],
        "messages":              [],
        "final_output":  None,
        "error":         None,
        "task_hint":     task or None,
    }

    async def _drive():
        try:
            result = await run_orchestrator(orchestrator_state)
            manager.set_result(job_id, _safe(result))
            verdict = (result.get("final_output") or {}).get("status")
            logger.info("run %s finished verdict=%s", job_id, verdict)
        except Exception as exc:
            manager.set_result(job_id, {"error": f"{type(exc).__name__}: {exc}"})
            logger.exception("run %s failed", job_id)
        finally:
            # Release VRAM no matter how the pipeline ended (success or crash):
            # kill MCP servers and multiprocessing workers spawned during this run,
            # flush CUDA cache, release the sentence transformer.
            try:
                from services.vram_cleanup import cleanup_vram
                cleanup_vram(label=f"after_run_{job_id}")
            except Exception as cleanup_exc:
                logger.warning("vram cleanup after run %s failed: %s", job_id, cleanup_exc)
            logging.getLogger().removeHandler(handler)
            await handle.queue.put(_RUN_DONE)

    asyncio.create_task(_drive())
    return JSONResponse({"run_id": job_id})


@router.get("/{job_id}/events")
async def stream_events(job_id: str) -> EventSourceResponse:
    manager = get_run_manager()
    handle  = manager.get(job_id)

    # Run not in memory (server restarted) — try to rebuild result from disk
    if handle is None:
        disk_result = _result_from_disk(job_id)
        if disk_result is None:
            raise HTTPException(404, "run not found")

        async def event_gen_disk():
            yield {"event": "open", "data": json.dumps({"run_id": job_id})}
            yield {"event": "log",  "data": f"[restored] Run {job_id} récupéré depuis le disque après redémarrage serveur"}
            payload = {"ok": True, "error": None, "result": disk_result}
            yield {"event": "done", "data": json.dumps(payload, default=str)}

        return EventSourceResponse(event_gen_disk())

    # Run is done but still in memory (reconnect after _RUN_DONE was consumed by a
    # previous SSE client).  Serve the stored result immediately.
    if handle.status != "running":
        stored = handle.result or {}

        async def event_gen_cached():
            yield {"event": "open", "data": json.dumps({"run_id": job_id})}
            cached_payload = {
                "ok":    not stored.get("error"),
                "error": stored.get("error"),
                "result": stored,
            }
            yield {"event": "done", "data": json.dumps(cached_payload, default=str)}

        return EventSourceResponse(event_gen_cached())

    async def event_gen():
        yield {"event": "open", "data": json.dumps({"run_id": job_id})}
        while True:
            line = await handle.queue.get()
            if line == _RUN_DONE:
                h = manager.get(job_id)
                result = h.result if h else None
                payload = {
                    "ok":     not (result or {}).get("error"),
                    "error":  (result or {}).get("error"),
                    "result": result,
                }
                yield {"event": "done", "data": json.dumps(payload, default=str)}
                return
            # Detect special structured events written directly into the queue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    if parsed.get("__confirm__"):
                        yield {"event": "confirm", "data": line}
                        continue
                    if parsed.get("__decision__"):
                        yield {"event": "decision", "data": line}
                        continue
            except (json.JSONDecodeError, AttributeError):
                pass
            yield {"event": "log", "data": line}

    return EventSourceResponse(event_gen())


@router.get("/{job_id}/result")
async def get_result(job_id: str) -> JSONResponse:
    handle = get_run_manager().get(job_id)

    if handle is None:
        # Server restarted — check if artifacts exist on disk
        disk_result = _result_from_disk(job_id)
        if disk_result is None:
            raise HTTPException(404, "run not found")
        return JSONResponse({"status": "done", "error": None, "result": disk_result})

    if handle.status == "running":
        return JSONResponse({"status": "running"}, status_code=202)
    return JSONResponse({
        "status": handle.status,
        "error":  handle.result.get("error") if handle.result else None,
        "result": handle.result,
    })


@router.post("/{job_id}/confirm")
async def confirm_action(job_id: str, decision: str = "approve") -> JSONResponse:
    """Unblock an orchestrator waiting for user approval (auto_fill_qa confirmation)."""
    if decision not in ("approve", "refuse"):
        raise HTTPException(400, "decision must be 'approve' or 'refuse'")
    handle = get_run_manager().get(job_id)
    if handle is None:
        raise HTTPException(404, "run not found")
    if handle.confirm_event is None or handle.confirm_event.is_set():
        raise HTTPException(400, "no pending confirmation for this run")
    handle.confirm_decision = decision
    handle.confirm_event.set()
    logger.info("confirm job=%s decision=%s", job_id, decision)
    return JSONResponse({"ok": True, "decision": decision})


@router.post("/{job_id}/additional-files")
async def upload_additional_files(
    job_id: str,
    files: list[UploadFile] = File(...),
) -> JSONResponse:
    """Upload extra files when user refuses auto-fill. Resumes the pipeline with all files."""
    handle = get_run_manager().get(job_id)
    if handle is None:
        raise HTTPException(404, "run not found")
    if handle.confirm_event is None or handle.confirm_event.is_set():
        raise HTTPException(400, "no pending confirmation for this run")
    if not files:
        raise HTTPException(400, "at least one file is required")

    saved: list[str] = []
    for f in files:
        name = Path(f.filename or "upload.bin").name
        dest = input_file_path(job_id, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(name)

    handle.additional_files = saved
    handle.confirm_decision = "upload"
    handle.confirm_event.set()
    logger.info("additional_files job=%s files=%s", job_id, saved)
    return JSONResponse({"ok": True, "files": saved})
