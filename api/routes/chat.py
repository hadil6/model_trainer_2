"""Chat with a fine-tuned LoRA model loaded in memory.

Endpoints
---------
POST /api/runs/{job_id}/chat/load    — load the adapter into VRAM (frees VRAM first)
POST /api/runs/{job_id}/chat/message — send a message, get a response
POST /api/runs/{job_id}/chat/unload  — release the model from VRAM
GET  /api/runs/{job_id}/chat/status  — check if the model is loaded
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/runs")

# ── Global singleton — one model in memory at a time ─────────────────────────
_state: dict = {
    "job_id":    None,
    "model":     None,
    "tokenizer": None,
    "device":    None,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_best_checkpoint(output_dir: str) -> str | None:
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
        return str(base) if (base / "adapter_config.json").exists() else None

    best_path, best_loss = None, float("inf")
    for cp in checkpoints:
        ts = cp / "trainer_state.json"
        if ts.exists():
            try:
                for entry in json.loads(ts.read_text(encoding="utf-8")).get("log_history", []):
                    if "eval_loss" in entry and entry["eval_loss"] < best_loss:
                        best_loss = entry["eval_loss"]
                        best_path = str(cp)
            except Exception:
                pass
    return best_path or str(checkpoints[-1])


def _flush_vram() -> None:
    try:
        from data.model_zoo.embeddings import release_sentence_transformer
        release_sentence_transformer()
    except Exception:
        pass
    try:
        from data.profilers.pdf_profiler import release_marker_models
        release_marker_models()
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _do_unload() -> None:
    if _state["model"] is not None:
        del _state["model"]
        _state["model"] = None
    if _state["tokenizer"] is not None:
        del _state["tokenizer"]
        _state["tokenizer"] = None
    _state["job_id"] = None
    _state["device"] = None
    _flush_vram()
    logger.info("chat: model unloaded")


def _do_load(job_id: str, model_id: str, adapter_path: str) -> dict:
    """Load base model + LoRA adapter. Returns info dict with device and free VRAM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Step 1 — free the currently loaded chat model (if any)
    _do_unload()

    # Step 2 — full VRAM cleanup: kills orphan workers, flushes cache, sleeps 2s
    # This is the same cleanup used before SWIFT training — the most thorough available.
    try:
        from services.vram_cleanup import cleanup_vram
        report = cleanup_vram(label="before_chat_load")
        logger.info(
            "chat: vram_cleanup done — freed=%.0f MB  before=%.0f MB  after=%.0f MB  errors=%s",
            report.get("freed_mb", 0),
            report.get("vram_before_mb") or 0,
            report.get("vram_after_mb") or 0,
            report.get("errors", []),
        )
    except Exception as exc:
        logger.warning("chat: vram_cleanup failed (non-fatal): %s", exc)
        _flush_vram()

    # Step 3 — measure free VRAM and decide device/precision
    free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3 if torch.cuda.is_available() else 0.0
    logger.info("chat: free VRAM after cleanup = %.2f GB", free_gb)

    _MIN_GPU_GB = 2.0
    if free_gb >= _MIN_GPU_GB:
        load_kwargs: dict = {
            "device_map":        "auto",
            "torch_dtype":       torch.float16,
            "trust_remote_code": True,
        }
        device = "gpu"
        logger.info("chat: will load on GPU (free=%.2f GB)", free_gb)
    else:
        load_kwargs = {
            "device_map":         "cpu",
            "torch_dtype":        torch.float32,
            "trust_remote_code":  True,
            "low_cpu_mem_usage":  True,
        }
        device = "cpu"
        logger.warning("chat: insufficient VRAM (%.2f GB < %.1f GB) — loading on CPU", free_gb, _MIN_GPU_GB)

    logger.info("chat: loading tokenizer %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    logger.info("chat: loading base model %s on %s", model_id, device)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        if device == "gpu":
            logger.warning("chat: GPU load failed (%s) — retrying on CPU as last resort", exc)
            _flush_vram()
            load_kwargs = {
                "device_map":         "cpu",
                "torch_dtype":        torch.float32,
                "trust_remote_code":  True,
                "low_cpu_mem_usage":  True,
            }
            device = "cpu"
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        else:
            raise

    logger.info("chat: loading adapter %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    _state["job_id"]    = job_id
    _state["model"]     = model
    _state["tokenizer"] = tokenizer
    _state["device"]    = device
    logger.info("chat: ready job=%s device=%s free_vram_gb=%.2f", job_id, device, free_gb)
    return {"device": device, "free_vram_gb": round(free_gb, 2)}


def _do_generate(message: str, max_new_tokens: int) -> str:
    import torch

    model     = _state["model"]
    tokenizer = _state["tokenizer"]

    prompt = f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    inputs   = tokenizer(prompt, return_tensors="pt").to(model.device)
    inp_len  = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )

    decoded = tokenizer.decode(out_ids[0][inp_len:], skip_special_tokens=False)
    decoded = re.sub(r"<\|im_end\|>.*", "", decoded, flags=re.DOTALL)
    return decoded.strip()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/{job_id}/chat/status")
async def chat_status(job_id: str) -> JSONResponse:
    loaded = _state["job_id"] == job_id and _state["model"] is not None
    return JSONResponse({
        "loaded": loaded,
        "device": _state["device"] if loaded else None,
    })


@router.post("/{job_id}/chat/load")
async def chat_load(job_id: str) -> JSONResponse:
    """Load the fine-tuned model. Frees all VRAM first."""
    from data.artifact_store import training_report_path, training_output_path

    rp = training_report_path(job_id)
    if not rp.exists():
        raise HTTPException(404, f"No training_report.json for job {job_id}")

    report     = json.loads(rp.read_text(encoding="utf-8"))
    model_id   = report.get("model_id", "")
    output_dir = (report.get("result") or {}).get("output_dir") or str(training_output_path(job_id))

    if not model_id:
        raise HTTPException(400, "model_id missing in training_report.json")

    # Find adapter — try direct path, then one level deeper (SWIFT timestamped subdir)
    adapter_path = _find_best_checkpoint(output_dir)
    if not adapter_path:
        base = Path(output_dir)
        for sub in (sorted(base.iterdir()) if base.exists() else []):
            if sub.is_dir():
                candidate = _find_best_checkpoint(str(sub))
                if candidate:
                    adapter_path = candidate
                    break

    if not adapter_path:
        raise HTTPException(400, f"No checkpoint found under {output_dir}")

    # Already loaded for this job — skip reload
    if _state["job_id"] == job_id and _state["model"] is not None:
        return JSONResponse({
            "status":       "already_loaded",
            "job_id":       job_id,
            "adapter_path": adapter_path,
            "device":       _state["device"],
        })

    loop = asyncio.get_event_loop()
    info = await loop.run_in_executor(None, lambda: _do_load(job_id, model_id, adapter_path))

    return JSONResponse({
        "status":        "loaded",
        "job_id":        job_id,
        "adapter_path":  adapter_path,
        "device":        info["device"],
        "free_vram_gb":  info["free_vram_gb"],
    })


class MessageBody(BaseModel):
    message:        str
    max_new_tokens: int = 256


@router.post("/{job_id}/chat/message")
async def chat_message(job_id: str, body: MessageBody) -> JSONResponse:
    """Generate a response from the loaded model."""
    if _state["job_id"] != job_id or _state["model"] is None:
        raise HTTPException(400, "Model not loaded — call POST /chat/load first")

    loop     = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: _do_generate(body.message, body.max_new_tokens)
    )
    return JSONResponse({"response": response})


@router.post("/{job_id}/chat/unload")
async def chat_unload(job_id: str) -> JSONResponse:
    """Release the model from memory and free VRAM."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _do_unload)
    return JSONResponse({"status": "unloaded", "job_id": job_id})
