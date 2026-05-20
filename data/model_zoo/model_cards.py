"""Model Zoo — built from SWIFT MODEL_MAPPING + HuggingFace Hub (real data only).

Public API
----------
get_all_cards()              -> list[dict]
get_card_by_id(model_id)     -> dict | None
get_cards_by_vram(max_gb)    -> list[dict]
refresh_zoo()                -> list[dict]
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

_ENRICHED_FILE = Path("artifacts/model_zoo_enriched.json")
_HF_API = HfApi()


# ── Memory estimation — physics-based, not arbitrary heuristics ───────────────
#
# VRAM = model_weights + adapter_weights + optimizer_states + overhead
#
# model_weights  = params × bytes_per_param
#   4-bit (QLoRA)  : 0.5 bytes/param  (NF4 via BitsAndBytes)
#   8-bit (int8)   : 1.0 bytes/param
#   bf16 (LoRA/full): 2.0 bytes/param
#
# adapter_weights = trainable_params × 4 bytes (kept in fp32 for stability)
#   LoRA variants  : ~1% of base model params (rank-16 adapters on attention)
#   Full fine-tune : 100% of params
#
# optimizer_states = 2 × adapter_weights  (AdamW: first + second moment)
#
# overhead        = 1.5 GB  (CUDA buffers, activations, KV cache at batch=1)

_BYTES_PER_PARAM: dict[str, float] = {
    "qlora":     0.5,
    "lora_int8": 1.0,
    "lora":      2.0,
    "full":      2.0,
}

_TRAINABLE_FRACTION: dict[str, float] = {
    "qlora":     0.01,
    "lora_int8": 0.01,
    "lora":      0.01,
    "full":      1.00,
}

_OVERHEAD_GB = 1.5


def _estimate_min_vram(params_b: float, method: str) -> float:
    """Return estimated minimum VRAM (GB) to fine-tune a model with the given method."""
    bpp  = _BYTES_PER_PARAM.get(method, 2.0)
    frac = _TRAINABLE_FRACTION.get(method, 0.01)

    model_gb    = params_b * bpp
    adapter_gb  = params_b * frac * 4.0
    optimizer_gb = adapter_gb * 2.0
    return round(model_gb + adapter_gb + optimizer_gb + _OVERHEAD_GB, 1)


# ── Dynamic PEFT method map — queried from SWIFT at runtime ──────────────────
#
# SWIFT tuner_type "lora" covers three quantization variants that this project
# exposes as separate methods (qlora = lora+quant4, lora_int8 = lora+quant8).
# We discover which base tuner types SWIFT supports, then expand them.

_PEFT_METHOD_MAP_CACHE: dict[str, dict] | None = None


def get_peft_method_map() -> dict[str, dict]:
    """Return {peft_method: {tuner_type, quant_bits?, quant_method?}} built
    dynamically from SWIFT's registered tuner types."""
    global _PEFT_METHOD_MAP_CACHE
    if _PEFT_METHOD_MAP_CACHE is not None:
        return _PEFT_METHOD_MAP_CACHE

    swift_tuners: set[str] = set()
    try:
        from swift.tuners import TunerType
        swift_tuners = {t.value for t in TunerType}
        logger.info("SWIFT tuner types (dynamic): %s", sorted(swift_tuners))
    except Exception as exc:
        logger.warning("Could not query SWIFT TunerType — using safe defaults: %s", exc)
        swift_tuners = {"lora", "full"}

    methods: dict[str, dict] = {}

    if "lora" in swift_tuners:
        methods["qlora"]     = {"tuner_type": "lora", "quant_bits": 4, "quant_method": "bnb"}
        methods["lora_int8"] = {"tuner_type": "lora", "quant_bits": 8, "quant_method": "bnb"}
        methods["lora"]      = {"tuner_type": "lora"}

    if "full" in swift_tuners:
        methods["full"] = {"tuner_type": "full"}

    # Include any additional SWIFT tuners (longlora, adalora, vera, …) as-is
    skip = {"lora", "full"}
    for tuner in sorted(swift_tuners - skip):
        if tuner not in methods:
            methods[tuner] = {"tuner_type": tuner}

    _PEFT_METHOD_MAP_CACHE = methods
    logger.info("PEFT method map built: %s", list(methods.keys()))
    return _PEFT_METHOD_MAP_CACHE


def _get_feasible_methods(params_b: float, vram_gb: float) -> list[str]:
    """Return methods from the dynamic PEFT map that fit within vram_gb."""
    return [
        method for method in get_peft_method_map()
        if _estimate_min_vram(params_b, method) <= vram_gb
    ]


_INCOMPATIBLE_ARCH_PATTERNS = (
    "mamba", "rwkv", "ssm", "state-space", "state_space",
    "retnet", "rnn", "lstm",
)

_INCOMPATIBLE_PIPELINES = {
    "feature-extraction",
    "sentence-similarity",
    "fill-mask",
}

_INCOMPATIBLE_TAG_HINTS = (
    "sentence-similarity", "sentence-transformers",
    "feature-extraction", "embeddings",
)

_PREQUANTIZED_SUFFIX_PATTERNS = (
    "-awq", "-gptq", "-gguf", "-fp8", "-int8", "-int4",
    "-w4a16", "-w8a16", "-w4", "-w8", "-nf4",
    "-bnb-4bit", "-bnb-8bit", "-4bit", "-8bit",
    "-compressed", "-marlin",
)

# Models/orgs with custom architectures incompatible with standard Transformers on Windows
# These use proprietary architectures (youtu, hunyuan, ernie...) not yet merged upstream
_BLACKLISTED_ORG_PREFIXES = (
    "tencent/",
    "baidu/",
    "ERNIE",
    "WisdomShell/",
)

_BLACKLISTED_ARCH_TYPES = (
    "youtu", "hunyuan", "ernie", "pangu", "chatglm",
)


def is_swift_lora_compatible(card: dict) -> tuple[bool, str]:
    raw_model_id = card.get("model_id") or ""
    model_id = raw_model_id.lower()
    arch = (card.get("swift_model_type") or "").lower()

    for prefix in _BLACKLISTED_ORG_PREFIXES:
        if raw_model_id.startswith(prefix):
            return False, f"blacklisted organization ({prefix.rstrip('/')})"

    for bad_arch in _BLACKLISTED_ARCH_TYPES:
        if bad_arch in arch or bad_arch in model_id:
            return False, f"unsupported proprietary architecture ({bad_arch})"

    for bad in _INCOMPATIBLE_ARCH_PATTERNS:
        if bad in arch or bad in model_id:
            return False, f"non-transformer architecture ({bad})"

    for suffix in _PREQUANTIZED_SUFFIX_PATTERNS:
        if suffix in model_id:
            return False, f"pre-quantized variant ({suffix.lstrip('-')})"

    if card.get("is_reward"):
        return False, "reward model (not an SFT target)"

    pipeline = (card.get("pipeline_tag") or "").lower()
    if pipeline in _INCOMPATIBLE_PIPELINES:
        return False, f"pipeline '{pipeline}' is not an SFT target"

    tags_lower = {str(t).lower() for t in (card.get("tags") or [])}
    for bad in _INCOMPATIBLE_TAG_HINTS:
        if bad in tags_lower:
            return False, f"tag '{bad}' indicates non-SFT use case"

    return True, ""


def _get_swift_models() -> list[dict]:
    try:
        from swift.model import MODEL_MAPPING
    except ImportError:
        logger.error("ms-swift not installed. Run: pip install ms-swift")
        return []

    seen: set[str] = set()
    records: list[dict] = []

    for model_type, meta in MODEL_MAPPING.items():
        if meta.is_multimodal:
            continue
        for group in meta.model_groups:
            for model in group.models:
                hf_id = getattr(model, "hf_model_id", None)
                if not hf_id or hf_id in seen:
                    continue
                seen.add(hf_id)
                records.append({
                    "model_id": hf_id,
                    "ms_model_id": getattr(model, "ms_model_id", None),
                    "swift_model_type": model_type,
                    "swift_template": meta.template or model_type,
                    "is_reward": bool(meta.is_reward),
                    "swift_task_type": str(meta.task_type) if meta.task_type else "causal_lm",
                })

    logger.info("SWIFT: %d unique text-only models found.", len(records))
    return records


def _fetch_config_json(model_id: str) -> dict:
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def _fetch_hf_metadata(model_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "parameters_b": None,
        "supported_languages": [],
        "license": "unknown",
        "pipeline_tag": "text-generation",
        "tags": [],
        "downloads": 0,
        "likes": 0,
        "context_length": None,
        "num_layers": None,
        "hidden_size": None,
        "hf_fetch_ok": False,
    }

    try:
        info = _HF_API.model_info(model_id, timeout=15)

        if info.safetensors and info.safetensors.total:
            result["parameters_b"] = round(info.safetensors.total / 1e9, 3)

        if info.card_data and info.card_data.language:
            langs = info.card_data.language
            if isinstance(langs, str):
                langs = [langs]
            result["supported_languages"] = sorted(set(langs))

        if info.card_data and getattr(info.card_data, "license", None):
            result["license"] = info.card_data.license

        result["pipeline_tag"] = info.pipeline_tag or "text-generation"
        result["tags"] = list(info.tags or [])
        result["downloads"] = info.downloads or 0
        result["likes"] = info.likes or 0
        result["hf_fetch_ok"] = True

    except Exception as exc:
        logger.debug("HF model_info failed for %s: %s", model_id, exc)
        return result

    cfg = _fetch_config_json(model_id)
    if cfg:
        for key in ("max_position_embeddings", "seq_length", "n_ctx", "model_max_length", "max_seq_len"):
            val = cfg.get(key)
            if val and isinstance(val, int) and 128 <= val <= 10_000_000:
                result["context_length"] = val
                break
        result["num_layers"] = cfg.get("num_hidden_layers") or cfg.get("n_layer")
        result["hidden_size"] = cfg.get("hidden_size") or cfg.get("d_model")

    return result


def _fetch_all_hf_metadata(model_ids: list[str], max_workers: int = 12) -> dict[str, dict]:
    results: dict[str, dict] = {}
    total = len(model_ids)
    logger.info("Fetching HF Hub metadata for %d models (workers=%d)…", total, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_hf_metadata, mid): mid for mid in model_ids}
        done = 0
        for future in as_completed(futures):
            mid = futures[future]
            results[mid] = future.result()
            done += 1
            if done % 100 == 0 or done == total:
                ok = sum(1 for v in results.values() if v.get("hf_fetch_ok"))
                logger.info("  HF fetch: %d/%d done, %d successful", done, total, ok)

    return results


def _tasks_from_hf(pipeline_tag: str, tags: list[str], swift_task_type: str, is_reward: bool) -> list[str]:
    tasks: set[str] = set()
    tag_map = {
        "text-generation": "text-generation",
        "text2text-generation": "text2text-generation",
        "text-classification": "classification",
        "token-classification": "token-classification",
        "question-answering": "question-answering",
        "summarization": "summarization",
        "translation": "translation",
        "fill-mask": "fill-mask",
        "feature-extraction": "feature-extraction",
        "sentence-similarity": "sentence-similarity",
    }
    if pipeline_tag in tag_map:
        tasks.add(tag_map[pipeline_tag])
    if swift_task_type == "reranker":
        tasks.add("reranking")
    if swift_task_type == "prm":
        tasks.add("process-reward")
    if is_reward:
        tasks.add("reward-modeling")
    for tag in tags:
        if tag in tag_map:
            tasks.add(tag_map[tag])
        if tag in ("conversational", "chat"):
            tasks.add("instruction-following")
        if tag in ("code", "coding"):
            tasks.add("code-generation")
    return sorted(tasks) if tasks else ["text-generation"]


def _build_card(swift_rec: dict, hf: dict) -> dict:
    params_b = hf.get("parameters_b")
    langs = hf.get("supported_languages") or []
    tasks = _tasks_from_hf(
        hf.get("pipeline_tag", "text-generation"),
        hf.get("tags", []),
        swift_rec["swift_task_type"],
        swift_rec["is_reward"],
    )

    if params_b is not None:
        min_vram_per_method = {
            method: _estimate_min_vram(params_b, method)
            for method in get_peft_method_map()
        }
        vram_fields = {
            "min_vram_per_method": min_vram_per_method,
            "feasible_peft_methods": _get_feasible_methods(params_b, 4.0),
        }
    else:
        vram_fields = {
            "min_vram_per_method": {},
            "feasible_peft_methods": [],
        }

    card = {
        "model_id": swift_rec["model_id"],
        "ms_model_id": swift_rec.get("ms_model_id"),
        "swift_model_type": swift_rec["swift_model_type"],
        "swift_template": swift_rec["swift_template"],
        "parameters_b": params_b,
        "supported_languages": langs,
        "supported_tasks": tasks,
        "pipeline_tag": hf.get("pipeline_tag", "text-generation"),
        "license": hf.get("license", "unknown"),
        "tags": hf.get("tags", []),
        "downloads": hf.get("downloads", 0),
        "likes": hf.get("likes", 0),
        "context_length": hf.get("context_length"),
        "num_layers": hf.get("num_layers"),
        "hidden_size": hf.get("hidden_size"),
        "is_reward": swift_rec["is_reward"],
        "hf_fetch_ok": hf.get("hf_fetch_ok", False),
        **vram_fields,
    }
    compatible, reason = is_swift_lora_compatible(card)
    card["lora_compatible"] = compatible
    card["lora_incompatible_reason"] = reason
    return card


_CARDS_CACHE: list[dict] | None = None
_CARDS_BY_ID: dict[str, dict] = {}


def _load_zoo() -> list[dict]:
    global _CARDS_CACHE, _CARDS_BY_ID
    if _CARDS_CACHE is not None:
        return _CARDS_CACHE

    if _ENRICHED_FILE.exists():
        try:
            data: list[dict] = json.loads(_ENRICHED_FILE.read_text(encoding="utf-8"))
            if data:
                _CARDS_CACHE = data
                _CARDS_BY_ID = {c["model_id"]: c for c in data}
                logger.info("Loaded %d model cards from enriched cache.", len(data))
                return _CARDS_CACHE
        except Exception as exc:
            logger.warning("Cache corrupt (%s) — rebuilding.", exc)

    return _build_zoo()


def _build_zoo() -> list[dict]:
    global _CARDS_CACHE, _CARDS_BY_ID

    swift_models = _get_swift_models()
    if not swift_models:
        _CARDS_CACHE = []
        return _CARDS_CACHE

    all_ids = [r["model_id"] for r in swift_models]
    hf_data = _fetch_all_hf_metadata(all_ids, max_workers=12)

    cards = [_build_card(rec, hf_data.get(rec["model_id"], {})) for rec in swift_models]

    _ENRICHED_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ENRICHED_FILE.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")

    _CARDS_CACHE = cards
    _CARDS_BY_ID = {c["model_id"]: c for c in cards}
    logger.info("Built zoo: %d cards saved.", len(cards))
    return _CARDS_CACHE


def get_all_cards() -> list[dict]:
    return _load_zoo()


def get_card_by_id(model_id: str) -> dict | None:
    _load_zoo()
    return _CARDS_BY_ID.get(model_id)


def get_cards_by_vram(max_vram_gb: float) -> list[dict]:
    result = []
    for c in _load_zoo():
        params = c.get("parameters_b")
        if params is None:
            continue
        compatible, _ = is_swift_lora_compatible(c)
        if not compatible:
            continue
        methods = _get_feasible_methods(params, max_vram_gb)
        if methods:
            card = dict(c)
            card["feasible_peft_methods"] = methods
            result.append(card)
    return result


def refresh_zoo() -> list[dict]:
    global _CARDS_CACHE, _CARDS_BY_ID
    _CARDS_CACHE = None
    _CARDS_BY_ID = {}
    if _ENRICHED_FILE.exists():
        _ENRICHED_FILE.unlink()
    return _build_zoo()
