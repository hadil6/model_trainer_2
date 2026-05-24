"""Model selection service — FAISS retrieval + LLM ranking.

Public API
----------
SelectionInput    — input dataclass
SelectionResult   — output dataclass
ModelSelector     — main class; call selector.select(inp: SelectionInput) -> SelectionResult
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from agents.llm_client import get_llm_client
from core.config import get_settings
from data.model_zoo.embeddings import build_query_text, search_similar_models
from data.model_zoo.model_cards import get_card_by_id

logger = logging.getLogger(__name__)


def _is_model_fully_cached(model_id: str, min_size_mb: float = 100.0) -> bool:
    """True if the HuggingFace cache for `model_id` contains the actual weights.

    A model whose cache directory only contains tokenizer/config files (~10 MB)
    is treated as NOT cached — its model.safetensors is missing and would need
    to be downloaded. We require at least one big file (~100 MB+) to consider
    the model usable without network.
    """
    import os
    from pathlib import Path
    cache_root = Path(os.environ.get("HF_HUB_CACHE",
                      Path.home() / ".cache" / "huggingface" / "hub"))
    folder_name = "models--" + model_id.replace("/", "--")
    model_dir = cache_root / folder_name
    if not model_dir.exists():
        return False
    try:
        total_bytes = sum(
            f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
        )
        return total_bytes >= min_size_mb * 1024 * 1024
    except Exception:
        return False


def _prefer_cached_models(candidates: list[dict], inp_objective: str = "balanced") -> list[dict]:
    """Reorder candidates so already-cached models come first.

    On Windows + small GPU (< 6 GB total) or when MODEL_SELECT_PREFER_CACHED=1,
    candidates whose weights are NOT yet downloaded are filtered out entirely
    (to avoid risky downloads that hang). Otherwise we just sort cached first
    without removing anything.
    """
    import os
    force_cached = os.environ.get("MODEL_SELECT_PREFER_CACHED", "").lower() in ("1", "true", "yes")
    auto_cached  = False
    try:
        import platform, torch
        if (platform.system() == "Windows"
                and torch.cuda.is_available()
                and torch.cuda.get_device_properties(0).total_memory / 1e9 < 6.0):
            auto_cached = True
    except Exception:
        pass

    cached, not_cached = [], []
    for c in candidates:
        if _is_model_fully_cached(c["model_id"]):
            c = {**c, "_already_cached": True}
            cached.append(c)
        else:
            c = {**c, "_already_cached": False}
            not_cached.append(c)

    if (force_cached or auto_cached) and cached:
        logger.info(
            "model_selection: filtered to %d cached model(s) (Windows + small GPU or explicit opt-in) — "
            "skipping %d uncached candidate(s)",
            len(cached), len(not_cached),
        )
        return cached

    if cached:
        logger.info(
            "model_selection: %d cached candidate(s) moved to top of list (over %d uncached)",
            len(cached), len(not_cached),
        )
    return cached + not_cached


@dataclass
class SelectionInput:
    task: str
    domain: str
    description: str
    data_language: str
    modality: str
    total_rows: int | None
    total_chars: int | None
    gpu_vram_gb: int
    objective: str = "balanced"          # "speed" | "balanced" | "performance"
    exclude_models: list[str] | None = None  # model_ids already tried — skip them


@dataclass
class SelectionResult:
    model_id: str
    peft_method: str
    reasoning: str
    swift_config: dict
    candidates: list[dict]

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "peft_method": self.peft_method,
            "reasoning": self.reasoning,
            "swift_config": self.swift_config,
            "candidates": self.candidates,
        }


class ModelSelector:
    """FAISS retrieval + LLM ranking for fine-tuning model selection."""

    def select(self, inp: SelectionInput) -> SelectionResult:
        settings = get_settings()
        query = build_query_text(
            task=inp.task,
            domain=inp.domain,
            description=inp.description,
            data_language=inp.data_language,
            modality=inp.modality,
            total_rows=inp.total_rows,
            total_chars=inp.total_chars,
            gpu_vram_gb=inp.gpu_vram_gb,
            objective=inp.objective,
        )
        logger.info("model_selection query=%s", query[:120])

        candidates = search_similar_models(
            query_text=query,
            max_vram_gb=float(inp.gpu_vram_gb),
            top_k=6,  # fetch more so exclusions still leave enough
            embedding_model_name=settings.embedding_model,
        )

        # Remove already-tried models when reselecting
        if inp.exclude_models:
            excluded = set(inp.exclude_models)
            candidates = [c for c in candidates if c["model_id"] not in excluded]
            logger.info("reselect: excluded %s → %d candidates remain", excluded, len(candidates))

        # On constrained / unreliable setups, prefer (or restrict to) models
        # whose weights are already in the local HuggingFace cache. This avoids
        # the silent download hang we keep hitting on Windows + 4 GB.
        candidates = _prefer_cached_models(candidates, inp.objective)

        candidates = candidates[:3]  # keep top-3 after exclusion

        if not candidates:
            logger.warning("No candidates found for VRAM=%dGB", inp.gpu_vram_gb)
            return SelectionResult(
                model_id="",
                peft_method="qlora",
                reasoning=f"No models fit within {inp.gpu_vram_gb}GB VRAM.",
                swift_config={},
                candidates=[],
            )

        selection = self._rank_with_llm(candidates, inp)
        model_id    = selection.get("selected_model_id", candidates[0]["model_id"])
        peft_method = selection.get("selected_peft_method", "qlora")

        # Validate model_id is actually in candidates
        candidate_ids = {c["model_id"] for c in candidates}
        if model_id not in candidate_ids:
            logger.warning(
                "LLM returned unknown model_id=%s — falling back to top candidate", model_id
            )
            model_id    = candidates[0]["model_id"]
            peft_method = (candidates[0].get("feasible_peft_methods") or ["qlora"])[0]

        swift_config = self._build_swift_config(model_id, peft_method, inp)

        logger.info("model_selected model=%s peft=%s", model_id, peft_method)
        return SelectionResult(
            model_id=model_id,
            peft_method=peft_method,
            reasoning=selection.get("reasoning", ""),
            swift_config=swift_config,
            candidates=[
                {
                    "rank": c.get("rank"),
                    "model_id": c["model_id"],
                    "parameters_b": c.get("parameters_b"),
                    "supported_tasks": c.get("supported_tasks"),
                    "supported_languages": c.get("supported_languages"),
                    "context_length": c.get("context_length"),
                    "feasible_peft_methods": c.get("feasible_peft_methods"),
                    "similarity_score": c.get("similarity_score"),
                }
                for c in candidates
            ],
        )

    def _rank_with_llm(self, candidates: list[dict], inp: SelectionInput) -> dict:
        candidates_text = ""
        for i, c in enumerate(candidates, 1):
            params   = c.get("parameters_b")
            langs    = ", ".join(c.get("supported_languages") or ["unknown"])
            tasks    = ", ".join(c.get("supported_tasks") or ["unknown"])
            methods  = c.get("feasible_peft_methods") or ["qlora"]
            ctx      = c.get("context_length")
            cache_tag = " [ALREADY CACHED LOCALLY — strongly preferred]" if c.get("_already_cached") else ""
            candidates_text += (
                f"\nCandidate {i}: {c['model_id']}{cache_tag}\n"
                f"  Parameters:       {params}B\n"
                f"  Supported tasks:  {tasks}\n"
                f"  Languages:        {langs}\n"
                f"  Context length:   {ctx or 'unknown'} tokens\n"
                f"  Feasible PEFT:    {', '.join(methods)}\n"
                f"  HF downloads:     {c.get('downloads', 0):,}\n"
                f"  License:          {c.get('license', 'unknown')}\n"
                f"  Similarity score: {c.get('similarity_score', 'N/A')}\n"
            )

        total_rows_str  = str(inp.total_rows)  if inp.total_rows  else "unknown"
        total_chars_str = str(inp.total_chars) if inp.total_chars else "unknown"
        total_words_str = str(inp.total_chars // 5) if inp.total_chars else "unknown"

        objective_rules = {
            "speed":       "Prioritise smaller models (< 3B) and qlora — fastest training, lowest VRAM.",
            "balanced":    "Balance quality and speed. Prefer 1–3B models with lora or qlora.",
            "performance": "Prioritise quality. Prefer larger models and lora/full when VRAM allows.",
        }.get(inp.objective, "Balance quality and speed.")

        from data.model_zoo.model_cards import get_peft_method_map
        available_methods = "|".join(get_peft_method_map().keys())

        system_prompt = (
            "You are an LLM fine-tuning expert.\n"
            "Select the best model AND PEFT method from the candidates below.\n\n"
            "=== PEFT selection rules ===\n"
            "The selected method MUST appear in the candidate's 'Feasible PEFT' list.\n"
            "Each candidate's 'Feasible PEFT' list was computed using physics-based VRAM\n"
            "estimation (model weights + adapter + optimizer states + overhead).\n\n"
            "Dataset size influence:\n"
            "  - dataset < 50 rows   → lowest-VRAM method (overfit risk)\n"
            "  - dataset 50–500 rows → lora_int8 or qlora\n"
            "  - dataset > 500 rows  → lora if VRAM allows\n"
            "  - dataset > 5 000 rows → full if it appears in Feasible PEFT\n\n"
            f"User objective — {inp.objective}: {objective_rules}\n\n"
            "Also evaluate: instruct vs base, context_length, recency, license.\n\n"
            "Reply ONLY with valid JSON:\n"
            f'{{"selected_model_id": "...", "selected_peft_method": "{available_methods}", "reasoning": "one sentence"}}'
        )

        user_content = (
            f"Task: {inp.task}\n"
            f"Domain: {inp.domain}\n"
            f"Data language: {inp.data_language}\n"
            f"Dataset rows: {total_rows_str} | chars: {total_chars_str} | ~words: {total_words_str}\n"
            f"GPU VRAM: {inp.gpu_vram_gb}GB\n"
            f"Objective: {inp.objective}\n\n"
            f"Candidates (pre-filtered by VRAM + semantic similarity):\n{candidates_text}"
        )

        try:
            raw = get_llm_client().complete(system=system_prompt, user=user_content)
            m = re.search(r"\{[\s\S]*?\}", raw)
            if m:
                return json.loads(m.group(0))
        except Exception as exc:
            logger.warning("llm_ranking_failed: %s — using top FAISS candidate", exc)

        top = candidates[0]
        methods = top.get("feasible_peft_methods") or ["qlora"]
        return {
            "selected_model_id":    top["model_id"],
            "selected_peft_method": methods[0],
            "reasoning": "LLM unavailable — defaulted to top FAISS similarity result.",
        }

    def _build_swift_config(self, model_id: str, peft_method: str, inp: SelectionInput) -> dict:
        from services.training.hparams import get_hparams

        card = get_card_by_id(model_id)
        if card is None:
            return {}

        # Use total_rows as a proxy for n_pairs at selection time (actual count
        # is computed again in trainer.py once qa_final.jsonl is written).
        n_pairs_estimate = inp.total_rows or 500

        hparams = get_hparams(
            model_name=model_id,
            peft=peft_method,
            n_pairs=n_pairs_estimate,
            vram_gb=inp.gpu_vram_gb,
            task=inp.task,
        )

        # Map peft_method → SWIFT sft_type
        tuner_map = {
            "qlora":     "lora",
            "lora_int8": "lora",
            "lora":      "lora",
            "dora":      "lora",
            "full":      "full",
            "auto":      "lora",
        }

        config: dict = {
            "model":               model_id,
            "swift_model_type":    card.get("swift_model_type"),
            "sft_type":            tuner_map.get(peft_method, "lora"),
            "quantization_required": inp.gpu_vram_gb <= 8,
        }

        # Quantization flags
        if peft_method == "qlora":
            config["quant_method"] = "bnb"
            config["quant_bits"]   = 4
        elif peft_method == "lora_int8":
            config["quant_method"] = "bnb"
            config["quant_bits"]   = 8

        # Merge evidence-based hparams (exclude internal metadata keys)
        config.update({k: v for k, v in hparams.items() if not k.startswith("_")})

        # Preview CLI command (informational only — actual command built in swift_runner)
        skip = {"swift_model_type", "quantization_required", "sft_type"}
        parts = [f"swift sft --model {model_id} --sft_type {config['sft_type']}"]
        for k, v in config.items():
            if k in skip:
                continue
            parts.append(f"--{k} {str(v).lower() if isinstance(v, bool) else v}")
        config["swift_cli_command"] = " ".join(parts)

        return config
