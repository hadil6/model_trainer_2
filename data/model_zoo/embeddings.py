"""FAISS-based cosine similarity search over the model zoo.

Public API
----------
build_query_text(task, domain, description, data_language, modality,
                 total_rows, gpu_vram_gb) -> str

search_similar_models(query_text, max_vram_gb, top_k,
                      embedding_model_name) -> list[dict]
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_ST_MODEL: Any = None
_ST_MODEL_NAME: str = ""

_INDEX: Any = None
_INDEX_CARDS: list[dict] = []
_INDEX_MODEL_NAME: str = ""


def _faiss():
    try:
        import faiss
        return faiss
    except ImportError as e:
        raise ImportError("faiss-cpu required: pip install faiss-cpu") from e


def _get_st(model_name: str) -> Any:
    global _ST_MODEL, _ST_MODEL_NAME
    if _ST_MODEL is not None and _ST_MODEL_NAME == model_name:
        return _ST_MODEL
    from sentence_transformers import SentenceTransformer
    logger.info("Loading sentence-transformer: %s", model_name)
    _ST_MODEL = SentenceTransformer(model_name)
    _ST_MODEL_NAME = model_name
    return _ST_MODEL


def release_sentence_transformer() -> None:
    """Move the cached sentence transformer off GPU and delete it.

    Call this before loading a large model for inference/evaluation so that
    the sentence transformer does not compete for VRAM.
    """
    global _ST_MODEL, _ST_MODEL_NAME
    import gc
    if _ST_MODEL is None:
        return
    try:
        _ST_MODEL.to("cpu")
    except Exception:
        pass
    del _ST_MODEL
    _ST_MODEL = None
    _ST_MODEL_NAME = ""
    gc.collect()
    logger.info("Sentence transformer released from GPU.")


def _needs_e5_prefix(model_name: str) -> bool:
    lower = model_name.lower()
    return "e5" in lower and "intfloat" in lower


def build_query_text(
    task: str,
    domain: str,
    description: str,
    data_language: str,
    modality: str,
    total_rows: int | None,
    gpu_vram_gb: int,
    total_chars: int | None = None,
    objective: str = "balanced",
) -> str:
    # Derive dataset size description from chars (PDFs) or rows (CSV)
    total_words = (total_chars // 5) if total_chars else None
    size_ref    = total_words or total_rows

    if size_ref is None:
        size_desc = "unknown dataset size"
    elif size_ref < 500:
        size_desc = "very small dataset"
    elif size_ref < 5_000:
        size_desc = "small dataset"
    elif size_ref < 50_000:
        size_desc = "medium dataset"
    else:
        size_desc = "large dataset"

    parts = [
        f"Task: {task}.",
        f"Domain: {domain}." if domain else "",
        f"Description: {description}." if description else "",
        f"Data language: {data_language}.",
        f"Modality: {modality}.",
        f"Dataset: {size_desc}.",
        f"Objective: {objective}.",
        f"Hardware: {gpu_vram_gb}GB VRAM available.",
        "Select a model that fits within VRAM and performs well on the task.",
    ]
    return " ".join(p for p in parts if p)


def _card_to_text(card: dict) -> str:
    tasks = ", ".join(card.get("supported_tasks") or ["text-generation"])
    langs = ", ".join(card.get("supported_languages") or ["en"])
    tags = ", ".join((card.get("tags") or [])[:10])
    return (
        f"Model: {card.get('model_id', '')}. "
        f"Parameters: {card.get('parameters_b', 0)}B. "
        f"Architecture: {card.get('swift_model_type', 'transformer')}. "
        f"Supported tasks: {tasks}. "
        f"Supported languages: {langs}. "
        f"Context length: {card.get('context_length', 4096)} tokens. "
        f"Min VRAM for QLoRA: {card.get('min_vram_gb_qlora', 2.0)}GB. "
        f"Tags: {tags}."
    ).strip()


def _build_index(cards: list[dict], model_name: str) -> Any:
    faiss = _faiss()
    st = _get_st(model_name)

    logger.info("Encoding %d model cards…", len(cards))
    texts = [_card_to_text(c) for c in cards]
    if _needs_e5_prefix(model_name):
        texts = [f"passage: {t}" for t in texts]
    vecs: np.ndarray = st.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    logger.info("FAISS index ready: %d vectors, dim=%d.", index.ntotal, dim)
    return index


def _get_index(model_name: str) -> tuple[Any, list[dict]]:
    global _INDEX, _INDEX_CARDS, _INDEX_MODEL_NAME

    if _INDEX is not None and _INDEX_MODEL_NAME == model_name:
        return _INDEX, _INDEX_CARDS

    from data.model_zoo.model_cards import get_all_cards
    cards = get_all_cards()
    if not cards:
        raise ValueError("Model zoo is empty.")

    _INDEX = _build_index(cards, model_name)
    _INDEX_CARDS = cards
    _INDEX_MODEL_NAME = model_name
    return _INDEX, _INDEX_CARDS


def search_similar_models(
    query_text: str,
    max_vram_gb: float = 4.0,
    top_k: int = 3,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    faiss = _faiss()
    index, all_cards = _get_index(embedding_model_name)

    from data.model_zoo.model_cards import _get_feasible_methods, is_swift_lora_compatible

    eligible_idx_list: list[int] = []
    skipped_reasons: dict[str, int] = {}
    for i, c in enumerate(all_cards):
        if c.get("parameters_b") is None:
            continue
        compatible, reason = is_swift_lora_compatible(c)
        if not compatible:
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue
        if not _get_feasible_methods(c["parameters_b"], max_vram_gb):
            continue
        eligible_idx_list.append(i)

    if skipped_reasons:
        logger.info(
            "Filtered out %d incompatible models: %s",
            sum(skipped_reasons.values()),
            ", ".join(f"{k} ({v})" for k, v in skipped_reasons.items()),
        )

    eligible_idx = np.array(eligible_idx_list)
    if len(eligible_idx) == 0:
        logger.warning("No compatible models fit within %.1f GB VRAM.", max_vram_gb)
        return []

    sub_vecs = np.vstack([index.reconstruct(int(i)) for i in eligible_idx])
    dim = sub_vecs.shape[1]
    sub_idx = faiss.IndexFlatIP(dim)
    sub_idx.add(sub_vecs)

    st = _get_st(embedding_model_name)
    q_input = (
        f"query: {query_text}"
        if _needs_e5_prefix(embedding_model_name)
        else query_text
    )
    q_vec: np.ndarray = st.encode(
        [q_input],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    k = min(top_k, len(eligible_idx))
    scores, local_ids = sub_idx.search(q_vec, k)

    results = []
    for rank, (lid, score) in enumerate(zip(local_ids[0], scores[0]), start=1):
        if lid < 0:
            continue
        card = dict(all_cards[int(eligible_idx[lid])])
        card["similarity_score"] = round(float(score), 4)
        card["rank"] = rank
        card["feasible_peft_methods"] = _get_feasible_methods(
            card.get("parameters_b", 1.0), max_vram_gb
        )
        results.append(card)

    return results
