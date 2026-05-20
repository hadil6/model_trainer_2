"""Text cleaning pipeline for PDF/TXT content.

Public API
----------
clean_text(text, filename) -> str
"""

from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

# ── DataFlow operators (optional — regex fallback if not installed) ────────────
try:
    import pandas as pd
    from dataflow.utils.storage import DummyStorage
    from dataflow.operators.general_text import (
        HtmlUrlRemoverRefiner,
        RemoveEmojiRefiner,
        RemoveExtraSpacesRefiner,
        SymbolWordRatioFilter,
        UniqueWordsFilter,
        WordNumberFilter,
    )
    _DATAFLOW_AVAILABLE = True
except ImportError:
    _DATAFLOW_AVAILABLE = False


def _strip_marker_artifacts(text: str) -> str:
    """Remove marker-pdf markdown artifacts before chunking."""
    # Image placeholders: ! [](_page_X_Picture_Y.jpeg) or ! [alt](_page...)
    text = re.sub(r'!\s*\[[^\]]*\]\([^)]*\.(jpeg|jpg|png|gif|webp|svg|figure|jpeg)[^)]*\)', '', text, flags=re.IGNORECASE)
    # Bare image lines with no alt text: [](_page_X_Figure_Y.jpeg)
    text = re.sub(r'^\s*\[\]\([^)]*\.(jpeg|jpg|png|gif|webp|svg|figure)[^)]*\)\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # HTML span anchors: <span id="page-3-0"></span>
    text = re.sub(r'<span[^>]*>.*?</span>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Internal reference links: [2](#page-9-0) → keep just "2"; [Docker](#page-5) → "Docker"
    text = re.sub(r'\[([^\]]*)\]\(#[^)]+\)', r'\1', text)
    # Superscript footnote markers: <sup>2</sup>
    text = re.sub(r'<sup>\d+</sup>', '', text, flags=re.IGNORECASE)
    # Collapse multiple blank lines left by removals
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_text(text: str, filename: str = "") -> str:
    """Clean raw extracted text: remove noise, boilerplate, and control chars."""
    # Step 0: strip marker-pdf markdown artifacts
    text = _strip_marker_artifacts(text)

    # Step 1: basic pre-clean (fast regex)
    text = text.replace("﻿", "")
    text = "".join(c for c in text if ord(c) >= 32 or c in "\n\t\r")
    for old, new in [("\r\n", "\n"), ("\r", "\n"), ("\x0c", "\n")]:
        text = text.replace(old, new)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    if not _DATAFLOW_AVAILABLE:
        logger.debug("dataflow_unavailable — using regex fallback for %s", filename)
        return _clean_text_regex(text)

    # Step 2: paragraph-level DataFlow pipeline
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return text

    try:
        df = pd.DataFrame({"text": paragraphs})
        storage = DummyStorage()
        storage.set_data(df)

        RemoveExtraSpacesRefiner().run(storage=storage, input_key="text")
        RemoveEmojiRefiner().run(storage=storage, input_key="text")
        HtmlUrlRemoverRefiner().run(storage=storage, input_key="text")
        WordNumberFilter(min_words=5, max_words=50000).run(storage=storage, input_key="text")
        SymbolWordRatioFilter(threshold=0.4).run(storage=storage, input_key="text")
        UniqueWordsFilter(threshold=0.05).run(storage=storage, input_key="text")

        result_df = storage.read("dataframe")
        cleaned = "\n\n".join(str(t) for t in result_df["text"].tolist() if str(t).strip())

        # Safety: if DataFlow over-removed content, fall back
        if len(cleaned) < len(text) * 0.10:
            logger.warning(
                "dataflow_clean over-removed file=%s — keeping pre-cleaned text", filename
            )
            return text

        return cleaned

    except Exception as exc:
        logger.warning(
            "dataflow_clean_failed file=%s error=%s — fallback to pre-cleaned text", filename, exc
        )
        return text


def _clean_text_regex(text: str) -> str:
    cleaned: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if re.match(r"^[\-\s]*\d{1,4}[\-\s]*$", s) and len(s) <= 8:
            continue
        if re.match(r"^page\s+\d+(\s+of\s+\d+)?$", s, re.IGNORECASE):
            continue
        if s and not re.search(r"[a-zA-Z0-9؀-ۿ]", s):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    lines = text.split("\n")
    non_empty = [ln.strip() for ln in lines if len(ln.strip()) > 3]
    if len(non_empty) >= 10:
        threshold = max(3, int(len(non_empty) * 0.02))
        repeated = {ln for ln, cnt in Counter(non_empty).items() if cnt >= threshold}
        if repeated:
            lines = [ln for ln in lines if ln.strip() not in repeated]

    text = "\n".join(lines)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
