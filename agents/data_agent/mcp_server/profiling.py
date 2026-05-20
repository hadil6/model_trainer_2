"""Profiling tool implementations (plain functions — registered on MCP app in server.py)."""

from __future__ import annotations

import json
import logging
import re

from agents.llm_client import get_llm_client
from agents.data_agent.services.cleaning import clean_text
from agents.data_agent.services.profiling import profile_file, summarize_profile
from data.artifact_store import (
    cleaned_path,
    input_file_path,
    merged_corpus_path,
    ocr_review_path,
    profile_path,
    raw_extraction_path,
    read_all_profiles,
    readiness_path,
    write_json,
)
from data.feasibility import check_prompt_vagueness, evaluate_feasibility

logger = logging.getLogger(__name__)


def tool_profile_file(job_id: str, filename: str) -> dict:
    """Profile an uploaded file. Returns file_kind, row/word counts, warnings, risk_flags.

    If a profile already exists on disk (e.g. created by the orchestrator's profile_files
    step), it is returned immediately without re-running the heavy extraction pipeline.
    """
    pp = profile_path(job_id, filename)
    if pp.exists():
        try:
            cached = json.loads(pp.read_text(encoding="utf-8"))
            logger.info("profile_cache_hit job=%s file=%s — skipping re-extraction", job_id, filename)
            return {
                "file_kind": cached.get("file_kind"),
                "row_count": (cached.get("metadata") or {}).get("row_count"),
                "word_count": (cached.get("metadata") or {}).get("word_count"),
                "warnings":   cached.get("warnings", []),
                "risk_flags": cached.get("risk_flags", []),
                "cached":     True,
            }
        except Exception:
            pass  # corrupt cache — fall through to re-profile

    path = input_file_path(job_id, filename)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    result = profile_file(path)
    write_json(pp, result)

    if result.get("file_kind") in ("pdf", "txt") and result.get("full_text"):
        ext_path = raw_extraction_path(job_id, filename)
        ext_path.parent.mkdir(parents=True, exist_ok=True)
        ext_path.write_text(result["full_text"], encoding="utf-8")
        logger.info("raw_extraction saved job=%s file=%s chars=%d", job_id, filename, len(result["full_text"]))

    logger.info("profiled job=%s file=%s kind=%s", job_id, filename, result.get("file_kind"))
    return {
        "file_kind": result.get("file_kind"),
        "row_count": (result.get("metadata") or {}).get("row_count"),
        "word_count": (result.get("metadata") or {}).get("word_count"),
        "warnings":   result.get("warnings", []),
        "risk_flags": result.get("risk_flags", []),
    }


def tool_summarize_file(job_id: str, filename: str, task: str) -> dict:
    """Generate a semantic summary of a file profile using the LLM."""
    p = profile_path(job_id, filename)
    if not p.exists():
        raise FileNotFoundError(f"Profile not found — run profile_file_tool first: {p}")

    profile = json.loads(p.read_text(encoding="utf-8"))
    summary = summarize_profile(profile, task, get_llm_client())
    profile["semantic_summary"] = summary
    write_json(p, profile)
    logger.info("summarized job=%s file=%s", job_id, filename)
    return {"summary_preview": summary[:300]}


def tool_clean_text(job_id: str, filename: str) -> dict:
    """Clean extracted text from a PDF or TXT file."""
    p = profile_path(job_id, filename)
    if not p.exists():
        raise FileNotFoundError(f"Profile not found — run profile_file_tool first: {p}")

    profile = json.loads(p.read_text(encoding="utf-8"))
    file_kind = profile.get("file_kind", "")
    if file_kind not in ("pdf", "txt"):
        return {"skipped": True, "reason": f"{file_kind.upper()} does not need text cleaning"}

    raw_text = profile.get("full_text", "")
    if file_kind == "pdf":
        ocr_pages: dict = profile.get("ocr_pages", {})
        if ocr_pages:
            ocr_combined = "\n\n".join(ocr_pages.values())
            raw_text = (raw_text + "\n\n" + ocr_combined).strip() if raw_text else ocr_combined

    original_chars = len(raw_text)
    cleaned = clean_text(raw_text, filename)
    cleaned_chars = len(cleaned)

    out = cleaned_path(job_id, filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cleaned, encoding="utf-8")

    removed_pct = round((1 - cleaned_chars / max(1, original_chars)) * 100, 1)
    logger.info("cleaned job=%s file=%s removed_pct=%.1f%%", job_id, filename, removed_pct)
    return {"original_chars": original_chars, "cleaned_chars": cleaned_chars, "removed_pct": removed_pct}


def tool_review_ocr_relevance(job_id: str, filename: str, domain: str) -> dict:
    """Review OCR-extracted pages for domain relevance."""
    p = profile_path(job_id, filename)
    if not p.exists():
        raise FileNotFoundError(f"Profile not found for {filename}")

    profile = json.loads(p.read_text(encoding="utf-8"))
    if profile.get("file_kind") != "pdf":
        return {"skipped": True, "reason": "Only applicable to PDF files"}

    pages = profile.get("metadata", {}).get("pages", [])
    if not pages:
        return {"total": 0, "relevant": 0, "rejected": 0}

    llm = get_llm_client()
    relevant = 0
    rejected = 0
    review_results = []

    for page in pages:
        text = page.get("text", "")
        page_num = page.get("page_num", 0)
        if len(text.split()) < 20:
            rejected += 1
            review_results.append({"page": page_num, "relevant": False, "reason": "too_short"})
            continue
        try:
            verdict = llm.complete(
                system="You are a document relevance filter.",
                user=(
                    f"Domain: {domain}\n"
                    f"Page text (first 500 chars): {text[:500]}\n\n"
                    "Is this page relevant to the domain? "
                    'Reply ONLY with JSON: {"relevant": true/false, "reason": "..."}'
                ),
            )
            m = re.search(r"\{[\s\S]*?\}", verdict)
            is_rel = bool(json.loads(m.group(0)).get("relevant", True)) if m else True
        except Exception:
            is_rel = True

        if is_rel:
            relevant += 1
        else:
            rejected += 1
        review_results.append({"page": page_num, "relevant": is_rel})

    write_json(ocr_review_path(job_id, filename), {"pages": review_results, "domain": domain})
    logger.info("ocr_review job=%s file=%s relevant=%d rejected=%d", job_id, filename, relevant, rejected)
    return {"total": len(pages), "relevant": relevant, "rejected": rejected}


def tool_merge_corpus(job_id: str) -> dict:
    """Merge all cleaned/extracted texts from every uploaded file into a single unified corpus.

    Priority per file:
      1. cleaned text  (artifacts/{job_id}/cleaned/{filename}.txt)
      2. raw extraction (artifacts/{job_id}/extractions/{filename}.md)
      3. full_text field inside the profile JSON

    The result is written to artifacts/{job_id}/merged_corpus.txt and is the
    single source of truth for all subsequent QA generation.
    """
    profiles = read_all_profiles(job_id)
    if not profiles:
        return {"error": "No profiles found — run profile_file_tool first"}

    separator = "\n\n" + "─" * 60 + "\n\n"
    parts: list[str] = []
    sources: list[dict] = []

    for profile in profiles:
        filename = profile.get("filename", "")
        file_kind = profile.get("file_kind", "txt")
        text = ""

        # 1. Cleaned text (PDF / TXT après nettoyage)
        cp = cleaned_path(job_id, filename)
        if cp.exists():
            text = cp.read_text(encoding="utf-8").strip()

        # 2. Raw extraction (markdown depuis Marker)
        if not text:
            rp = raw_extraction_path(job_id, filename)
            if rp.exists():
                text = rp.read_text(encoding="utf-8").strip()

        # 3. full_text dans le profil JSON
        if not text:
            text = (profile.get("full_text") or "").strip()

        # 4. CSV : construire un texte lisible depuis les lignes d'exemple
        if not text and file_kind == "csv":
            rows = (profile.get("metadata") or {}).get("sample_rows", [])
            if rows:
                text = f"Données tabulaires extraites de {filename}:\n" + "\n".join(
                    str(r) for r in rows[:200]
                )

        if not text:
            logger.warning("merge_corpus: no text found for file=%s — skipped", filename)
            continue

        parts.append(f"=== SOURCE : {filename} ===\n\n{text}")
        sources.append({"filename": filename, "chars": len(text)})
        logger.info("merge_corpus: added file=%s chars=%d", filename, len(text))

    if not parts:
        return {"error": "Aucun texte extrait — vérifiez que les fichiers sont valides"}

    merged_text = separator.join(parts) + "\n"
    out = merged_corpus_path(job_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(merged_text, encoding="utf-8")

    total_chars = len(merged_text)
    logger.info("merge_corpus job=%s files=%d total_chars=%d", job_id, len(sources), total_chars)
    return {
        "merged_chars": total_chars,
        "file_count": len(sources),
        "sources": sources,
        "corpus_path": str(out),
    }


def tool_assess_readiness(job_id: str, task: str, user_prompt: str = "") -> dict:
    """Run deterministic feasibility rules across all profiled files."""
    vagueness = check_prompt_vagueness(user_prompt or task)
    if vagueness.status.value == "BLOCKED":
        result = {"verdict": "BLOCKED", "issues": vagueness.blocking_reasons, "file_count": 0}
        write_json(readiness_path(job_id), result)
        return result

    profiles = read_all_profiles(job_id)
    if not profiles:
        result = {"verdict": "BLOCKED", "issues": ["No profiles found — run profile_file_tool first"], "file_count": 0}
        write_json(readiness_path(job_id), result)
        return result

    feasibility = evaluate_feasibility(profiles)
    result = {
        "verdict": feasibility.status.value,
        "issues": feasibility.blocking_reasons,
        "warnings": feasibility.warnings,
        "file_count": len(profiles),
    }
    write_json(readiness_path(job_id), result)
    logger.info("assess_readiness job=%s verdict=%s", job_id, feasibility.status.value)
    return result
