"""Generation tool implementations (plain functions — registered on MCP app in server.py)."""

from __future__ import annotations

import json
import logging

from agents.llm_client import get_llm_client
from agents.data_agent.services.qa import QAService, QAPair
from agents.data_agent.services.readiness import ReadinessService
from core.config import get_settings
from data.artifact_store import (
    append_jsonl,
    chunks_path,
    cleaned_path,
    coherence_path,
    domain_path,
    dropped_qa_path,
    eval_qa_path,
    merged_corpus_path,
    profile_path,
    qa_final_path,
    qa_path,
    read_all_profiles,
    read_all_qa,
    read_jsonl,
    readiness_report_path,
    write_json,
)
from data.model_registry import GENERATION_STYLES, get_requirements

logger = logging.getLogger(__name__)


def _qa_service() -> QAService:
    return QAService(get_llm_client(), embedding_model=get_settings().embedding_model)


def tool_detect_domain(job_id: str) -> dict:
    """Auto-detect domain and language from document summaries after profiling."""
    profiles = read_all_profiles(job_id)
    summaries = [p.get("semantic_summary", "") for p in profiles if p.get("semantic_summary")]

    if not summaries:
        fallback = {"domain": "general", "language": "en", "detected": False}
        write_json(domain_path(job_id), fallback)
        return fallback

    content = "\n\n---\n\n".join(summaries[:3])[:3000]
    try:
        import re
        raw = get_llm_client().complete(
            system="You detect the knowledge domain and language of a document collection for ML fine-tuning.",
            user=(
                f"Document summaries:\n{content}\n\n"
                "Identify the specific domain (e.g. 'DevOps', 'cardiology', 'contract law') "
                "and the primary language.\n"
                'Return ONLY JSON: {"domain": "...", "language": "fr|en|ar|other", "reason": "..."}'
            ),
        )
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            result = json.loads(m.group(0))
            result["detected"] = True
            write_json(domain_path(job_id), result)
            logger.info("detect_domain job=%s domain=%s lang=%s", job_id, result.get("domain"), result.get("language"))
            return result
    except Exception as exc:
        logger.warning("detect_domain failed: %s", exc)

    fallback = {"domain": "general", "language": "en", "detected": False}
    write_json(domain_path(job_id), fallback)
    return fallback


def tool_check_coherence(job_id: str, task: str) -> dict:
    """Check whether the profiled files form a coherent dataset for the given task."""
    profiles = read_all_profiles(job_id)
    if not profiles:
        return {"coherent": False, "score": 0.0, "dominant_domain": None, "conflicting_files": []}

    file_entries = []
    for p in profiles:
        semantic = p.get("semantic_summary", "")
        # Fall back to a text excerpt when the semantic summary is absent or very short
        if len(semantic.split()) < 20:
            full = p.get("full_text", "")
            semantic = full[:600] if full else semantic
        file_entries.append({
            "filename": p.get("filename"),
            "file_kind": p.get("file_kind"),
            "content_preview": semantic[:400],
        })
    summary = json.dumps(file_entries, ensure_ascii=False)[:3500]

    try:
        import re
        raw = get_llm_client().complete(
            system="You are a dataset coherence evaluator. Judge based on document CONTENT, not filenames.",
            user=(
                f"Task: {task}\n\nFile content previews:\n{summary}\n\n"
                "Evaluate whether these files share a common knowledge domain and would form a "
                "coherent training dataset. Base your judgement on the CONTENT shown, not filenames.\n"
                "Give a high score (≥0.7) when documents clearly share a topic, even if written "
                "in different styles (textbook vs research paper vs tutorial).\n"
                'Return ONLY JSON: {"coherent": bool, "score": 0.0-1.0, '
                '"dominant_domain": "...", "conflicting_files": [], "reason": "..."}'
            ),
        )
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            result = json.loads(m.group(0))
            write_json(coherence_path(job_id), result)
            logger.info("coherence job=%s score=%s", job_id, result.get("score"))
            return {
                "coherent": result.get("coherent", True),
                "score": result.get("score", 0.8),
                "dominant_domain": result.get("dominant_domain"),
                "conflicting_files": result.get("conflicting_files", []),
            }
    except Exception as exc:
        logger.error("check_coherence failed: %s", exc)

    fallback = {"coherent": True, "score": 0.8, "dominant_domain": None, "conflicting_files": []}
    write_json(coherence_path(job_id), fallback)
    return fallback


def tool_generate_qa(
    job_id: str, filename: str, task: str, pass_num: int = 1, model_id: str = ""
) -> dict:
    """Generate Q&A pairs from a file or from the unified merged corpus.

    Pass filename="merged_corpus" (or "merged_corpus.txt") to generate from the
    unified corpus produced by merge_corpus().  This is the recommended approach
    when multiple files were uploaded — all text is already merged into one file.

    For a single-file job, filename is the original filename as usual.
    """
    _CORPUS_NAMES = {"merged_corpus", "merged_corpus.txt"}

    if filename in _CORPUS_NAMES:
        # ── Unified corpus path ───────────────────────────────────────────────
        corpus_file = merged_corpus_path(job_id)
        if not corpus_file.exists():
            raise FileNotFoundError(
                "merged_corpus.txt introuvable — appelez merge_corpus() en premier"
            )
        corpus_text = corpus_file.read_text(encoding="utf-8")
        profile = {
            "filename": "merged_corpus",
            "file_kind": "txt",
            "full_text": corpus_text,
            "metadata": {
                "char_count": len(corpus_text),
                "word_count": len(corpus_text.split()),
            },
        }
        cleaned_text = corpus_text
        out_key = "merged_corpus"
    else:
        # ── Per-file path (fallback / single file) ────────────────────────────
        p = profile_path(job_id, filename)
        if not p.exists():
            raise FileNotFoundError(f"Profil introuvable pour {filename}")
        profile = json.loads(p.read_text(encoding="utf-8"))
        cleaned_text = None
        cp = cleaned_path(job_id, filename)
        if cp.exists():
            cleaned_text = cp.read_text(encoding="utf-8")
        out_key = filename

    pairs = _qa_service().generate_for_file(
        filename=out_key,
        profile=profile,
        cleaned_text=cleaned_text,
        task=task,
        model_id=model_id,
        debug_chunks_path=chunks_path(job_id, out_key),
    )

    records = [
        {"question": p.question, "answer": p.answer,
         "source_file": p.source_file, "confidence": p.confidence}
        for p in pairs
    ]
    out = qa_path(job_id, out_key)
    append_jsonl(out, records)
    total = len(read_jsonl(out))
    logger.info(
        "generate_qa job=%s source=%s pass=%d new=%d total=%d",
        job_id, out_key, pass_num, len(pairs), total,
    )
    return {"pairs_generated": len(pairs), "total_pairs": total}


def tool_evaluate_and_refine_qa(job_id: str, task: str, sample_size: int = 20) -> dict:
    """Sample pairs, LLM-score KEEP/REWRITE/DROP, write qa_final.jsonl."""
    all_raw = read_all_qa(job_id)
    if not all_raw:
        return {"sampled": 0, "rewritten": 0, "dropped": 0, "total_after": 0}

    pairs = [QAPair(question=r["question"], answer=r["answer"],
                    source_file=r.get("source_file", ""), confidence=r.get("confidence", 0.5))
             for r in all_raw]

    kept, dropped = _qa_service().evaluate_and_refine(pairs, sample_size=sample_size)
    rewritten = sum(1 for p in kept if "[refined]" in p.source_file)

    out = qa_final_path(job_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps({"question": p.question, "answer": p.answer,
                               "source_file": p.source_file, "confidence": p.confidence},
                              ensure_ascii=False) for p in kept) + "\n",
        encoding="utf-8",
    )
    write_json(dropped_qa_path(job_id),
               [{"question": p.question, "answer": p.answer} for p in dropped])

    logger.info("eval_refine job=%s dropped=%d kept=%d", job_id, len(dropped), len(kept))
    return {"sampled": min(sample_size, len(pairs)), "rewritten": rewritten,
            "dropped": len(dropped), "total_after": len(kept)}


def tool_deduplicate_qa(job_id: str) -> dict:
    """Lexical + semantic dedup of qa_final.jsonl.

    Near-duplicate questions (same idea, different phrasing) are NOT dropped —
    they are rephrased by the LLM and saved to eval_qa.jsonl for use as the
    test/eval set (Clone + Rephrase approach).
    """
    out = qa_final_path(job_id)
    raw = read_jsonl(out)
    if not raw:
        return {"before": 0, "after": 0, "removed_lexical": 0, "removed_semantic": 0, "eval_pairs": 0}

    before = len(raw)
    pairs = [QAPair(question=r["question"], answer=r["answer"],
                    source_file=r.get("source_file", ""), confidence=r.get("confidence", 0.5))
             for r in raw]

    svc = _qa_service()

    # Lexical dedup — true exact/near-exact duplicates
    dropped_lex: list[QAPair] = []
    unique = svc._lexical_dedup(pairs, dropped_lex)

    # Semantic dedup — question-only comparison
    # near_dups: same idea, different phrasing → rephrase for eval
    # dropped_sem: identical answers → true copies, discard
    dropped_sem: list[QAPair] = []
    near_dups: list[QAPair] = []
    unique = svc._semantic_dedup(unique, dropped_sem, near_dups=near_dups)

    # Write unique pairs back to qa_final
    out.write_text(
        "\n".join(json.dumps({"question": p.question, "answer": p.answer,
                               "source_file": p.source_file, "confidence": p.confidence},
                              ensure_ascii=False) for p in unique) + "\n",
        encoding="utf-8",
    )

    # Rephrase near-dups → eval set
    eval_pairs = svc.rephrase_for_eval(near_dups)
    if eval_pairs:
        ep = eval_qa_path(job_id)
        ep.parent.mkdir(parents=True, exist_ok=True)
        ep.write_text(
            "\n".join(json.dumps({"question": p.question, "answer": p.answer,
                                   "source_file": p.source_file, "confidence": p.confidence},
                                  ensure_ascii=False) for p in eval_pairs) + "\n",
            encoding="utf-8",
        )

    logger.info(
        "dedup job=%s before=%d after=%d eval_pairs=%d", job_id, before, len(unique), len(eval_pairs)
    )
    return {
        "before": before,
        "after": len(unique),
        "removed_lexical": len(dropped_lex),
        "removed_semantic": len(dropped_sem),
        "eval_pairs": len(eval_pairs),
    }


def tool_evaluate_readiness(
    job_id: str, task: str, target_peft: str = "qlora", target_model: str = ""
) -> dict:
    """Run 6-axis readiness judge on the final Q&A dataset."""
    raw = read_jsonl(qa_final_path(job_id)) or read_all_qa(job_id)
    pairs = [QAPair(question=r["question"], answer=r["answer"],
                    source_file=r.get("source_file", ""), confidence=r.get("confidence", 0.5))
             for r in raw]

    summaries = [p.get("semantic_summary", "") for p in read_all_profiles(job_id)
                 if p.get("semantic_summary")]

    report = ReadinessService(get_llm_client()).evaluate(
        pairs, task=task, peft=target_peft, model_name=target_model, file_summaries=summaries
    )
    write_json(readiness_report_path(job_id), report)
    logger.info("readiness job=%s verdict=%s n=%d", job_id, report.get("verdict"), len(pairs))
    return report


def tool_auto_fill_qa_preview(
    job_id: str, target_model: str, task: str, target_peft: str = "qlora",
    model_id: str = "", max_samples: int = 5,
) -> dict:
    """Generate a small in-memory preview of pairs that auto_fill_qa would produce.

    Nothing is written to disk. Used for the user-confirmation dialog so the user
    sees the EXACT style/content of what will be appended to the dataset before
    they approve.
    """
    profiles = read_all_profiles(job_id)
    if not profiles:
        return {"samples": [], "message": "No profiles available for preview."}

    svc = _qa_service()
    samples: list[dict] = []

    for profile in profiles:
        if len(samples) >= max_samples:
            break
        filename = profile.get("filename", "")
        if not filename:
            continue
        cp = cleaned_path(job_id, filename)
        cleaned_text = cp.read_text(encoding="utf-8") if cp.exists() else None

        try:
            pairs = svc.generate_for_file(
                filename=filename,
                profile=profile,
                cleaned_text=cleaned_text,
                task=task,
                style="conceptual",
                model_id=model_id or target_model,
            )
        except Exception as exc:
            logger.warning("auto_fill_preview_failed file=%s: %s", filename, exc)
            continue

        for p in pairs or []:
            if len(samples) >= max_samples:
                break
            samples.append({
                "question":    p.question,
                "answer":      str(p.answer)[:500],
                "source_file": p.source_file,
            })

    logger.info(
        "auto_fill_preview job=%s generated_samples=%d", job_id, len(samples)
    )
    return {"samples": samples, "count": len(samples)}


def tool_auto_fill_qa(
    job_id: str, target_model: str, task: str, target_peft: str = "qlora",
    model_id: str = "",
) -> dict:
    """Generate extra pairs until the model-specific target count is reached.
    For summarization: generates (document→summary) pairs.
    For other tasks: generates Q&A pairs with varied styles (conceptual/applied/analytical).
    Appends directly to qa_final.jsonl."""
    req = get_requirements(target_model, target_peft, task)
    target_count = req["target"]

    current_raw = read_jsonl(qa_final_path(job_id))
    if not current_raw:
        current_raw = read_all_qa(job_id)
    current_count = len(current_raw)

    if current_count >= target_count:
        logger.info(
            "auto_fill job=%s already at target (%d >= %d)", job_id, current_count, target_count
        )
        return {
            "needed": 0, "generated": 0, "total": current_count,
            "target": target_count, "model": target_model,
            "peft": target_peft, "reached_target": True,
            "message": "Target already reached — no generation needed.",
        }

    needed = target_count - current_count
    profiles = read_all_profiles(job_id)
    if not profiles:
        return {
            "needed": needed, "generated": 0, "total": current_count,
            "target": target_count, "model": target_model, "peft": target_peft,
            "reached_target": False, "message": "No profiles found.",
        }

    svc = _qa_service()
    total_generated = 0

    for style in ("conceptual", "applied", "analytical"):
        if current_count + total_generated >= target_count:
            break
        for profile in profiles:
            if current_count + total_generated >= target_count:
                break
            filename = profile.get("filename", "")
            if not filename:
                continue
            cp = cleaned_path(job_id, filename)
            cleaned_text = cp.read_text(encoding="utf-8") if cp.exists() else None

            try:
                pairs = svc.generate_for_file(
                    filename=filename,
                    profile=profile,
                    cleaned_text=cleaned_text,
                    task=task,
                    style=style,
                    model_id=model_id or target_model,
                )
            except Exception as exc:
                logger.warning("auto_fill_failed style=%s file=%s: %s", style, filename, exc)
                continue

            if pairs:
                records = [
                    {"question": p.question, "answer": p.answer,
                     "source_file": p.source_file, "confidence": p.confidence}
                    for p in pairs
                ]
                append_jsonl(qa_final_path(job_id), records)
                total_generated += len(pairs)
                logger.info(
                    "auto_fill job=%s style=%s file=%s new=%d running_total=%d",
                    job_id, style, filename, len(pairs), current_count + total_generated,
                )

    final_count = current_count + total_generated
    logger.info(
        "auto_fill done job=%s generated=%d final=%d target=%d",
        job_id, total_generated, final_count, target_count,
    )
    return {
        "needed": needed,
        "generated": total_generated,
        "total": final_count,
        "target": target_count,
        "model": target_model,
        "peft": target_peft,
        "reached_target": final_count >= target_count,
    }
