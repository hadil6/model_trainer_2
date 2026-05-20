from __future__ import annotations

import concurrent.futures
import logging
import tempfile
from pathlib import Path
from typing import Any

from data.profilers.base import BaseProfiler
from utils.text_utils import (
    compute_text_quality_signals,
    compute_training_readiness,
    detect_structure_hints,
    normalize_text,
)

logger = logging.getLogger(__name__)

_marker_models = None


def release_marker_models() -> None:
    """Delete cached marker-pdf models and free their VRAM.

    Call this after all PDFs have been profiled so the GPU is free for training.
    """
    global _marker_models
    import gc
    if _marker_models is None:
        return
    try:
        import torch
        for obj in _marker_models.values():
            try:
                if hasattr(obj, "to"):
                    obj.to("cpu")
            except Exception:
                pass
        del _marker_models
        _marker_models = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("marker-pdf models released from GPU.")
    except Exception as exc:
        logger.warning("release_marker_models failed: %s", exc)
        _marker_models = None


def _get_marker_models() -> dict:
    global _marker_models
    if _marker_models is None:
        from marker.models import create_model_dict
        logger.info("Loading marker-pdf models (first load downloads weights)...")
        _marker_models = create_model_dict()
        logger.info("marker-pdf models ready")
    return _marker_models


class PdfProfiler(BaseProfiler):

    def _extract_with_marker(self, content: bytes) -> tuple[str, dict, list[str]]:
        warnings: list[str] = []
        tmp_path = None
        try:
            from marker.converters.pdf import PdfConverter
            from marker.output import text_from_rendered

            artifact_dict = _get_marker_models()
            converter = PdfConverter(artifact_dict=artifact_dict)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            rendered = converter(tmp_path)
            text, _, _ = text_from_rendered(rendered)

            raw_metadata = getattr(rendered, "metadata", {}) or {}
            page_count = raw_metadata.get("page_count", 0)

            return text.strip(), {"page_count": page_count}, warnings

        except Exception as exc:
            logger.error("marker_extraction_failed: %s", exc)
            warnings.append(f"marker_extraction_failed:{exc}")
            return "", {}, warnings
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _extract_with_timeout(self, content: bytes, timeout: int = 7200) -> tuple:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._extract_with_marker, content)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise RuntimeError(f"PDF extraction timed out after {timeout}s")
        finally:
            executor.shutdown(wait=False)

    @staticmethod
    def _adaptive_timeout(size_bytes: int) -> int:
        """Estimate timeout from file size: ~5 000 bytes/s processing rate, capped by config."""
        from core.config import get_settings
        cap = get_settings().pdf_extraction_timeout_seconds
        estimated = max(1800, size_bytes // 5_000)
        return min(cap, estimated)

    @staticmethod
    def _extract_with_fitz(content: bytes, filename: str) -> tuple[str, int]:
        """Fast fallback extraction using PyMuPDF (no ML models needed)."""
        import fitz  # type: ignore[import]
        parts: list[str] = []
        with fitz.open(stream=content, filetype="pdf") as doc:
            page_count = len(doc)
            for page in doc:
                parts.append(page.get_text())
        return "\n".join(parts).strip(), page_count

    def profile(self, filename: str, content: bytes) -> dict[str, Any]:
        timeout = self._adaptive_timeout(len(content))
        logger.info("pdf_extraction_timeout=%ds for %s (%.1f MB)", timeout, filename, len(content) / 1e6)
        fitz_fallback = False
        pdf_meta: dict = {}
        warnings: list[str] = []
        try:
            raw_text, pdf_meta, warnings = self._extract_with_timeout(content, timeout=timeout)
        except RuntimeError as exc:
            logger.warning("marker_timeout file=%s — trying PyMuPDF fallback", filename)
            warnings = [f"marker_timeout:{exc}"]
            try:
                raw_text, page_count = self._extract_with_fitz(content, filename)
                pdf_meta = {"page_count": page_count}
                fitz_fallback = True
                logger.info("fitz_fallback_ok file=%s chars=%d", filename, len(raw_text))
            except Exception as fitz_exc:
                logger.error("fitz_fallback_failed file=%s: %s", filename, fitz_exc)
                return {
                    "file_kind": "pdf",
                    "metadata": {"filename": filename, "size_bytes": len(content)},
                    "quality_checks": {},
                    "samples": {},
                    "full_text": "",
                    "ocr_pages": {},
                    "ocr_summary": {"n_pages": 0, "total_chars": 0},
                    "ocr_mode": "timeout",
                    "ocr_relevant_text": "",
                    "warnings": [f"marker_timeout:{exc}", f"fitz_fallback_failed:{fitz_exc}"],
                    "risk_flags": ["profile_extraction_timed_out"],
                }

        normalized = normalize_text(raw_text)
        words = normalized.split()
        lines = [ln for ln in raw_text.split("\n") if ln.strip()]

        quality = compute_text_quality_signals(raw_text)
        training = compute_training_readiness(raw_text)
        structure = detect_structure_hints(raw_text)

        mid = len(raw_text) // 2
        snippets = {
            "head": raw_text[:400],
            "middle": raw_text[max(0, mid - 200): mid + 200],
            "tail": raw_text[-400:],
        }

        warnings.extend([f"text_noise:{f}" for f in quality.get("suspected_noise", [])])
        warnings.extend([f"training:{w}" for w in training.get("suitability_warnings", [])])

        risk_flags: list[str] = []
        if quality["garbled_ratio"] > 0.2:
            risk_flags.append("high_garbled_text")
        if quality["boilerplate_ratio"] > 0.3:
            risk_flags.append("high_boilerplate_text")
        if training["training_suitability"] < 0.4:
            risk_flags.append("low_training_suitability")
        if not raw_text:
            risk_flags.append("empty_extraction")
        if fitz_fallback:
            risk_flags.append("marker_timeout_fitz_fallback")

        return {
            "file_kind": "pdf",
            "metadata": {
                "filename": filename,
                "size_bytes": len(content),
                "page_count": pdf_meta.get("page_count", 0),
                "character_count": len(raw_text),
                "word_count": len(words),
                "line_count": len(lines),
                "detected_language": training["language"],
                "content_structure": "fitz_fallback" if fitz_fallback else "marker_extracted",
            },
            "quality_checks": {
                **quality,
                "training_readiness": training,
                "structure_hints": structure,
            },
            "samples": snippets,
            "full_text": raw_text,
            "ocr_pages": {},
            "ocr_summary": {"n_pages": 0, "total_chars": 0},
            "ocr_mode": "fitz_fallback" if fitz_fallback else "marker_integrated",
            "ocr_relevant_text": "",
            "warnings": sorted(set(warnings)),
            "risk_flags": sorted(set(risk_flags)),
        }
