from __future__ import annotations

from typing import Any

from data.profilers.base import BaseProfiler
from utils.text_utils import (
    compute_text_quality_signals,
    compute_training_readiness,
    decode_text_bytes,
    detect_structure_hints,
    normalize_text,
)


class TxtProfiler(BaseProfiler):
    def profile(self, filename: str, content: bytes) -> dict[str, Any]:
        decoded, encoding = decode_text_bytes(content)
        text = normalize_text(decoded)

        words = text.split()
        lines = [line for line in decoded.split("\n") if line.strip()]

        quality = compute_text_quality_signals(decoded)
        training = compute_training_readiness(decoded)
        structure = detect_structure_hints(decoded)

        snippets = {
            "head": text[:400],
            "middle": text[max(0, len(text) // 2 - 200): max(0, len(text) // 2 + 200)],
            "tail": text[-400:],
        }

        warnings = [f"text_noise:{flag}" for flag in quality.get("suspected_noise", [])]
        warnings += [f"training:{w}" for w in training.get("suitability_warnings", [])]
        risk_flags: list[str] = []
        if quality["garbled_ratio"] > 0.2:
            risk_flags.append("high_garbled_text")
        if quality["boilerplate_ratio"] > 0.3:
            risk_flags.append("high_boilerplate_text")
        if training["training_suitability"] < 0.4:
            risk_flags.append("low_training_suitability")

        return {
            "file_kind": "txt",
            "metadata": {
                "filename": filename,
                "size_bytes": len(content),
                "encoding": encoding,
                "character_count": len(text),
                "word_count": len(words),
                "line_count": len(lines),
                "detected_language": training["language"],
            },
            "quality_checks": {
                **quality,
                "training_readiness": training,
                "structure_hints": structure,
            },
            "samples": snippets,
            "full_text": decoded,
            "warnings": sorted(set(warnings)),
            "risk_flags": sorted(set(risk_flags)),
        }
