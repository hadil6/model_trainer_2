import re
import unicodedata
from typing import Any

import chardet


WHITESPACE_PATTERN = re.compile(r"\s+")

_LANG_MARKERS: dict[str, set[str]] = {
    "fr": {"le", "la", "les", "de", "du", "des", "un", "une", "est", "sont",
           "que", "qui", "dans", "avec", "pour", "sur", "par", "mais", "aussi",
           "cette", "tout", "plus", "pas", "au", "aux", "leur", "leurs"},
    "en": {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
           "at", "to", "for", "with", "this", "that", "it", "its", "be",
           "have", "has", "had", "not", "by", "from", "they", "we", "you"},
    "ar": {"في", "من", "على", "إلى", "هذا", "هذه", "التي", "الذي", "كان",
           "مع", "عن", "أن", "لا", "ما", "كل", "قد", "إن", "عند", "بين"},
}


def detect_language(text: str) -> str:
    sample = text[:3000].lower()
    words = set(re.findall(r'\b\w+\b', sample))
    if not words:
        return "unknown"
    scores = {lang: len(words & markers) for lang, markers in _LANG_MARKERS.items()}
    best_lang, best_score = max(scores.items(), key=lambda x: x[1])
    return best_lang if best_score >= 3 else "unknown"


def decode_text_bytes(raw: bytes) -> tuple[str, str]:
    detected = chardet.detect(raw) or {}
    encoding = (detected.get("encoding") or "utf-8").lower()
    try:
        decoded = raw.decode(encoding)
    except (LookupError, UnicodeDecodeError):
        encoding = "utf-8"
        decoded = raw.decode("utf-8", errors="replace")
    return decoded, encoding


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def compute_text_quality_signals(text: str) -> dict[str, Any]:
    if not text:
        return {
            "garbled_ratio": 1.0,
            "boilerplate_ratio": 0.0,
            "suspected_noise": ["empty_text"],
        }

    total_chars = len(text)
    replacement_chars = text.count("�")
    non_printable = sum(1 for ch in text if not ch.isprintable() and ch not in "\n\t")
    garbled_ratio = (replacement_chars + non_printable) / max(total_chars, 1)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    freq: dict[str, int] = {}
    for line in lines:
        freq[line] = freq.get(line, 0) + 1

    repeated_lines = sum(count for count in freq.values() if count > 1)
    boilerplate_ratio = repeated_lines / max(len(lines), 1)

    suspected_noise: list[str] = []
    if garbled_ratio > 0.1:
        suspected_noise.append("high_garbled_ratio")
    if boilerplate_ratio > 0.3:
        suspected_noise.append("high_boilerplate")

    return {
        "garbled_ratio": round(garbled_ratio, 4),
        "boilerplate_ratio": round(boilerplate_ratio, 4),
        "suspected_noise": suspected_noise,
    }


def compute_training_readiness(text: str) -> dict[str, Any]:
    if not text or len(text.strip()) < 50:
        return {
            "language": "unknown",
            "sentence_count": 0,
            "avg_sentence_length": 0,
            "lexical_diversity": 0.0,
            "short_sentence_ratio": 0.0,
            "long_sentence_ratio": 0.0,
            "training_suitability": 0.0,
            "suitability_warnings": ["text_too_short"],
        }

    language = detect_language(text)

    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
    sentence_count = len(sentences)

    words_per_sentence = [len(s.split()) for s in sentences]
    avg_sentence_length = sum(words_per_sentence) / max(sentence_count, 1)

    short_sentences = sum(1 for w in words_per_sentence if w < 5)
    long_sentences = sum(1 for w in words_per_sentence if w > 60)
    short_sentence_ratio = short_sentences / max(sentence_count, 1)
    long_sentence_ratio = long_sentences / max(sentence_count, 1)

    all_words = text.lower().split()
    sample_words = all_words[:500]
    lexical_diversity = len(set(sample_words)) / max(len(sample_words), 1)

    suitability_warnings: list[str] = []
    score = 1.0

    if lexical_diversity < 0.3:
        score -= 0.3
        suitability_warnings.append(f"low_lexical_diversity:{lexical_diversity:.2f}")
    if short_sentence_ratio > 0.5:
        score -= 0.2
        suitability_warnings.append(f"many_short_fragments:{short_sentence_ratio:.0%}")
    if avg_sentence_length < 5:
        score -= 0.2
        suitability_warnings.append(f"very_short_avg_sentence:{avg_sentence_length:.1f}_words")
    if sentence_count < 5:
        score -= 0.2
        suitability_warnings.append(f"very_few_sentences:{sentence_count}")
    if language == "unknown":
        suitability_warnings.append("language_not_detected")

    return {
        "language": language,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "lexical_diversity": round(lexical_diversity, 4),
        "short_sentence_ratio": round(short_sentence_ratio, 4),
        "long_sentence_ratio": round(long_sentence_ratio, 4),
        "training_suitability": round(max(0.0, score), 4),
        "suitability_warnings": suitability_warnings,
    }


def detect_structure_hints(text: str) -> list[str]:
    hints: list[str] = []
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    heading_like = 0
    for line in lines[:300]:
        if len(line) < 120 and (
            line.isupper() or re.match(r"^\d+(\.\d+)*\s+", line) or line.endswith(":")):
            heading_like += 1
    if heading_like >= 3:
        hints.append("contains_heading_like_structure")
    if any("table of contents" in line.lower() for line in lines[:60]):
        hints.append("possible_table_of_contents")
    if not hints:
        hints.append("unstructured_or_light_structure")
    return hints
