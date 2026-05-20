"""QA generation, chunking, filtering, deduplication, and dataset export.

Public API
----------
QAPair               — dataclass for a single Q&A pair
QAService            — main class; call generate_for_file(), deduplicate(), evaluate_and_refine()
export_dataset()     — format final pairs into fine-tuning JSONL
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any

from agents.llm_client import LLMClient
from data.model_registry import GENERATION_STYLES

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

_SCORE_DISCARD    = 0.35
_SCORE_REFINE     = 0.62
_EMBED_SIM_THRESH = 0.90
_ANSWER_SIM_THRESH = 0.92
_MIN_RICHNESS     = 0.05
_LLM_EVAL_BATCH   = 8


_VAGUE_ANSWERS = {
    "oui", "non", "peut-être", "je ne sais pas", "ça dépend",
    "yes", "no", "maybe", "i don't know", "it depends", "not mentioned",
    "the text does not specify", "le texte ne précise pas",
}

_DOC_REF_PATTERNS = re.compile(
    r"\b(in the (rapport|document|text|chunk|passage|source|context)|"
    r"according to (the )?(rapport|document|text|source)|"
    r"the (rapport|document|text) (says|states|mentions|explains|describes)|"
    r"dans (le |la )?(rapport|document|texte|source)|"
    r"selon (le |la )?(rapport|document|texte))\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "est", "sont",
    "que", "qui", "ce", "il", "elle", "ils", "elles",
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for",
    "with", "this", "that", "it", "its",
}

_METADATA_PATTERNS = [
    r"\btitle of (the |this )?(document|article|paper|pdf|text|book|whitepaper|report)\b",
    r"\btitre (du|de la|de ce|de cet) (document|article|rapport|texte|livre)\b",
    r"\bwho (is|are) (identified as|listed as|credited|the author|the authors)\b",
    r"\bwho wrote\b", r"\bwho authored\b",
    r"\bqui (a écrit|a rédigé|est l'auteur|sont les auteurs)\b",
    r"\b(what|which) (page|section|chapter) (number )?(is|comes|appears|shows)",
    r"\bpage (number|numéro)\b", r"\bnuméro de page\b",
    r"\btable of contents\b", r"\btable des matières\b",
    r"\b(copyright|©|all rights reserved)\b",
    r"\bwhat (year|date) (of )?(copyright|publication|publish)",
    r"\bwhat (organization|company|publisher|firm) (is associated|published|produced)",
    r"\bwhat (job|position|role|title) (does|do) \w+ (hold|have|occupy)",
    r"\bquel (poste|rôle|titre) (occupe|a)\b",
    r"\b(affiliation|university|employer) of\b",
    r"\bphrase\b.*\b(appears|shows|is mentioned)\b",
    r"\baccording to the (text|document|author|article|passage|paper)\b",
    r"\b(selon|d'après) (le texte|le document|l'auteur|l'article)\b",
    r"\bwhat does the (text|document|article|passage) (say|state|describe|explain|mention)\b",
    r"\bque (dit|précise|mentionne|explique) le (texte|document|paragraphe|passage)\b",
    r"\bwhich section (comes|appears|is|follows)\b",
    r"\bimmediately (after|before) [\"']", r"\blast visible topic\b",
    r"\btopic (comes|visible) (immediately )?(after|before|in the)",
    r"\bfinal topic\b", r"\bheader\b",
    r"\bwebsite (mentioned|listed|referenced|shown|is listed|for learning)\b",
    r"\bcertification (path|program|levels?) (validate|offer|include|mentioned|are listed)\b",
    r"\bwhat (academic|professional) affiliation is listed\b",
    r"\bwhat job (title|role) (and company )?(are|is) associated\b",
    r"\bwhich certification levels? (are )?(mentioned|listed)\b",
    r"\bwhat year (is )?(referenced|associated|mentioned) in (the|this) (document|text)\b",
    r"\bwhat heading introduces\b",
    r"\bwho is (the )?intended audience\b",
    r"\bwhat phrase is used to describe\b",
    r"\baccording to the (abstract|timeline|figure)\b",
    r"\bwhat does the (abstract|timeline|figure) (show|state|indicate|describe)\b",
]
_METADATA_RE: re.Pattern | None = None

_BOILERPLATE_SIGNS = [
    "all rights reserved", "table of contents", "table des matières",
    "©", "disclaimer:", "trademarks are trademarks",
    "this publication is provided", "knowledge sharing article",
]


_TASK_FORMATS: dict[str, str] = {
    "question-answering":    "alpaca",
    "conversational-qa":     "sharegpt",
    "classification":        "alpaca_label",
    "text-classification":   "alpaca_label",
    "sentiment-analysis":    "alpaca_label",
    "instruction-following": "alpaca",
    "translation":           "alpaca",
    "code-generation":       "alpaca",
    # summarization handled separately — format depends on model architecture
}

# ── Summarization format detection ────────────────────────────────────────────
# Encoder-decoder (seq2seq) models need {"document": "...", "summary": "..."}
_SEQ2SEQ_KEYWORDS = {"t5", "bart", "pegasus", "flan", "mt5", "mbart", "opus", "led"}
# Decoder-only chat/instruction models need messages format
_CHAT_KEYWORDS    = {"mistral", "llama", "qwen", "phi", "gemma", "chatglm", "deepseek",
                     "solar", "falcon", "vicuna", "intern", "yi", "baichuan"}

# Chunk size for summarization: 2000 chars ≈ 330 words.
# Rationale: CNN/DailyMail benchmark (BART/PEGASUS) uses ~800-word articles → ~55-word
# summaries (14:1 ratio). For domain fine-tuning, 330 words → ~60-word summary (5:1)
# is the minimum meaningful compression. 500-char chunks (~80 words) produce paraphrases,
# not summaries.
_SUM_CHUNK_SIZE = 2000

_SUM_INSTRUCTIONS = [
    "Résume ce texte en quelques phrases claires et concises.",
    "Donne un résumé concis de ce passage en 3 à 5 phrases.",
    "En quelques phrases, résume les idées principales de ce texte.",
    "Fournis un résumé synthétique de ce passage.",
]


def _detect_summarization_format(model_id: str) -> str:
    """Return 'seq2seq', 'chat', or 'alpaca' based on model architecture."""
    mid = model_id.lower()
    if any(k in mid for k in _SEQ2SEQ_KEYWORDS):
        return "seq2seq"
    if any(k in mid for k in _CHAT_KEYWORDS):
        return "chat"
    return "alpaca"

_GROUNDING_IGNORE = {
    "approach", "approaches", "practice", "practices", "process", "processes",
    "system", "systems", "method", "methods", "concept", "concepts",
    "example", "examples", "team", "teams", "user", "users", "solution",
    "solutions", "result", "results", "value", "values", "benefit", "benefits",
    "feature", "features", "overall", "various", "different", "several",
    "including", "typically", "generally", "usually", "often",
    "cela", "cette", "ceux", "celui", "celle", "celles", "ces", "cet",
    "parce", "pourquoi", "comment", "quand",
    "ainsi", "alors", "donc", "mais", "car", "puis", "aussi",
    "cependant", "néanmoins", "toutefois", "pourtant",
    "selon", "après", "avant", "pendant", "depuis", "malgré", "comme",
    "lors", "lorsque", "tandis", "voici", "voilà", "ici",
    "nous", "vous", "ils", "elles", "leur", "leurs", "elle",
    "quelques", "plusieurs", "tous", "tout", "toute", "toutes",
    "tel", "telle", "tels", "telles",
    "dans", "pour", "sur", "sous", "par", "sans", "vers", "avec", "entre",
    "chez", "hors",
    "this", "that", "these", "those", "because", "therefore", "thus",
    "however", "moreover", "furthermore", "while", "when", "where", "why",
    "although", "despite", "indeed", "instead", "also", "there", "here",
    "they", "their", "them", "such", "some", "many", "most", "each", "every",
}

_ANSWER_LEAK_PATTERNS = [
    r"^\s*(?:according to|per|as stated in|as per|based on)\s+"
    r"(?:the\s+)?(?:text|document|abstract|article|paper|passage|timeline|figure|author)[,:\s]+",
    r"^\s*the\s+(?:text|document|abstract|article|paper|passage|timeline|figure)\s+"
    r"(?:says|states|explains|mentions|describes|notes|indicates|reports|claims)[,:\s]+",
    r"^\s*(?:selon|d['']après|comme (?:le )?(?:précise|indique|mentionne))\s+"
    r"(?:le\s+)?(?:texte|document|article|passage|auteur)[,:\s]+",
    r"^\s*le\s+(?:texte|document|article|passage)\s+"
    r"(?:dit|précise|explique|mentionne|décrit|indique)[,:\s]+",
]
_ANSWER_LEAK_RE: re.Pattern | None = None


# ── Data type ─────────────────────────────────────────────────────────────────

@dataclass
class QAPair:
    question: str
    answer: str
    source_file: str = ""
    confidence: float = 0.5
    dropped_pairs: list[dict] = field(default_factory=list, repr=False)


# ── QAService ─────────────────────────────────────────────────────────────────

class QAService:
    """Generate, filter, deduplicate, and refine Q&A pairs for fine-tuning."""

    def __init__(self, llm_client: LLMClient, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self.llm = llm_client
        self.embedding_model = embedding_model
        self._sentence_model: Any = None
        self._dropped: list[dict] = []

    # ── Public entry points ───────────────────────────────────────────────────

    def generate_for_file(
        self,
        filename: str,
        profile: dict,
        cleaned_text: str | None = None,
        task: str = "question-answering",
        debug_chunks_path=None,
        style: str = "general",
        model_id: str = "",
    ) -> list[QAPair]:
        """Generate pairs for one file. Summarization uses (document→summary) pairs;
        all other tasks use Q&A pairs."""
        file_kind = profile.get("file_kind", "")
        self._dropped = []

        if file_kind in ("pdf", "txt"):
            fallback = profile.get("full_text", "") or profile.get("metadata", {}).get("text", "") or ""
            if file_kind == "pdf":
                ocr_pages: dict = profile.get("ocr_pages", {})
                if ocr_pages:
                    ocr_combined = "\n\n".join(ocr_pages.values())
                    fallback = (fallback + "\n\n" + ocr_combined).strip() if fallback else ocr_combined
            text = cleaned_text or fallback

            if task == "summarization":
                return self.generate_for_summarization(filename, text, model_id=model_id)

            doc_summary = self._summarize_text(text)
            chunks = self._chunk_nltk(text, chunk_size=500, context=200)
            if debug_chunks_path is not None:
                debug_chunks_path.parent.mkdir(parents=True, exist_ok=True)
                debug_chunks_path.write_text(
                    "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n",
                    encoding="utf-8",
                )
                logger.info("debug_chunks saved: %d chunks → %s", len(chunks), debug_chunks_path)
            pairs: list[QAPair] = []
            for chunk in chunks:
                pairs.extend(self._generate_text_chunk(filename, chunk, doc_summary=doc_summary, style=style))
            return pairs

        if file_kind in ("csv", "excel"):
            rows = profile.get("rows") or []
            columns = profile.get("columns") or []
            chunks = self._chunk_csv(rows, columns)
            pairs = []
            for chunk in chunks:
                pairs.extend(self._generate_csv_chunk(filename, chunk, profile, task))
            return pairs

        logger.warning("generate_for_file: unsupported file_kind=%s for %s", file_kind, filename)
        return []

    def generate_for_summarization(
        self, filename: str, text: str, model_id: str = ""
    ) -> list[QAPair]:
        """Generate (document → summary) pairs for summarization fine-tuning.

        Chunk size = 2000 chars (~330 words). A 500-char chunk (~80 words) produces
        paraphrases, not compressions. CNN/DailyMail (BART/PEGASUS baseline) uses
        ~800-word articles → ~55-word summaries (14:1). 330 words → ~60-word summary
        achieves a 5:1 ratio — the minimum for real compression learning.
        """
        if not text.strip():
            return []
        chunks = self._chunk_nltk(text, chunk_size=_SUM_CHUNK_SIZE, context=0)
        pairs: list[QAPair] = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "").strip()
            if len(chunk_text.split()) < 50:
                continue
            prompt = (
                "Résume le texte suivant en 3 à 5 phrases concises, "
                "en capturant les idées principales :\n\n" + chunk_text
            )
            try:
                summary = self.llm.complete(
                    system=(
                        "Tu es un expert en résumé de texte. "
                        "Produis des résumés concis, fidèles et informatifs."
                    ),
                    user=prompt,
                ).strip()
                if len(summary.split()) < 10:
                    continue
                pairs.append(QAPair(
                    question=chunk_text,
                    answer=summary,
                    source_file=f"{filename} (sum chunk {i + 1})",
                    confidence=0.85,
                ))
            except Exception as exc:
                logger.error("sum_chunk_failed file=%s chunk=%d: %s", filename, i, exc)
        logger.info(
            "generate_for_summarization file=%s chunks=%d pairs=%d model=%s",
            filename, len(chunks), len(pairs), model_id or "default",
        )
        return pairs

    def deduplicate(self, pairs: list[QAPair]) -> tuple[list[QAPair], list[QAPair]]:
        """Lexical + semantic deduplication. Returns (unique, dropped)."""
        dropped: list[QAPair] = []
        unique = self._lexical_dedup(pairs, dropped)
        unique = self._semantic_dedup(unique, dropped)
        return unique, dropped

    def evaluate_and_refine(
        self, pairs: list[QAPair], sample_size: int = 20
    ) -> tuple[list[QAPair], list[QAPair]]:
        """Sample up to sample_size pairs, score with LLM, KEEP/REWRITE/DROP. Returns (kept, dropped)."""
        if not pairs:
            return pairs, []

        sample_indices = random.sample(range(len(pairs)), min(sample_size, len(pairs)))
        sampled = [pairs[i] for i in sample_indices]

        scores = self._score_batch(sampled)
        kept: list[QAPair] = []
        dropped: list[QAPair] = []
        sampled_set = {id(p) for p in sampled}

        sampled_iter = iter(zip(sampled, scores))
        for pair in pairs:
            if id(pair) not in sampled_set:
                kept.append(pair)
                continue
            p, score = next(sampled_iter)
            if score >= _SCORE_REFINE:
                kept.append(p)
            elif score >= _SCORE_DISCARD:
                refined = self._refine_pair(p, score)
                if (
                    refined
                    and not _is_vague(refined.answer)
                    and not _is_paraphrase(refined.question, refined.answer)
                ):
                    kept.append(refined)
                else:
                    dropped.append(p)
            else:
                dropped.append(p)

        logger.info(
            "qa_eval_refine total=%d kept=%d dropped=%d", len(pairs), len(kept), len(dropped)
        )
        return kept, dropped

    # ── Summarization ─────────────────────────────────────────────────────────

    def _summarize_text(self, text: str) -> str:
        """Summarize the full document text; used as context for every chunk."""
        if not text.strip():
            return ""
        prompt = f"""Please provide a comprehensive summary of the following text.
The summary should:
1. Capture the main ideas and key points
2. Preserve important details and examples
3. Maintain the logical flow of the original text
4. Be concise while retaining the essential information

Text to summarize:
{text}"""
        try:
            summary = self.llm.complete(
                system="You are a document summarization expert.", user=prompt
            )
            logger.info("doc_summary generated: %d chars", len(summary))
            return summary
        except Exception as exc:
            logger.warning("doc_summarization_failed: %s", exc)
            return ""

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk_nltk(self, text: str, chunk_size: int = 500, context: int = 200) -> list[dict]:
        """NLTK sentence splitting + sentence-aware grouping at ~chunk_size chars.

        Each chunk groups whole sentences up to chunk_size chars.
        prev_context / next_context = last/first `context` chars of the
        neighbouring chunk — passed to the LLM but questions target the core only.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Group sentences into core chunks of ~chunk_size chars
        cores: list[str] = []
        current: list[str] = []
        current_len = 0
        for sent in sentences:
            sent_len = len(sent)
            if current and current_len + sent_len > chunk_size:
                cores.append(" ".join(current))
                current = [sent]
                current_len = sent_len
            else:
                current.append(sent)
                current_len += sent_len + 1
        if current:
            cores.append(" ".join(current))

        # Build chunks with bidirectional context
        chunks: list[dict] = []
        for i, core in enumerate(cores):
            prev_ctx = cores[i - 1][-context:] if i > 0 else ""
            next_ctx = cores[i + 1][:context] if i < len(cores) - 1 else ""
            chunks.append({
                "chunk_id": i,
                "text": core,
                "prev_context": prev_ctx,
                "next_context": next_ctx,
                "char_count": len(core),
                "sentence_count": core.count("."),
                "strategy": "nltk_500",
                "has_overlap": False,
                "overlap_size": 0,
            })

        logger.info("nltk_chunking: %d chunks (size=%d context=%d)", len(chunks), chunk_size, context)
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences — NLTK first, regex fallback."""
        try:
            import nltk
            try:
                sentences = nltk.sent_tokenize(text)
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
                sentences = nltk.sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except ImportError:
            return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _chunk_text(
        self, text: str, chunk_size_chars: int = 1500, overlap_chars: int = 200
    ) -> list[dict]:
        if not text.strip():
            return []
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        boundaries = self._semantic_boundaries(sentences, chunk_size_chars)
        if boundaries is None:
            return self._chunk_text_fixed(sentences, chunk_size_chars, overlap_chars)

        min_chars = max(200, chunk_size_chars // 3)
        max_chars = max(chunk_size_chars * 2, chunk_size_chars + 400)
        chunks: list[dict] = []
        cid = start = 0

        while start < len(sentences):
            end = start + 1
            current_len = len(sentences[start])
            while end < len(sentences):
                next_len = current_len + len(sentences[end]) + 1
                hit_boundary = end in boundaries
                if hit_boundary and current_len >= min_chars:
                    break
                if next_len > max_chars and current_len >= min_chars:
                    break
                if next_len > max_chars:
                    end += 1
                    current_len = next_len
                    break
                current_len = next_len
                end += 1
            chunk_sents = sentences[start:end]
            chunk_text = " ".join(chunk_sents)
            chunks.append({
                "chunk_id": cid, "start_sent_idx": start, "end_sent_idx": end,
                "text": chunk_text, "char_count": len(chunk_text),
                "sentence_count": len(chunk_sents), "has_overlap": False,
                "overlap_size": 0, "strategy": "semantic",
            })
            cid += 1
            start = end
        return chunks

    def _chunk_text_fixed(
        self, sentences: list[str], chunk_size_chars: int, overlap_chars: int
    ) -> list[dict]:
        chunks, cid, si = [], 0, 0
        while si < len(sentences):
            chunk_sents: list[str] = []
            chunk_len = 0
            start_si = si
            while si < len(sentences):
                s = sentences[si]
                if chunk_len + len(s) > chunk_size_chars and chunk_sents:
                    break
                chunk_sents.append(s)
                chunk_len += len(s) + 1
                si += 1
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(chunk_sents):
                if overlap_len + len(s) > overlap_chars and overlap_sents:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s) + 1
            chunk_text = " ".join(chunk_sents)
            chunks.append({
                "chunk_id": cid, "start_sent_idx": start_si, "end_sent_idx": si,
                "text": chunk_text, "char_count": len(chunk_text),
                "sentence_count": len(chunk_sents), "has_overlap": cid > 0,
                "overlap_size": overlap_len if cid > 0 else 0, "strategy": "fixed",
            })
            cid += 1
            if si < len(sentences) and overlap_sents:
                si -= len(overlap_sents)
        return chunks

    def _semantic_boundaries(self, sentences: list[str], target_chars: int) -> set[int] | None:
        if len(sentences) < 5:
            return None
        model = self._get_sentence_model()
        if model is None:
            return None
        try:
            embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normed = embeddings / norms
            sims = np.sum(normed[:-1] * normed[1:], axis=1)
            distances = 1.0 - sims
            avg_len = max(1, sum(len(s) for s in sentences) // len(sentences))
            target_sents = max(3, target_chars // avg_len)
            target_breaks = max(1, len(sentences) // target_sents)
            pct = max(50.0, 100.0 * (1 - target_breaks / max(1, len(distances))))
            threshold = float(np.percentile(distances, pct))
            return {i + 1 for i, d in enumerate(distances) if d >= threshold}
        except Exception as exc:
            logger.warning("semantic_chunking_failed, falling back: %s", exc)
            return None

    @staticmethod
    def _adaptive_csv_params(n: int) -> tuple[int, int]:
        if n <= 50:   return n, 0
        if n <= 200:  return 50, 5
        if n <= 1000: return 100, 10
        return 200, 20

    def _chunk_csv(
        self,
        rows: list[dict],
        columns: list[dict],
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[dict]:
        n = len(rows)
        if chunk_size is None or overlap is None:
            chunk_size, overlap = self._adaptive_csv_params(n)
        if n == 0:
            return []
        chunks, step, idx, cid = [], max(1, chunk_size - overlap), 0, 0
        while idx < n:
            start = max(0, idx - (overlap if cid > 0 else 0))
            end = min(n, idx + chunk_size)
            chunks.append({
                "chunk_id": cid, "start_idx": start, "end_idx": end,
                "rows": rows[start:end], "row_count": end - start,
                "columns": columns, "has_overlap": cid > 0,
                "overlap_size": overlap if cid > 0 else 0,
            })
            idx += step
            cid += 1
            if step == 0:
                break
        return chunks

    # ── QA generation ─────────────────────────────────────────────────────────

    def _generate_text_chunk(
        self,
        filename: str,
        chunk: dict,
        doc_summary: str = "",
        style: str = "general",
    ) -> list[QAPair]:
        text = chunk.get("text", "")
        cid = chunk.get("chunk_id", 0)
        if not text.strip() or self._is_boilerplate_chunk(text):
            return []

        prev_ctx = chunk.get("prev_context", "")
        next_ctx = chunk.get("next_context", "")

        context_block = (
            f"Given the following document summary:\n{doc_summary}\n\n"
            if doc_summary else ""
        )
        prev_block = f"[Previous context]:\n{prev_ctx}\n\n" if prev_ctx else ""
        next_block = f"\n\n[Following context]:\n{next_ctx}" if next_ctx else ""

        style_instruction = GENERATION_STYLES.get(style, GENERATION_STYLES["general"])
        prompt = f"""{context_block}{prev_block}[Current chunk — generate questions from THIS]:
---
{text}
---{next_block}

{style_instruction}

Generate 3 to 5 high-quality, DIVERSE question-answer pairs from the [Current chunk] ONLY.

STRICT RULES:
- Focus EXCLUSIVELY on facts, concepts, and details present in [Current chunk]. Do NOT ask about content that appears only in [Previous context] or [Following context].
- NEVER mention "the document", "the rapport", "the text", "the chunk", "the passage", "the source", or any reference to where the information comes from.
- Write every question as if testing knowledge directly — not asking what a document says.
- Each question must target a DIFFERENT concept, fact, or angle — no two questions should be about the same idea.
- BAD:  "According to the rapport, what is a Deep Agent?"
- GOOD: "What is a Deep Agent and what makes it different from a classic workflow?"
- Each answer must be a complete, standalone explanation derived from the chunk content.

FORMAT: Return a JSON array only, no other text:
[{{"question": "...", "answer": "...", "confidence": 0.0}}]

EXAMPLE:
[
  {{
    "question": "What is gradient descent and how does it optimize a neural network?",
    "answer": "Gradient descent is an optimization algorithm that minimizes a loss function by iteratively adjusting model weights in the direction of the steepest descent of the gradient. In each step, the weights are updated by subtracting a fraction (the learning rate) of the gradient of the loss with respect to the weights. This process repeats until the loss converges to a minimum, effectively training the neural network to make better predictions.",
    "confidence": 0.9
  }}
]"""

        try:
            raw = self.llm.complete(
                system="You are an expert at creating high-quality question-answer pairs for LLM fine-tuning.",
                user=prompt,
            )
            pairs: list[QAPair] = []
            for it in _parse_qa_json(raw):
                if "question" not in it or "answer" not in it:
                    continue
                answer = _sanitize_answer(it["answer"])
                if not answer or len(answer.split()) < 10:
                    continue
                c = float(it.get("confidence", 0.5))
                pairs.append(QAPair(
                    question=str(it["question"]).strip(),
                    answer=answer,
                    source_file=f"{filename} (seg {cid + 1})",
                    confidence=c if 0 <= c <= 1 else 0.5,
                ))
            pairs = self._filter_pairs(pairs, f"{filename}:chunk{cid}")
            pairs = self._verify_grounding(pairs, text, f"{filename}:chunk{cid}")
            return pairs
        except Exception as exc:
            logger.error("text_qa_failed file=%s chunk=%d error=%s", filename, cid, exc)
            return []

    def _generate_csv_chunk(
        self, filename: str, chunk: dict, profile: dict, task: str
    ) -> list[QAPair]:
        rows = chunk.get("cleaned_rows") or chunk.get("rows", [])
        if not rows:
            return []

        richness = self._csv_richness(chunk)
        min_qa = 1 if richness < _MIN_RICHNESS else max(3, int(richness * 8))
        cid = chunk.get("chunk_id", 0)

        col_summaries = profile.get("column_summaries") or profile.get("columns", [])
        col_lines = []
        for cs in col_summaries:
            name = cs.get("name", "?")
            typ  = cs.get("type", cs.get("dtype", "?"))
            null = cs.get("null_ratio", 0)
            parts = [f"  • {name} ({typ}, null={null:.0%})"]
            if cs.get("range"):
                parts.append(f"range={cs['range']}")
            if cs.get("top_values"):
                parts.append(f"top: {', '.join(str(v) for v in cs['top_values'][:5])}")
            col_lines.append(" | ".join(parts))
        col_context = "\n".join(col_lines) or "  (no column metadata)"

        rows_str = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))[:4000]

        prompt = f"""OUTPUT LANGUAGE: all Q&A must be written in English.
You are an expert data analyst generating Q&A pairs for LLM fine-tuning.

DATASET: {filename} — Segment #{cid + 1} (richness={richness:.2f})
Task context: {task}

COLUMN SCHEMA:
{col_context}

DATA (None = missing value):
{rows_str}

Generate AT LEAST {min_qa} high-quality Q&A pairs covering:
1. STATISTICAL  — averages, totals, ranges (cite real numbers)
2. COMPARATIVE  — differences between rows/groups
3. PATTERN      — trends, correlations between columns
4. ANOMALY      — outliers, missing values
5. DOMAIN       — real-world implications of the data
6. PREDICTIVE   — expected values given other columns

QUALITY RULES:
- Every answer must cite REAL values from the data
- Minimum 20 words per answer
- Never ask yes/no questions
- Never invent values not in the data
- Questions must be self-contained

STRICT FORMAT (JSON array only):
[{{"question":"...","answer":"...","confidence":0.0}}]"""

        try:
            raw = self.llm.complete(system="You are a fine-tuning dataset generator.", user=prompt)
            pairs: list[QAPair] = []
            for it in _parse_qa_json(raw):
                if "question" not in it or "answer" not in it:
                    continue
                c = float(it.get("confidence", 0.5))
                pairs.append(QAPair(
                    question=it["question"], answer=it["answer"],
                    source_file=f"{filename} (seg {cid+1}, richness:{richness:.2f})",
                    confidence=c if 0 <= c <= 1 else 0.5,
                ))
            return self._filter_pairs(pairs, f"{filename}:chunk{cid}")
        except Exception as exc:
            logger.error("csv_qa_failed file=%s chunk=%d error=%s", filename, cid, exc)
            return []

    # ── Filtering ─────────────────────────────────────────────────────────────

    def _filter_pairs(self, pairs: list[QAPair], src: str) -> list[QAPair]:
        kept: list[QAPair] = []
        for p in pairs:
            if _is_metadata_question(p.question):
                continue
            if _is_vague(p.answer):
                continue
            if _is_paraphrase(p.question, p.answer):
                continue
            if _DOC_REF_PATTERNS.search(p.question):
                continue
            kept.append(p)
        removed = len(pairs) - len(kept)
        if removed:
            logger.info("qa_filter [%s]: removed=%d kept=%d", src, removed, len(kept))
        return kept

    def _verify_grounding(
        self, pairs: list[QAPair], source_text: str, src: str
    ) -> list[QAPair]:
        if not source_text or not pairs:
            return pairs
        source_tokens = _extract_specific_tokens(source_text)
        source_lower = source_text.lower()
        kept: list[QAPair] = []
        for p in pairs:
            claim_tokens = _extract_specific_tokens(p.answer)
            if len(claim_tokens) < 3:
                kept.append(p)
                continue
            missing = {
                t for t in claim_tokens
                if t not in source_tokens and t not in source_lower
            }
            if len(missing) / len(claim_tokens) > 0.50:
                logger.info("qa_ungrounded [%s]: missing=%s", src, sorted(missing)[:5])
                continue
            kept.append(p)
        return kept

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _lexical_dedup(self, pairs: list[QAPair], dropped: list[QAPair]) -> list[QAPair]:
        seen_exact: set[str] = set()
        seen_prefix: set[str] = set()
        seen_sig: list[set[str]] = []
        unique: list[QAPair] = []

        for pair in pairs:
            norm = re.sub(r"[^\w\s]", "", pair.question.lower()).strip()
            if norm in seen_exact:
                dropped.append(pair)
                continue
            prefix = norm[:60]
            if len(prefix) >= 30 and prefix in seen_prefix:
                dropped.append(pair)
                continue
            sig = _sig_words(pair.question)
            if sig and any(
                len(sig & ex) / max(len(sig | ex), 1) > 0.50 for ex in seen_sig
            ):
                dropped.append(pair)
                continue
            seen_exact.add(norm)
            seen_prefix.add(prefix)
            seen_sig.append(sig)
            unique.append(pair)

        removed = len(pairs) - len(unique)
        if removed:
            logger.info("lexical_dedup: %d → %d (%d removed)", len(pairs), len(unique), removed)
        return unique

    def _semantic_dedup(
        self,
        pairs: list[QAPair],
        dropped: list[QAPair],
        near_dups: list[QAPair] | None = None,
    ) -> list[QAPair]:
        """Semantic dedup using question-only embeddings for the primary pass.

        near_dups: if provided, semantically similar questions go here (for
        rephrasing into an eval set) instead of being silently dropped.
        Answer-only duplicates (true copies) always go to dropped.
        """
        if len(pairs) < 2:
            return pairs
        model = self._get_sentence_model()
        if model is None:
            return pairs

        try:
            mname = self.embedding_model.lower()
            e5 = "e5" in mname and "intfloat" in mname

            # ── Pass 1: question-only similarity ─────────────────────────────
            # Using questions only avoids false duplicates caused by shared
            # domain vocabulary in answers (e.g. all answers about DevOps).
            q_texts = [p.question for p in pairs]
            if e5:
                q_texts = [f"query: {t}" for t in q_texts]
            q_emb = np.array(
                model.encode(q_texts, convert_to_tensor=False, show_progress_bar=False),
                dtype=np.float32,
            )
            q_emb /= np.maximum(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-9)
            q_sim = q_emb @ q_emb.T

            keep = [True] * len(pairs)
            for i in range(len(pairs)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(pairs)):
                    if keep[j] and q_sim[i, j] >= _EMBED_SIM_THRESH:
                        keep[j] = False
                        # Near-duplicate question → candidate for eval rephrase
                        if near_dups is not None:
                            near_dups.append(pairs[j])
                        else:
                            dropped.append(pairs[j])

            unique = [p for p, k in zip(pairs, keep) if k]
            if len(unique) < 2:
                return unique

            # ── Pass 2: answer-only similarity (true copies → always drop) ───
            ans_texts = [p.answer for p in unique]
            if e5:
                ans_texts = [f"query: {t}" for t in ans_texts]
            ans_emb = np.array(
                model.encode(ans_texts, convert_to_tensor=False, show_progress_bar=False),
                dtype=np.float32,
            )
            ans_emb /= np.maximum(np.linalg.norm(ans_emb, axis=1, keepdims=True), 1e-9)
            ans_sim = ans_emb @ ans_emb.T
            keep_a = [True] * len(unique)
            for i in range(len(unique)):
                if not keep_a[i]:
                    continue
                for j in range(i + 1, len(unique)):
                    if keep_a[j] and ans_sim[i, j] >= _ANSWER_SIM_THRESH:
                        keep_a[j] = False
                        dropped.append(unique[j])

            final = [p for p, k in zip(unique, keep_a) if k]
            logger.info(
                "semantic_dedup: %d → %d unique | %d near_dups | %d true_dropped",
                len(pairs), len(final),
                len(near_dups) if near_dups is not None else 0,
                len(dropped),
            )
            return final

        except Exception as exc:
            logger.warning("embed_dedup_failed: %s", exc)
            return pairs

    # ── Eval rephrase ─────────────────────────────────────────────────────────

    def _rephrase_question(self, pair: QAPair) -> QAPair | None:
        """Ask the LLM to rephrase the question for an eval set."""
        prompt = (
            "Rephrase the following question to test the same knowledge from a different angle.\n"
            "Rules:\n"
            "- Same factual meaning, different wording\n"
            "- No document references ('the text', 'the rapport', etc.)\n"
            "- Must sound like a natural standalone question\n\n"
            f"Original: {pair.question}\n\n"
            'Return ONLY JSON: {"question": "..."}'
        )
        try:
            raw = self.llm.complete(
                system="You create evaluation question variants for LLM benchmarking.",
                user=prompt,
            )
            m = re.search(r'\{[^}]+\}', raw)
            if not m:
                return None
            q = json.loads(m.group(0)).get("question", "").strip()
            if not q or q.lower() == pair.question.lower():
                return None
            return QAPair(
                question=q,
                answer=pair.answer,
                source_file=pair.source_file + " [eval]",
                confidence=pair.confidence,
            )
        except Exception:
            return None

    def rephrase_for_eval(self, near_dups: list[QAPair], max_pairs: int = 100) -> list[QAPair]:
        """Rephrase near-duplicate questions to build a paraphrase eval set.

        Train: What is DevOps?
        Eval:  Can you explain the concept of DevOps?
        → Tests real comprehension, not memorisation.
        """
        eval_pairs: list[QAPair] = []
        for pair in near_dups[:max_pairs]:
            rephrased = self._rephrase_question(pair)
            if rephrased:
                eval_pairs.append(rephrased)
        logger.info(
            "rephrase_for_eval: %d candidates → %d eval pairs", len(near_dups), len(eval_pairs)
        )
        return eval_pairs

    # ── LLM judge ─────────────────────────────────────────────────────────────

    def _score_batch(self, pairs: list[QAPair]) -> list[float]:
        if not pairs:
            return []
        results: list[float] = []
        for i in range(0, len(pairs), _LLM_EVAL_BATCH):
            batch = pairs[i : i + _LLM_EVAL_BATCH]
            numbered = "\n".join(
                f"{j+1}. Q: {p.question}\n   A: {p.answer}" for j, p in enumerate(batch)
            )
            prompt = (
                f"Score each Q&A pair 0.0–1.0 for LLM fine-tuning quality.\n"
                f"Return ONLY a JSON array of {len(batch)} floats.\n\nPAIRS:\n{numbered}"
            )
            try:
                raw = self.llm.complete(system="You are a QA quality evaluator.", user=prompt)
                m = re.search(r"\[[\s\S]*?\]", raw)
                if m:
                    for s in json.loads(m.group(0)):
                        results.append(float(s) if isinstance(s, (int, float)) else 0.5)
                else:
                    results.extend([0.5] * len(batch))
            except Exception:
                results.extend([0.5] * len(batch))
        return results

    def _refine_pair(self, pair: QAPair, score: float) -> QAPair | None:
        prompt = (
            f"Rewrite this Q&A pair for clarity and pedagogical value "
            f"(current score {score:.2f}/1.0).\n\n"
            "RULES:\n"
            "1. Do NOT add any new fact not in the original answer.\n"
            "2. You may ONLY: reorder ideas, clarify phrasing, remove filler, tighten wording.\n"
            "3. Keep answer between 30 and 200 words.\n"
            "4. Return empty if the original answer is hollow — do NOT save it by adding info.\n\n"
            f"ORIGINAL Q: {pair.question}\n"
            f"ORIGINAL A: {pair.answer}\n\n"
            'Return ONLY JSON: {"question":"...","answer":"..."}'
        )
        try:
            raw = self.llm.complete(system="You are a QA pair refiner.", user=prompt)
            m = re.search(r"\{[\s\S]*?\}", raw)
            if not m:
                return None
            data = json.loads(m.group(0))
            if "question" not in data or "answer" not in data:
                return None
            orig_sig = _sig_words(pair.answer)
            new_sig  = _sig_words(data["answer"])
            if orig_sig and new_sig and len(orig_sig & new_sig) / len(new_sig) < 0.50:
                return None
            refined_answer = _sanitize_answer(data["answer"])
            if _is_metadata_question(data["question"]):
                return None
            return QAPair(
                question=data["question"],
                answer=refined_answer,
                source_file=pair.source_file + " [refined]",
                confidence=min(1.0, pair.confidence + 0.15),
            )
        except Exception:
            return None

    # ── Richness ──────────────────────────────────────────────────────────────

    @staticmethod
    def _text_richness(chunk: dict) -> float:
        text = chunk.get("text", "")
        if not text:
            return 0.0
        words      = text.split()
        sentences  = [s for s in text.split(".") if s.strip()]
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        lower      = text.lower().split()
        vocab_div  = len(set(lower)) / max(1, len(lower))
        con_den    = sum(1 for w in lower if w and w[0].isupper()) / max(1, len(lower))
        return (
            min(1.0, len(text) / 3000.0)    * 0.25 +
            min(1.0, len(sentences) / 50.0)  * 0.25 +
            vocab_div                         * 0.20 +
            min(1.0, len(paragraphs) / 10.0) * 0.15 +
            con_den                           * 0.15
        )

    @staticmethod
    def _csv_richness(chunk: dict) -> float:
        rows = chunk.get("rows", [])
        cols = chunk.get("columns", [])
        if not rows or not cols:
            return 0.0
        diversity = sum(
            len({str(r.get(c.get("name", ""), "")) for r in rows if c.get("name", "") in r})
            / max(1, len(rows))
            for c in cols
        ) / max(1, len(cols))
        col_score = min(1.0, len(cols) / 8.0)
        row_score = min(1.0, len(rows) / 100.0)
        null_penalty = sum(
            sum(1 for v in row.values() if v is None or str(v).strip() == "") / max(1, len(row))
            for row in rows
        ) / max(1, len(rows))
        return col_score * 0.30 + diversity * 0.30 + row_score * 0.20 + max(0.0, 1.0 - null_penalty) * 0.20

    @staticmethod
    def _is_boilerplate_chunk(text: str) -> bool:
        if not text:
            return True
        words = text.split()
        word_count = len(words)
        lower = text.lower()
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        # Table of contents: lines ending with page numbers (.... 5)
        if len(re.findall(r"\.{4,}\s*\d+", text)) >= 4 and word_count < 300:
            return True

        # Reference list: 3+ numbered citations like "- 1. Title..." or "1. Author..."
        if len(re.findall(r"^\s*[-•]?\s*\d+\.\s+[\"'“]?[A-Z]", text, re.MULTILINE)) >= 3:
            return True

        # Legal disclaimer: multiple legal keywords
        legal_hits = sum(1 for phrase in [
            "all rights reserved", "no representations", "warranties of any kind",
            "fitness for a particular", "applicable software license",
            "specifically disclaims", "subject to change without notice",
        ] if phrase in lower)
        if legal_hits >= 2:
            return True

        # Cover page / title page: mostly very short lines (names, roles, affiliations)
        if lines:
            short_lines = sum(1 for ln in lines if 0 < len(ln.split()) < 6)
            if short_lines / max(len(lines), 1) > 0.60 and word_count < 150:
                return True

        # High capitalisation ratio in short chunks (headings-only chunk)
        if word_count < 60:
            cap_ratio = sum(1 for w in words if w and w[0].isupper()) / max(1, word_count)
            if cap_ratio > 0.45:
                return True

        # Inline reference-only lines (e.g. "[2](#page-9-0)\n• Docker is...")
        if len(re.findall(r"\b\d{1,2}\.\s+[A-Z][^.\n]{0,80}(?:https?://|www\.)", text)) >= 4 and word_count < 500:
            return True

        sign_hits = sum(1 for s in _BOILERPLATE_SIGNS if s in lower)
        if sign_hits >= 3 and word_count < 250:
            return True

        return False

    def _get_sentence_model(self) -> Any:
        if not _ST_AVAILABLE:
            return None
        if self._sentence_model is None:
            try:
                self._sentence_model = SentenceTransformer(self.embedding_model)
            except Exception as exc:
                logger.warning("sentence_model_load_failed: %s", exc)
        return self._sentence_model


# ── Module-level helpers ──────────────────────────────────────────────────────

def _sig_words(text: str) -> set[str]:
    return {w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) > 3 and w not in _STOPWORDS}


def _is_vague(answer: str) -> bool:
    s = answer.strip().rstrip(".!?").lower()
    return len(s.split()) < 5 or s in _VAGUE_ANSWERS


def _is_paraphrase(q: str, a: str) -> bool:
    aw = _sig_words(a)
    return not aw or len(aw - _sig_words(q)) / len(aw) < 0.20


def _is_metadata_question(question: str) -> bool:
    global _METADATA_RE
    if _METADATA_RE is None:
        _METADATA_RE = re.compile("|".join(_METADATA_PATTERNS), re.IGNORECASE)
    return bool(_METADATA_RE.search(question))


def _sanitize_answer(answer: str) -> str:
    global _ANSWER_LEAK_RE
    if _ANSWER_LEAK_RE is None:
        _ANSWER_LEAK_RE = re.compile("|".join(_ANSWER_LEAK_PATTERNS), re.IGNORECASE)
    cleaned = _ANSWER_LEAK_RE.sub("", answer).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _extract_specific_tokens(text: str) -> set[str]:
    if not text:
        return set()
    tokens: set[str] = set()
    tokens |= {m.lower() for m in re.findall(r"\b\d{3,}(?:[.,]\d+)?%?\b", text)}
    tokens |= {m.lower() for m in re.findall(r"\b[A-Z]{2,}[A-Z0-9]*\b", text)}
    tokens |= {m.lower() for m in re.findall(r"\b[A-Za-z]+[A-Z][A-Za-z]+\b", text)}
    tokens |= {m.lower() for m in re.findall(r"\b[A-Z][a-z]+-[A-Za-z]+\b", text)}
    return {t for t in tokens if t not in _GROUNDING_IGNORE and len(t) > 1}


def _parse_qa_json(raw: str) -> list[dict]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    m = re.search(r"\[[\s\S]*\]", cleaned)
    if not m:
        return []
    json_str = m.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    try:
        # Escape bare control chars inside string values
        result = []
        i = 0
        while i < len(json_str):
            ch = json_str[i]
            if ch == "\\" and i + 1 < len(json_str):
                result.append(ch)
                result.append(json_str[i + 1])
                i += 2
                continue
            if ch == "\n":
                result.append("\\n")
            elif ch == "\r":
                result.append("\\r")
            elif ch == "\t":
                result.append("\\t")
            elif ord(ch) < 0x20:
                pass
            else:
                result.append(ch)
            i += 1
        return json.loads("".join(result))
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(json_str.replace("'", '"'))
    except json.JSONDecodeError:
        return []


# ── Dataset export ────────────────────────────────────────────────────────────

def export_dataset(
    pairs: list[QAPair],
    task: str = "question-answering",
    system_prompt: str = "You are a helpful assistant.",
    model_id: str = "",
) -> list[dict]:
    """Convert QAPairs to fine-tuning format records.

    Summarization format is model-aware:
      seq2seq (T5/BART/PEGASUS) → {"document": "...", "summary": "..."}
      chat    (Mistral/LLaMA…)  → {"messages": [{"role":"user",...}, {"role":"assistant",...}]}
      alpaca  (default)         → {"instruction": "...", "input": "...", "output": "..."}
    """
    # ── Summarization: format depends on model architecture ───────────────────
    if task == "summarization":
        fmt = _detect_summarization_format(model_id)
        records: list[dict] = []
        for i, p in enumerate(pairs):
            instruction = _SUM_INSTRUCTIONS[i % len(_SUM_INSTRUCTIONS)]
            if fmt == "seq2seq":
                records.append({"document": p.question, "summary": p.answer})
            elif fmt == "chat":
                records.append({"messages": [
                    {"role": "user",      "content": f"{instruction}\n\n{p.question}"},
                    {"role": "assistant", "content": p.answer},
                ]})
            else:  # alpaca
                records.append({
                    "instruction": instruction,
                    "input":       p.question,
                    "output":      p.answer,
                })
        return records

    # ── All other tasks ───────────────────────────────────────────────────────
    fmt = _TASK_FORMATS.get(task, "alpaca")
    records = []
    for p in pairs:
        if fmt == "sharegpt":
            records.append({
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human",  "value": p.question},
                    {"from": "gpt",    "value": p.answer},
                ]
            })
        elif fmt == "alpaca_label":
            records.append({"instruction": p.question, "input": "", "output": p.answer})
        else:  # alpaca
            records.append({"instruction": p.question, "input": "", "output": p.answer})
    return records
