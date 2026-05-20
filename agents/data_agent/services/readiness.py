"""Readiness judge wrapper.

Public API
----------
ReadinessService.evaluate(pairs, task, peft, model_name, summaries) -> dict
"""

from __future__ import annotations

import logging
from typing import Any

from agents.llm_client import LLMClient
from agents.data_agent.services.qa import QAPair

logger = logging.getLogger(__name__)

# Kept as fallback when no model_name is provided
_PEFT_MIN_ROWS: dict[str, int] = {
    "lora":      100,
    "qlora":      50,
    "lora_int8":  50,
    "dora":      100,
    "full":     1000,
    "auto":       50,
}


class ReadinessService:
    """6-axis LLM judge — decides if the Q&A dataset is ready for fine-tuning."""

    _READY_HIGH       = 0.80
    _READY_MED        = 0.65

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def evaluate(
        self,
        pairs: list[QAPair],
        task: str = "conversational-qa",
        peft: str = "auto",
        model_name: str = "",
        file_summaries: list[str] | None = None,
        sample_size: int = 12,
    ) -> dict[str, Any]:
        from data.model_registry import get_requirements

        req = get_requirements(model_name, peft, task) if model_name else None
        min_viable = req["min_viable"] if req else _PEFT_MIN_ROWS.get(peft, 50)
        target     = req["target"]     if req else min_viable * 3
        saturate   = req["saturate"]   if req else min_viable * 10

        if not pairs:
            return {
                "readiness_score": 0.0,
                "ready_for_finetuning": False,
                "verdict": "NOT_READY",
                "target_task": task,
                "target_peft": peft,
                "target_model": model_name,
                "n_pairs": 0,
                "requirements": {"min_viable": min_viable, "target": target, "saturate": saturate},
                "axes": {},
                "blocking_issues": [
                    {"category": "volume_sufficiency", "severity": "high",
                     "detail": "No Q&A pairs to evaluate.",
                     "action": "Run generate_qa before calling readiness."}
                ],
                "warnings": [],
                "recommendations": ["Generate Q&A pairs first."],
            }

        axes = {
            "volume_sufficiency": self._score_volume(len(pairs), min_viable, target, saturate),
            "diversity":          self._score_diversity(pairs),
            "task_alignment":     self._score_task_alignment_llm(pairs, task, sample_size),
        }
        score = sum(axes.values()) / len(axes)

        blocking: list[dict] = []
        warnings_: list[str] = []

        if len(pairs) < min_viable:
            if req:
                detail = (
                    f"Only {len(pairs)} pairs — {model_name} ({req['size_b']}B, {peft}) "
                    f"needs ≥{min_viable} minimum pairs (target: {target})."
                )
                action = f"Run auto_fill_qa to reach the recommended target of {target} pairs."
            else:
                detail = f"Only {len(pairs)} pairs — {peft} requires ≥{min_viable}."
                action = "Generate more Q&A pairs."
            blocking.append({
                "category": "volume_sufficiency",
                "severity": "high",
                "detail": detail,
                "action": action,
            })
        elif len(pairs) < target:
            hint = f" (target for {model_name}: {target})" if model_name else ""
            warnings_.append(
                f"Dataset has {len(pairs)} pairs{hint}. "
                "Consider running auto_fill_qa to improve model performance."
            )

        if score >= self._READY_HIGH and not blocking:
            verdict, ready = "READY", True
        elif score >= self._READY_MED and not blocking:
            verdict, ready = "READY_WITH_WARNINGS", True
        else:
            verdict, ready = "NOT_READY", False

        return {
            "readiness_score": round(score, 3),
            "ready_for_finetuning": ready,
            "verdict": verdict,
            "target_task": task,
            "target_peft": peft,
            "target_model": model_name,
            "n_pairs": len(pairs),
            "requirements": {"min_viable": min_viable, "target": target, "saturate": saturate},
            "axes": {k: round(v, 3) for k, v in axes.items()},
            "blocking_issues": blocking,
            "warnings": warnings_,
            "recommendations": self._recommendations(axes, task, len(pairs), target, model_name),
        }

    # ── Axes ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_volume(n: int, min_viable: int, target: int, saturate: int) -> float:
        if n >= saturate:    return 1.0
        if n >= target:      return 0.85
        if n >= min_viable:  return 0.70
        return max(0.0, n / max(min_viable, 1) * 0.70)

    @staticmethod
    def _score_diversity(pairs: list[QAPair]) -> float:
        if not pairs:
            return 0.0
        sources = {p.source_file for p in pairs}
        source_score = min(1.0, len(sources) / 3.0)
        avg_a_len = sum(len(p.answer.split()) for p in pairs) / len(pairs)
        len_score = min(1.0, avg_a_len / 80.0)
        return source_score * 0.4 + len_score * 0.6

    def _score_task_alignment_llm(
        self, pairs: list[QAPair], task: str, sample_size: int
    ) -> float:
        import random
        sample = random.sample(pairs, min(sample_size, len(pairs)))
        numbered = "\n".join(
            f"{i+1}. Q: {p.question}\n   A: {p.answer[:200]}" for i, p in enumerate(sample)
        )
        prompt = (
            f"Task: {task}\n\n"
            f"Rate how well these {len(sample)} Q&A pairs align with the task. "
            "Return ONLY a single float 0.0-1.0.\n\nPAIRS:\n" + numbered
        )
        try:
            raw = self.llm.complete(
                system="You are a dataset quality evaluator.", user=prompt
            )
            import re
            m = re.search(r"\b(0?\.\d+|1\.0|0|1)\b", raw)
            return float(m.group(0)) if m else 0.7
        except Exception:
            return 0.7

    def _recommendations(
        self, axes: dict, task: str, n: int, target: int, model_name: str
    ) -> list[str]:
        recs: list[str] = []
        if axes.get("volume_sufficiency", 1.0) < 0.7:
            model_hint = f" for {model_name}" if model_name else ""
            recs.append(
                f"Increase dataset to at least {target} pairs{model_hint} — "
                "run auto_fill_qa with conceptual/applied/analytical styles."
            )
        if axes.get("diversity", 1.0) < 0.6:
            recs.append("Improve answer diversity — many answers have similar length or content.")
        if axes.get("task_alignment", 1.0) < 0.6:
            recs.append(f"Review Q&A pairs for {task} alignment — some may not match the target format.")
        if not recs:
            recs.append("Dataset looks ready. Review blocking issues if any before training.")
        return recs
