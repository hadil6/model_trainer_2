"""User intent extraction — one LLM call to parse user's goal into structured fields."""

from __future__ import annotations

import json
import logging
import re

from agents.llm_client import LLMClient

logger = logging.getLogger(__name__)


class UserIntent:
    def __init__(
        self,
        task: str,
        domain: str,
        language: str,
        tone: str,
        description: str,
    ) -> None:
        self.task = task
        self.domain = domain
        self.language = language
        self.tone = tone
        self.description = description

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "domain": self.domain,
            "language": self.language,
            "tone": self.tone,
            "description": self.description,
        }


_VALID_TASKS = {
    "question-answering", "conversational-qa", "chatbot",
    "summarization", "instruction-following", "classification",
    "text-classification", "code-generation", "translation",
}

_VALID_TONES = {"academic", "conversational", "technical", "simplified"}


def extract_intent(user_goal: str, llm_client: LLMClient) -> UserIntent:
    """Parse a free-text user goal into structured intent fields."""
    raw = llm_client.complete(
        system=(
            "You are a dataset preparation assistant. Extract structured intent from a user's goal.\n"
            f"Valid tasks: {', '.join(sorted(_VALID_TASKS))}\n"
            f"Valid tones: {', '.join(sorted(_VALID_TONES))}\n"
            "Return ONLY JSON with keys: task, domain, language (ISO 639-1), tone, description."
        ),
        user=f"User goal: {user_goal}",
    )
    try:
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            data = json.loads(m.group(0))
            task = data.get("task", "question-answering")
            if task not in _VALID_TASKS:
                task = "question-answering"
            tone = data.get("tone", "conversational")
            if tone not in _VALID_TONES:
                tone = "conversational"
            return UserIntent(
                task=task,
                domain="auto",  # detected from documents, not from user goal
                language=data.get("language", "en"),
                tone=tone,
                description=data.get("description", user_goal),
            )
    except Exception as exc:
        logger.warning("intent_parse_failed: %s", exc)

    # Fallback: minimal heuristic
    return _heuristic_intent(user_goal)


def _heuristic_intent(goal: str) -> UserIntent:
    text = goal.lower()
    if any(k in text for k in ("chat", "conv", "dialogue")):
        task = "conversational-qa"
    elif any(k in text for k in ("summar", "resume", "tl;dr")):
        task = "summarization"
    elif any(k in text for k in ("classif", "label", "sentiment", "categor")):
        task = "classification"
    elif any(k in text for k in ("code", "program")):
        task = "code-generation"
    else:
        task = "question-answering"

    return UserIntent(
        task=task,
        domain="auto",  # detected from documents, not from user goal
        language="en",
        tone="conversational",
        description=goal,
    )
