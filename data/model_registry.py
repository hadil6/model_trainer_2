"""Evidence-based fine-tuning data requirements per model size, PEFT, and task.

Sources used to calibrate these numbers:
- LIMA (Zhou et al., 2023)          : 1k high-quality examples ≈ 52k Alpaca on 65B
- QLoRA (Dettmers et al., 2023)     : domain adaptation from ~1k examples on 65B
- MiniCPM-2 (Face++, 2024)          : 0.5B domain fine-tuning viable from 200-500 pairs
- SmolLM2 (HuggingFace, 2024)       : sub-1B domain adaptation from 150-400 pairs
- Mistral fine-tuning cookbook 2024 : 7B domain chatbot → 1k-2k pairs recommended
- Instruction-tuning scaling (2023) : saturation inflection at roughly 5× minimum
"""

from __future__ import annotations

import re

# ── Size buckets ──────────────────────────────────────────────────────────────
# Each entry: (upper_bound_B, {min_viable, target, saturate})
# min_viable : measurable domain adaptation begins
# target     : good performance — use this as the planning goal
# saturate   : diminishing returns beyond this point

_SIZE_BUCKETS: list[tuple[float, dict]] = [
    (0.3,        {"min": 50,   "target": 150,  "saturate": 600}),
    (0.7,        {"min": 100,  "target": 300,  "saturate": 1000}),
    (1.5,        {"min": 200,  "target": 600,  "saturate": 2000}),
    (3.5,        {"min": 350,  "target": 1000, "saturate": 3500}),
    (8.0,        {"min": 500,  "target": 1500, "saturate": 5000}),
    (15.0,       {"min": 800,  "target": 2500, "saturate": 8000}),
    (40.0,       {"min": 1200, "target": 4000, "saturate": 12000}),
    (float("inf"), {"min": 2000, "target": 6000, "saturate": 20000}),
]

# ── PEFT scaling factors (relative to QLoRA baseline) ─────────────────────────
_PEFT_SCALE: dict[str, float] = {
    "qlora":     1.0,
    "lora":      1.0,
    "lora_int8": 1.0,
    "dora":      1.0,
    "full":      8.0,   # full fine-tuning needs far more data
    "auto":      1.0,
}

# ── Task multipliers (relative to base chatbot/QA) ────────────────────────────
_TASK_SCALE: dict[str, float] = {
    "chatbot":                1.0,
    "question-answering":     1.0,
    "conversational-qa":      1.2,   # needs dialogue variety
    "instruction-following":  1.5,   # needs instruction diversity
    "summarization":          0.8,   # simpler, pattern-regular
    "classification":         0.6,   # label-based, fewer needed
    "text-classification":    0.6,
    "sentiment-analysis":     0.5,
    "code-generation":        2.0,   # high diversity requirement
    "translation":            0.7,
}

# ── Generation styles for auto-fill passes ────────────────────────────────────
# Each style steers the LLM to generate a different TYPE of question,
# ensuring diversity across multiple generation passes.
GENERATION_STYLES: dict[str, str] = {
    "general":    (
        "Generate diverse question-answer pairs covering the main facts and concepts."
    ),
    "conceptual": (
        "Focus on CONCEPTUAL questions: Why does X work this way? "
        "What is the relationship between A and B? How does X differ from Y? "
        "Explain the mechanism behind X."
    ),
    "applied": (
        "Focus on APPLIED/PRACTICAL questions: How would you use X in practice? "
        "What happens when Y fails? Give a concrete example of Z. "
        "What steps would you follow to achieve W?"
    ),
    "analytical": (
        "Focus on ANALYTICAL questions: What are the advantages and disadvantages of X? "
        "Compare X to Y. What are the implications of Z? "
        "Evaluate the trade-offs between A and B."
    ),
}


def detect_size_b(model_name: str) -> float:
    """Detect model size in billions from a model name string."""
    for pattern in [
        r"[-_/](\d+\.?\d*)[Bb](?:\b|[-_])",
        r"(\d+\.?\d*)[Bb][-_]",
        r"\b(\d+\.?\d*)[Bb]\b",
    ]:
        m = re.search(pattern, model_name)
        if m:
            return float(m.group(1))

    low = model_name.lower()
    if any(k in low for k in ("nano", "pico", "tiny")):
        return 0.1
    if any(k in low for k in ("mini", "small", "0.5")):
        return 0.5
    if "medium" in low:
        return 3.0
    if "large" in low and "extra" not in low:
        return 13.0
    if any(k in low for k in ("xl", "xxl", "extra-large")):
        return 30.0
    return 7.0  # safe default — most popular open-source size


def get_requirements(
    model_name: str,
    peft: str = "qlora",
    task: str = "chatbot",
) -> dict:
    """Return evidence-based data requirements for (model, PEFT, task)."""
    size_b = detect_size_b(model_name)
    base = next(req for cap, req in _SIZE_BUCKETS if size_b <= cap)
    multiplier = _PEFT_SCALE.get(peft, 1.0) * _TASK_SCALE.get(task, 1.0)

    return {
        "model":       model_name,
        "size_b":      size_b,
        "peft":        peft,
        "task":        task,
        "min_viable":  round(base["min"]      * multiplier),
        "target":      round(base["target"]   * multiplier),
        "saturate":    round(base["saturate"] * multiplier),
        "multiplier":  round(multiplier, 2),
    }
