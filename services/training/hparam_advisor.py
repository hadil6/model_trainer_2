"""LLM-based hyperparameter recommendation for SWIFT fine-tuning.

The LLM receives:
  - model name + size
  - PEFT method chosen by auto_peft
  - dataset size + task
  - evidence-based baseline (from hparams.py) as context

It reasons about overfitting risk, VRAM limits, dataset size vs model capacity,
and returns specific values for each hyperparameter.

Falls back to hparams.get_hparams() if the LLM call fails or returns invalid JSON.

Public API
----------
recommend_hparams(model_name, peft_method, n_pairs, vram_gb, task, llm_client) -> dict
recommend_n_trials(model_name, peft_method, n_pairs, vram_gb, llm_client)       -> int
"""

from __future__ import annotations

import json
import logging
import re

from agents.llm_client import LLMClient
from data.model_registry import detect_size_b
from services.training.hparams import get_hparams

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert in LLM fine-tuning. "
    "You recommend precise, well-justified hyperparameters for SWIFT fine-tuning jobs."
)

_PROMPT = """\
Recommend optimal SWIFT fine-tuning hyperparameters for this job.

=== Context ===
Model:        {model_name}
Model size:   {size_b}B parameters
PEFT method:  {peft_method}
Dataset:      {n_pairs} Q&A pairs
Task:         {task}
GPU VRAM:     {vram_gb} GB

=== Evidence-based baseline (from QLoRA paper, LIMA, Mistral cookbook) ===
{baseline_json}

=== What to reason about ===
- lora_rank: larger rank = more adapter capacity. Use higher rank for complex tasks or large models.
  Typical: 4 (<0.5B), 8 (0.5–1.5B), 16 (1.5–7B), 32 (7–15B), 64 (>15B).
- lora_alpha: controls effective scaling. Standard: alpha = 2 × rank.
- lora_dropout: 0.0 for large datasets (regularization not needed), up to 0.1 for very small datasets.
- learning_rate: 2e-4 for qlora (quantization adds noise that stabilizes gradients),
  1e-4 for bf16 lora, 2e-5 for full fine-tuning.
- num_train_epochs: small datasets (<500 pairs) need more passes (5–10 epochs, LIMA principle).
  Large datasets (>5000 pairs) need fewer (1–2 epochs).
- warmup_ratio: 0.05 is safe. Increase to 0.1 for unstable training (very small datasets).
- weight_decay: 0.01 standard. Increase to 0.05 if overfitting risk is high.

Return ONLY valid JSON with exactly these keys:
{{
  "lora_rank":        <int>,
  "lora_alpha":       <int>,
  "lora_dropout":     <float 0.0–0.15>,
  "learning_rate":    <float>,
  "num_train_epochs": <int>,
  "warmup_ratio":     <float 0.02–0.20>,
  "weight_decay":     <float 0.0–0.10>,
  "rationale":        "<one sentence>"
}}"""

# Keys we expect and their types for validation
_EXPECTED: dict[str, type] = {
    "lora_rank":        int,
    "lora_alpha":       int,
    "lora_dropout":     float,
    "learning_rate":    float,
    "num_train_epochs": int,
    "warmup_ratio":     float,
    "weight_decay":     float,
}


def recommend_hparams(
    model_name: str,
    peft_method: str,
    n_pairs: int,
    vram_gb: int,
    task: str,
    llm_client: LLMClient,
) -> dict:
    """LLM recommends hyperparameters. Falls back to static hparams.py on failure."""
    size_b   = detect_size_b(model_name)
    baseline = get_hparams(model_name, peft_method, n_pairs, vram_gb, task)
    baseline_public = {k: v for k, v in baseline.items() if not k.startswith("_")}

    prompt = _PROMPT.format(
        model_name=model_name,
        size_b=size_b,
        peft_method=peft_method,
        n_pairs=n_pairs,
        task=task,
        vram_gb=vram_gb,
        baseline_json=json.dumps(baseline_public, indent=2),
    )

    try:
        raw = llm_client.complete(system=_SYSTEM, user=prompt)
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            parsed = json.loads(m.group(0))
            validated = _validate(parsed, baseline)
            if validated:
                logger.info(
                    "hparam_advisor model=%s peft=%s n=%d → rank=%d lr=%s epochs=%d | %s",
                    model_name, peft_method, n_pairs,
                    validated["lora_rank"], validated["learning_rate"],
                    validated["num_train_epochs"],
                    parsed.get("rationale", ""),
                )
                validated["_source"]    = "llm"
                validated["_rationale"] = parsed.get("rationale", "")
                return validated
    except Exception as exc:
        logger.warning("hparam_advisor_llm_failed: %s — using static baseline", exc)

    baseline["_source"] = "static"
    return baseline


_TRIALS_SYSTEM = (
    "You are an expert in LLM fine-tuning and hyperparameter optimization. "
    "You determine the optimal number of Optuna HPO trials based on dataset size, "
    "model size, PEFT method, and hardware constraints."
)

_TRIALS_PROMPT = """\
Determine the optimal number of Optuna HPO trials for this fine-tuning job.
Each trial is a FULL training run with dynamic early stopping — trials stop
as soon as eval_loss rises above the best seen, so each trial may be shorter
than the full epoch count but still takes 10–50 minutes on a single GPU.

=== Context ===
Model:        {model_name}
Model size:   {size_b}B parameters
PEFT method:  {peft_method}
Dataset:      {n_pairs} Q&A pairs
GPU VRAM:     {vram_gb} GB
Objective:    {objective}
Early stopping: ALWAYS ACTIVE (stops at first eval_loss rise)

=== Rules ===
PRIORITY 1 — Objective:
- objective=speed:   minimize total training time → prefer 2–3 trials
- objective=balanced: balance quality vs time   → prefer 3–4 trials
- objective=quality: maximize model quality     → prefer 4–6 trials

PRIORITY 2 — Early stopping reduces Optuna information per trial:
- With early stopping, Optuna sees truncated loss curves → each trial
  gives LESS signal than a full run → more trials do NOT help much.
- Therefore: never recommend more than 4 trials when early stopping is active,
  unless objective=quality AND dataset > 2000 pairs.

PRIORITY 3 — Dataset size (secondary, early stopping dominates):
- < 300 pairs:   landscape is noisy → +1 trial
- 300–2000 pairs: moderate → no adjustment
- > 2000 pairs:  stable → -1 trial

PRIORITY 4 — Model size:
- > 7B parameters: each trial is very slow → cap at 3 trials maximum
- < 1B parameters: trials are fast → allow up to max for objective

=== Hard caps ===
- Minimum: 2 trials always
- Maximum: 6 trials (never exceed, regardless of dataset size)
- speed objective: maximum 3 trials
- balanced objective: maximum 4 trials

Return ONLY valid JSON:
{{"n_trials": <int 2–6>, "rationale": "<one sentence>"}}"""


def recommend_n_trials(
    model_name: str,
    peft_method: str,
    n_pairs: int,
    vram_gb: int,
    llm_client: LLMClient,
    objective: str = "balanced",
) -> int:
    """LLM recommends number of Optuna HPO trials. Falls back to 3 on failure.

    Factors considered:
      - objective: speed→2-3, balanced→3-4, quality→4-6
      - early stopping always active → reduces info per trial → fewer needed
      - n_pairs and model size are secondary adjustments
    """
    _MAX_BY_OBJECTIVE = {"speed": 3, "balanced": 4, "quality": 6}
    size_b = detect_size_b(model_name)
    prompt = _TRIALS_PROMPT.format(
        model_name=model_name,
        size_b=size_b,
        peft_method=peft_method,
        n_pairs=n_pairs,
        vram_gb=vram_gb,
        objective=objective,
    )
    try:
        raw = llm_client.complete(system=_TRIALS_SYSTEM, user=prompt)
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            d = json.loads(m.group(0))
            n = int(d["n_trials"])
            max_trials = _MAX_BY_OBJECTIVE.get(objective, 4)
            n = max(2, min(max_trials, n))
            logger.info(
                "recommend_n_trials model=%s n_pairs=%d vram=%dGB objective=%s → n_trials=%d | %s",
                model_name, n_pairs, vram_gb, objective, n, d.get("rationale", ""),
            )
            return n
    except Exception as exc:
        logger.warning("recommend_n_trials_failed: %s — using default 3", exc)
    return 3


def _validate(parsed: dict, fallback: dict) -> dict | None:
    """Return merged hparams if all expected keys are present and typed correctly."""
    result = dict(fallback)  # start with full baseline (includes batch_size, etc.)
    for key, typ in _EXPECTED.items():
        val = parsed.get(key)
        if val is None:
            logger.warning("hparam_advisor: missing key '%s' — using baseline", key)
            return None
        try:
            result[key] = typ(val)
        except (TypeError, ValueError):
            logger.warning("hparam_advisor: invalid type for '%s' — using baseline", key)
            return None
    return result
