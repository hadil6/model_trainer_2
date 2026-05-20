"""LLM-based PEFT method selection from hardware + model analysis.

The LLM receives: VRAM, model size, dataset size, task — and decides which
PEFT method to use, with a rationale. Falls back to rule-based logic if the
LLM call fails.

Public API
----------
recommend_peft(model_name, vram_gb, n_pairs, task, llm_client) -> dict
"""

from __future__ import annotations

import json
import logging
import re

from agents.llm_client import LLMClient
from data.model_registry import detect_size_b

logger = logging.getLogger(__name__)

def _get_valid_methods() -> set[str]:
    """Return the set of valid PEFT method names, queried dynamically from SWIFT."""
    from data.model_zoo.model_cards import get_peft_method_map
    return set(get_peft_method_map().keys())

_SYSTEM = (
    "You are an expert in LLM fine-tuning infrastructure. "
    "Your job is to choose the safest and most effective PEFT method "
    "given hardware constraints, model characteristics, dataset size, and user objective."
)

_PROMPT = """\
Analyze the following setup and recommend the optimal PEFT fine-tuning method.

=== Setup ===
Model:        {model_name}
Model size:   {size_b}B parameters
GPU VRAM:     {vram_gb} GB
Dataset:      {n_pairs} Q&A pairs  (~{n_words} words)
Task:         {task}
Objective:    {objective}

=== Available methods (with estimated minimum VRAM for this model) ===
{methods_section}

=== Selection rules ===
- Choose from the available methods above ONLY.
- The method's estimated VRAM must be ≤ {vram_gb} GB.
- speed objective      → prefer the lowest-VRAM method.
- balanced objective   → prefer lora_int8 or qlora.
- performance objective → prefer lora or full if VRAM allows.

Return ONLY valid JSON:
{{
  "peft_method": "<one of the method names above>",
  "rationale":   "one sentence explanation",
  "warning":     "optional — memory risk or quality note, else null"
}}"""


def recommend_peft(
    model_name: str,
    vram_gb: int,
    n_pairs: int,
    task: str,
    llm_client: LLMClient,
    objective: str = "balanced",
) -> dict:
    """LLM decides the PEFT method. Falls back to rule-based logic on failure."""
    from data.model_zoo.model_cards import get_peft_method_map, _estimate_min_vram
    size_b = detect_size_b(model_name)

    # Build method descriptions dynamically from the PEFT map + VRAM estimates
    method_map = get_peft_method_map()
    lines = []
    for method, cfg in method_map.items():
        est = _estimate_min_vram(size_b, method)
        quant = f" (quant {cfg['quant_bits']}-bit)" if "quant_bits" in cfg else ""
        fits = "✓ fits" if est <= vram_gb else "✗ too large"
        lines.append(f"- {method:<12}: tuner={cfg['tuner_type']}{quant}  min_vram={est}GB  [{fits}]")
    methods_section = "\n".join(lines)

    prompt = _PROMPT.format(
        model_name=model_name,
        size_b=size_b,
        vram_gb=vram_gb,
        n_pairs=n_pairs,
        n_words=n_pairs * 80,
        task=task,
        objective=objective,
        methods_section=methods_section,
    )

    try:
        raw = llm_client.complete(system=_SYSTEM, user=prompt)
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            result = json.loads(m.group(0))
            method = result.get("peft_method", "")
            if method in _get_valid_methods():
                logger.info(
                    "auto_peft model=%s size=%.1fB vram=%dGB → %s | %s",
                    model_name, size_b, vram_gb, method, result.get("rationale", ""),
                )
                return {
                    "peft_method": method,
                    "rationale":   result.get("rationale", ""),
                    "warning":     result.get("warning"),
                    "source":      "llm",
                }
    except Exception as exc:
        logger.warning("auto_peft_llm_failed: %s — falling back to rules", exc)

    # ── Rule-based fallback ───────────────────────────────────────────────────
    return _rule_based(size_b, vram_gb, objective)


def _rule_based(size_b: float, vram_gb: int, objective: str = "balanced") -> dict:
    """Pick the best feasible method using physics-based VRAM estimates."""
    from data.model_zoo.model_cards import get_peft_method_map, _estimate_min_vram

    method_map = get_peft_method_map()
    feasible = [m for m in method_map if _estimate_min_vram(size_b, m) <= vram_gb]

    if not feasible:
        logger.warning("auto_peft: no method fits %.1fB in %dGB — forcing qlora", size_b, vram_gb)
        method, reason = "qlora", "no method fits — qlora is the most memory-efficient option"
    elif objective == "speed":
        # Prefer the method with lowest VRAM requirement
        method = min(feasible, key=lambda m: _estimate_min_vram(size_b, m))
        reason = f"speed objective — {method} has lowest VRAM footprint"
    elif objective == "performance":
        # Prefer lora > lora_int8 > qlora > others (higher precision first)
        priority = ["lora", "lora_int8", "qlora"]
        method = next((m for m in priority if m in feasible), feasible[-1])
        reason = f"performance objective — {method} offers best precision within {vram_gb}GB"
    else:
        # balanced: prefer lora_int8, fallback to qlora
        priority = ["lora_int8", "qlora", "lora"]
        method = next((m for m in priority if m in feasible), feasible[0])
        reason = f"balanced objective — {method} fits in {vram_gb}GB VRAM"

    logger.info("auto_peft rule-based → %s (%s)", method, reason)
    return {"peft_method": method, "rationale": reason, "warning": None, "source": "rules"}
