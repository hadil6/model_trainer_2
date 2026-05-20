"""Domain- and task-aware metric thresholds for fine-tuning evaluation.

All thresholds and eval strategy are determined exclusively by the LLM.
No hardcoded values, no static tables, no fallbacks.

Public API
----------
get_thresholds(task, domain, llm_client) -> MetricThresholds
detect_eval_strategy(task, domain, user_goal, llm_client) -> EvalStrategy
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricThresholds:
    rouge1_good:       float   # at or above → "good" quality
    rouge1_acceptable: float   # at or above → "acceptable" quality (minimum floor)
    rouge2_good:       float
    rougeL_good:       float
    bleu4_good:        float
    bleu4_acceptable:  float
    rationale:         str
    rouge1_suspicious: float = 0.88

    def quality_tier(self, rouge1: float, bleu4: float) -> str:
        if rouge1 >= self.rouge1_good and bleu4 >= self.bleu4_good:
            return "good"
        if rouge1 >= self.rouge1_acceptable:
            return "acceptable"
        return "poor"

    def is_suspicious(self, rouge1: float) -> bool:
        return rouge1 > self.rouge1_suspicious

    def quality_label(self, rouge1: float, bleu4: float) -> str:
        tier = self.quality_tier(rouge1, bleu4)
        if tier == "good":
            return "Bonne qualité"
        if tier == "acceptable":
            return "Qualité acceptable"
        return "Qualité insuffisante"

    def to_dict(self) -> dict:
        return {
            "rouge1_good":        self.rouge1_good,
            "rouge1_acceptable":  self.rouge1_acceptable,
            "rouge1_suspicious":  self.rouge1_suspicious,
            "rouge2_good":        self.rouge2_good,
            "rougeL_good":        self.rougeL_good,
            "bleu4_good":         self.bleu4_good,
            "bleu4_acceptable":   self.bleu4_acceptable,
            "rationale":          self.rationale,
        }


# ── LLM threshold generation ──────────────────────────────────────────────────

_LLM_SYSTEM = (
    "You are an NLP evaluation expert. "
    "You set precise, evidence-based metric thresholds for fine-tuned language model evaluation."
)

_LLM_PROMPT = """\
A language model has been fine-tuned for the following use case:
  Task  : {task}
  Domain: {domain}

Set appropriate evaluation thresholds for ROUGE and BLEU metrics.
Consider:
- How much paraphrase is acceptable in this domain (legal/medical → low, conversation → high)?
- How dense and precise are the reference answers?
- What do domain-specific NLP benchmarks use as quality bars?

Return ONLY valid JSON with exactly these keys:
{{
  "rouge1_good":       <float 0.3–0.7>,
  "rouge1_acceptable": <float 0.15–0.5>,
  "rouge2_good":       <float 0.1–0.5>,
  "rougeL_good":       <float 0.25–0.65>,
  "bleu4_good":        <float 0.05–0.35>,
  "bleu4_acceptable":  <float 0.02–0.15>,
  "rationale":         "<one sentence>"
}}"""


def _llm_thresholds(task: str, domain: str, llm_client) -> MetricThresholds | None:
    try:
        raw = llm_client.complete(
            system=_LLM_SYSTEM,
            user=_LLM_PROMPT.format(task=task, domain=domain),
        )
        m = re.search(r"\{[\s\S]*?\}", raw)
        if not m:
            return None
        d = json.loads(m.group(0))
        return MetricThresholds(
            rouge1_good       = float(d["rouge1_good"]),
            rouge1_acceptable = float(d["rouge1_acceptable"]),
            rouge2_good       = float(d["rouge2_good"]),
            rougeL_good       = float(d["rougeL_good"]),
            bleu4_good        = float(d["bleu4_good"]),
            bleu4_acceptable  = float(d["bleu4_acceptable"]),
            rationale         = str(d.get("rationale", "")),
        )
    except Exception as exc:
        logger.warning("metric_thresholds_llm_failed: %s", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_thresholds(
    task: str,
    domain: str,
    llm_client,
) -> MetricThresholds:
    """Return LLM-generated metric thresholds for this task+domain combination.

    Raises RuntimeError if llm_client is None or the LLM call fails.
    """
    if llm_client is None:
        raise RuntimeError(
            "get_thresholds: llm_client is required — no hardcoded fallback available"
        )

    thresh = _llm_thresholds(task, domain, llm_client)
    if thresh is None:
        raise RuntimeError(
            f"get_thresholds: LLM failed to generate thresholds for task={task!r} domain={domain!r}"
        )

    logger.info(
        "metric_thresholds domain=%s task=%s source=llm rouge1_good=%.2f",
        domain, task, thresh.rouge1_good,
    )
    return thresh


# ══════════════════════════════════════════════════════════════════════════════
# EvalStrategy — task-aware metric selection
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalStrategy:
    """Which metrics to compute and which one drives the improvement loop."""
    strategy:             str         # "generation" | "classification" | "ner" | "hybrid"
    metrics:              list[str]
    primary_metric:       str
    good_threshold:       float
    acceptable_threshold: float
    rationale:            str

    def to_dict(self) -> dict:
        return {
            "strategy":             self.strategy,
            "metrics":              self.metrics,
            "primary_metric":       self.primary_metric,
            "good_threshold":       self.good_threshold,
            "acceptable_threshold": self.acceptable_threshold,
            "rationale":            self.rationale,
        }


# ── LLM strategy detection ────────────────────────────────────────────────────

_STRATEGY_SYSTEM = (
    "You are an ML evaluation expert. "
    "Given a fine-tuning task and user objective, determine which evaluation "
    "strategy and metrics are most appropriate."
)

_STRATEGY_PROMPT = """\
A language model is being fine-tuned and must be evaluated on a held-out test set.

  Task      : {task}
  Domain    : {domain}
  User goal : {user_goal}

=== EVALUATION SETUP — facts you MUST reason from ===
- Each test example has exactly ONE reference answer (no multi-reference).
  Single-reference scoring penalizes every valid paraphrase, so lexical metrics
  score LOWER than people intuitively expect.
- For generation tasks the fine-tuned model writes free-form (abstractive)
  answers — it does not copy spans from a source.
- Reference answers are natural-language text, not category labels.
- {bertscore_line}

=== METRIC KNOWLEDGE BASE — realistic ranges to ground your thresholds ===
ROUGE-1 / ROUGE-L (Lin 2004): lexical n-gram overlap vs ONE reference; penalizes
  paraphrase. Abstractive-summarization SOTA (CNN/DailyMail) reaches ROUGE-1
  ~0.40-0.44; free-form QA vs a single reference typically lands 0.20-0.40.
BLEU-4 (Papineni 2002): n-gram PRECISION, designed for machine translation; very
  strict on free-form text — generative QA scores only 0.03-0.15. Meaningful as a
  primary metric ONLY for translation.
BERTScore-F1 (Zhang 2020): cosine of contextual BERT embeddings — measures MEANING
  and tolerates paraphrase. It has an inflated floor: two unrelated English
  sentences still score ~0.80-0.85, so useful discrimination lives in 0.83-0.95.
F1-token (SQuAD, Rajpurkar 2016): token-set overlap F1, order-independent, more
  forgiving than ROUGE; generative QA typically 0.30-0.55.
G-Eval (Liu 2023) and Faithfulness: LLM-judged quality, range ~0.5-0.9 — computed
  ONLY on the first evaluation, so they are NOT eligible as primary_metric.
F1-macro: classification correctness, balanced over classes, range 0-1.
Entity-F1: named-entity extraction F1, range 0-1.

=== YOUR DECISION ===
1. strategy: choose "generation" | "classification" | "ner" | "hybrid".
2. metrics: the list of metrics to report.
3. primary_metric: the SINGLE metric that decides good/acceptable/poor. It MUST be
   computed on EVERY evaluation — allowed values:
   rouge1, rouge2, rougeL, bleu4, f1_token{bertscore_primary_opt} (generation),
   f1_macro (classification), entity_f1 (ner).
   Prefer a MEANING-based metric for conversational/QA tasks (chatbot, question
   answering); use bleu4 only for translation.
4. good_threshold and acceptable_threshold FOR THE PRIMARY METRIC. Reason
   EXPLICITLY from the KNOWLEDGE BASE and the EVALUATION SETUP above:
   - the thresholds must be on the SCALE of the metric you chose
     (a BERTScore bar is far higher than a ROUGE bar — they are not interchangeable);
   - account for single-reference scoring and abstractive output;
   - adjust for domain: precise domains (legal, medical) tolerate less paraphrase
     → slightly higher bars; conversational/general domains → slightly lower;
   - good_threshold must be strictly greater than acceptable_threshold; both in [0,1].

Reply ONLY with valid JSON:
{{
  "strategy":             "...",
  "metrics":              ["...", "..."],
  "primary_metric":       "...",
  "good_threshold":       <float on the primary metric's own scale>,
  "acceptable_threshold": <float on the primary metric's own scale>,
  "rationale":            "name the primary metric, its nature, and the benchmark range you based the thresholds on"
}}"""


def _bertscore_available() -> bool:
    """True if the bert-score package is importable (BERTScore can be computed)."""
    import importlib.util
    return importlib.util.find_spec("bert_score") is not None


def _llm_strategy(task: str, domain: str, user_goal: str, llm_client) -> EvalStrategy | None:
    # Tell the LLM whether bertscore_f1 is a usable primary metric on this machine.
    if _bertscore_available():
        bertscore_line = (
            "BERTScore IS installed — bertscore_f1 is computed on every evaluation "
            "and IS eligible as primary_metric (preferred for semantic/conversational tasks)."
        )
        bertscore_primary_opt = ", bertscore_f1"
    else:
        bertscore_line = (
            "BERTScore is NOT installed — bertscore_f1 will not be computed; "
            "do NOT choose it as primary_metric (pick f1_token or rougeL instead)."
        )
        bertscore_primary_opt = ""
    try:
        raw = llm_client.complete(
            system=_STRATEGY_SYSTEM,
            user=_STRATEGY_PROMPT.format(
                task=task, domain=domain, user_goal=user_goal,
                bertscore_line=bertscore_line,
                bertscore_primary_opt=bertscore_primary_opt,
            ),
        )
        m = re.search(r"\{[\s\S]*?\}", raw)
        if not m:
            return None
        d = json.loads(m.group(0))
        # The LLM decides the thresholds. The only validation here is MATHEMATICAL:
        # ROUGE/BLEU/BERTScore/F1 are all defined on [0, 1] — a value outside that
        # range is impossible, not a quality choice. No quality bar is hardcoded.
        good       = min(1.0, max(0.0, float(d["good_threshold"])))
        acceptable = min(1.0, max(0.0, float(d["acceptable_threshold"])))
        if acceptable >= good:
            logger.warning(
                "eval_strategy: LLM returned acceptable (%.3f) >= good (%.3f) — "
                "thresholds inconsistent, keeping LLM values as-is",
                acceptable, good,
            )
        return EvalStrategy(
            strategy             = str(d["strategy"]),
            metrics              = list(d["metrics"]),
            primary_metric       = str(d["primary_metric"]),
            good_threshold       = good,
            acceptable_threshold = acceptable,
            rationale            = str(d.get("rationale", "")),
        )
    except Exception as exc:
        logger.warning("eval_strategy_llm_failed: %s", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def detect_eval_strategy(
    task: str,
    domain: str,
    user_goal: str,
    llm_client,
) -> EvalStrategy:
    """Detect the best evaluation strategy via LLM.

    Raises RuntimeError if llm_client is None or the LLM call fails.
    """
    if llm_client is None:
        raise RuntimeError(
            "detect_eval_strategy: llm_client is required — no hardcoded fallback available"
        )

    strat = _llm_strategy(task, domain, user_goal, llm_client)
    if strat is None:
        raise RuntimeError(
            f"detect_eval_strategy: LLM failed to determine strategy for task={task!r}"
        )

    logger.info(
        "eval_strategy task=%s source=llm strategy=%s primary=%s",
        task, strat.strategy, strat.primary_metric,
    )
    return strat
