"""ReAct Orchestrator — LLM pilots the entire pipeline via tool calls.

The LLM receives the job context and available tools, then autonomously:
  - Decides which tool to call at each step
  - Analyzes results and adapts
  - Diagnoses poor performance and takes corrective action
  - Concludes when satisfied or after exhausting improvement options

Graph
-----
  react_agent ⟺ tool_executor  (loop until LLM calls finish)
       └──────────────────────► finalize → END

Public API
----------
run_orchestrator(state: OrchestratorState) -> OrchestratorState
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from langgraph.graph import END, StateGraph

from agents.llm_client import get_llm_client
from agents.orchestrator.state import OrchestratorState
from data.artifact_store import ensure_job_dirs, write_json

logger = logging.getLogger(__name__)

HARD_MAX_ITERATIONS   = 10
HPO_ENABLED           = True  # set False to skip HPO and use run_training() directly


async def _notify(state: "OrchestratorState", msg: str) -> None:
    """Push a user-friendly status message to the SSE queue."""
    from services.run_manager import get_run_manager
    handle = get_run_manager().get(state["job_id"])
    if handle and hasattr(handle, "queue") and handle.queue is not None:
        await handle.queue.put(f"[PIPELINE] {msg}")


async def _emit_decision(
    state: "OrchestratorState",
    diagnosis: str,
    action: str,
    reason: str,
    changes: dict | None = None,
) -> None:
    """Push a structured 'decision' event to the SSE queue.

    The frontend uses these to render the 'Orchestrator Decisions' panel where
    the user can audit each corrective action: what was diagnosed, what action
    was taken, what specifically changed, and why.
    """
    from services.run_manager import get_run_manager
    handle = get_run_manager().get(state["job_id"])
    if handle and hasattr(handle, "queue") and handle.queue is not None:
        payload = json.dumps({
            "__decision__": True,
            "iteration":    state.get("improvement_iteration", 0),
            "diagnosis":    diagnosis,
            "action":       action,
            "changes":      changes or {},
            "reason":       reason,
        }, ensure_ascii=False)
        await handle.queue.put(payload)
STAGNATION_MIN_DELTA  = 0.02  # rouge1 must improve by at least this per iteration
STAGNATION_PATIENCE   = 2     # consecutive non-improving iterations → stagnation

# SWIFT CLI errors that will repeat identically regardless of hparam changes.
# Retrying train_model with the same model is pointless — must reselect or finish.
_FATAL_SWIFT_PATTERNS = [
    "remaining_argv",
    "unrecognized argument",
    "unrecognized arguments",
    "error: argument",
    "valueerror",
    "no such option",
]

def _is_fatal_swift_error(error_text: str) -> bool:
    """Return True if the SWIFT error is a CLI config error that would repeat on retry."""
    if not error_text:
        return False
    low = error_text.lower()
    return any(p in low for p in _FATAL_SWIFT_PATTERNS)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are a senior ML engineer and fine-tuning specialist with deep expertise in:
  - LLM fine-tuning (LoRA, QLoRA, full fine-tuning) and training dynamics
  - Diagnosing training failures: overfitting, underfitting, data quality issues,
    hyperparameter misconfiguration, model-task mismatch
  - Reading loss curves, metric trajectories, and dataset statistics to identify root causes
  - Selecting and adjusting hyperparameters based on observed evidence

Your mission: take uploaded documents + a user goal, produce a high-quality fine-tuned model.
You autonomously observe every tool result, reason about what the numbers tell you,
and call the most appropriate tool to fix the identified problem.
You do not follow a fixed script — you think like an expert who reads the evidence and acts.

Job context:
  Job ID                 : {job_id}
  Goal                   : {user_goal}
  Files                  : {filenames}
  GPU VRAM               : {gpu_vram_gb} GB
  Language               : {language}
  Domain                 : {domain}
  Objective              : {objective}
  Dataset size           : {dataset_size}
  Iterations done        : {improvement_iteration}
  Last {primary_metric}   : {last_rouge1}
  Stagnation detected    : {stagnating}
  Overfitting detected   : {overfitting}
  Underfitting detected  : {underfitting}
  Insufficient data      : {insufficient_data}
  Model mismatch         : {model_mismatch}
  Swift fatal error      : {swift_fatal_error}
  auto_fill_qa attempted : {auto_fill_attempted}
  Hard safety cap        : {hard_max} iterations (never exceed)

Pipeline order (first run):
  1. profile_files              — profile all files (always first)
  2. check_domain_compatibility — verify files share the same domain
  3. extract_intent             — clarify task type and domain
  4. detect_eval_strategy       — determine the right metrics for this task
  5. check_feasibility          — verify data is usable; stop if BLOCKED
  6. select_model               — choose best model for task, hardware, objective, dataset size
  7. prepare_data               — generate training pairs, deduplicate, export splits
  8. train_model                — fine-tune with SWIFT
  9. evaluate_model             — measure quality with task-appropriate metrics
  10. → improvement loop (reason → diagnose → act → repeat)
  11. finish                    — when quality is good OR impossible to improve

⚠ If check_domain_compatibility returns compatible=false → call finish(status="blocked") immediately.

=== Evaluation strategy (detected for this run) ===
  Strategy       : {eval_strategy}
  Primary metric : {primary_metric}
  All metrics    : {metrics_list}
  GOOD       : {primary_metric} >= {primary_good}        → finish(status="trained")
  ACCEPTABLE : {primary_metric} >= {primary_acceptable}  → finish(status="trained") only if stagnating
  POOR       : {primary_metric} <  {primary_acceptable}  → always diagnose and act
  Rationale  : {strategy_rationale}

=== MANDATORY CORRECTIVE ACTIONS (these BLOCK finish) ===

You are FORBIDDEN to call finish while ANY of these flags is True AND
improvement_iteration < {hard_max}. Each flag has a REQUIRED action:

  overfitting_detected = True
      → MUST call train_model(hparam_overrides=..., reasoning=...)
        Suggested fixes: lower num_train_epochs, increase lora_dropout,
        increase weight_decay, reduce learning_rate.
      → A model that overfits is unusable in production even if test
        metrics look acceptable — you MUST attempt at least one correction.

  underfitting_detected = True
      → MUST call train_model(hparam_overrides=..., reasoning=...)
        Suggested fixes: increase lora_rank, more num_train_epochs,
        higher learning_rate, reduce gradient_accumulation.

  insufficient_data = True  AND  auto_fill_qa_attempted = False
      → MUST call auto_fill_qa(target_pairs=..., reasoning=...)
        The dataset is too small for reliable metrics.

  model_mismatch = True
      → MUST call reselect_model(reasoning=...)
        Hyperparameter tuning failed across iterations — change architecture.

  swift_fatal_error = True
      → SWIFT rejected the CLI configuration (unsupported argument / ValueError).
        Retrying train_model with the same model produces the IDENTICAL crash — DO NOT call train_model again.
        Check how many models have been attempted (see "attempted_models" in tool history):
          • If only 1 model tried so far  → MUST call reselect_model(reasoning=...) to switch architecture.
          • If 2 or more models tried     → MUST call finish(status="failed") — hardware or config is incompatible.

=== STOP conditions (only when ALL warning flags are False) ===

You may call finish ONLY when overfitting_detected=False AND
underfitting_detected=False AND model_mismatch=False AND
(insufficient_data=False OR auto_fill_qa_attempted=True), AND:

  1. {primary_metric} >= {primary_good}                      → finish(status="trained")
  2. stagnating AND {primary_metric} >= {primary_acceptable} → finish(status="trained")
  3. stagnating AND {primary_metric} <  {primary_acceptable} → finish(status="trained_low_quality")
  4. improvement_iteration >= {hard_max}                     → finish(status="trained_low_quality")

=== YOUR DIAGNOSTIC REASONING — after every evaluate_model ===

You have full visibility over all tool results in your message history.
Read every signal available and reason like a specialist:

  Signals you can observe in tool results:
    train_loss, eval_loss         — training dynamics (divergence = overfitting)
    overfitting_detected          — computed flag: eval_loss diverging while train_loss falls
    underfitting_detected         — computed flag: both losses remain high
    insufficient_data             — computed flag: dataset too small for reliable training
    model_mismatch                — computed flag: hparam tuning failed across iterations
    {primary_metric}, quality_tier — model output quality on held-out test set
    n_pairs                       — dataset size
    hparam_overrides              — what was already tried
    model_id                      — which model was used
    improvement_log               — full history of every iteration

  Reason explicitly BEFORE calling any tool:
    → Which flags are currently True? Each True flag mandates a specific action.
    → What do the loss curves tell me? Are train_loss and eval_loss converging or diverging?
    → Is the primary metric improving, flat, or degrading across iterations?
    → What has already been tried (improvement_log)? What remains untried?

=== JUSTIFICATION OF EVERY CORRECTIVE ACTION (mandatory) ===

Every call to train_model, auto_fill_qa, or reselect_model MUST include
a 'reasoning' argument explaining:
  1. The diagnosis: what specifically is wrong (cite numbers — losses, metrics)
  2. The action: what you are changing
  3. The rationale: WHY these specific values address the problem

Example of a valid reasoning argument:
  "Overfitting detected: eval_loss=2.38 while train_loss=0.86 (gap>2.5×).
   Reducing num_train_epochs 8→4 (cut training time in half),
   increasing lora_dropout 0.05→0.15 (force regularization),
   and adding weight_decay 0.01→0.05 (penalize large weights).
   These changes target the divergence by limiting how much the model
   can memorize the training set."

A vague reasoning like 'tweaking hparams' is REJECTED.

=== improvement_log entries ===
  {{iteration, primary_metric_value, quality_tier, model_id,
    hparam_overrides, n_pairs, train_loss, eval_loss,
    overfitting_detected, action_taken}}

Rules:
  - Call ONE tool at a time.
  - Never call finish while any warning flag is True AND iteration < {hard_max}.
  - Every corrective action MUST include the 'reasoning' argument.
  - Never repeat an identical action already recorded in improvement_log.\
"""


def _detect_stagnation(improvement_log: list[dict]) -> bool:
    """True if last STAGNATION_PATIENCE iterations improved primary_metric_value by less than MIN_DELTA."""
    if len(improvement_log) < STAGNATION_PATIENCE:
        return False
    recent = improvement_log[-STAGNATION_PATIENCE:]
    for i in range(1, len(recent)):
        curr = recent[i].get("primary_metric_value", recent[i].get("rouge1", 0.0))
        prev = recent[i - 1].get("primary_metric_value", recent[i - 1].get("rouge1", 0.0))
        if curr - prev >= STAGNATION_MIN_DELTA:
            return False
    return True


def _detect_overfitting(improvement_log: list[dict]) -> bool:
    """Detect overfitting from the trend of train_loss vs eval_loss across iterations.

    Two signals (either is sufficient):
      1. Divergence trend — eval_loss went UP while train_loss went DOWN between
         the last two iterations (the curves are separating).
      2. Widening gap — the gap (eval_loss - train_loss) grew by more than 30%
         between the last two iterations.

    A single-entry log only triggers if the gap is extreme (eval_loss > train_loss * 2.5),
    i.e. the model was never even close on the validation set.
    """
    entries = [
        e for e in improvement_log
        if e.get("train_loss") is not None and e.get("eval_loss") is not None
    ]
    if not entries:
        return False

    last = entries[-1]
    t1, e1 = last["train_loss"], last["eval_loss"]

    if len(entries) == 1:
        # Only flag if the initial gap is extreme
        return e1 > t1 * 2.5

    prev = entries[-2]
    t0, e0 = prev["train_loss"], prev["eval_loss"]

    # Signal 1: curves diverging in opposite directions
    if e1 > e0 and t1 < t0:
        return True

    # Signal 2: gap grew by more than 30%
    gap0 = e0 - t0
    gap1 = e1 - t1
    if gap0 > 0 and gap1 > gap0 * 1.3:
        return True

    return False


def _detect_underfitting(improvement_log: list[dict]) -> bool:
    """Detect underfitting from training dynamics.

    Underfitting = the model failed to learn even the training data:
      - train_loss stays high (> 2.5) in the last iteration
      - AND eval_loss is also high (> 2.5)
    Indicates the model lacks capacity, lr is too low, or epochs too few.
    """
    entries = [
        e for e in improvement_log
        if e.get("train_loss") is not None and e.get("eval_loss") is not None
    ]
    if not entries:
        return False
    last = entries[-1]
    return last["train_loss"] > 2.5 and last["eval_loss"] > 2.5


def _detect_insufficient_data(
    state: OrchestratorState,
    improvement_log: list[dict],
    primary_acceptable: float,
) -> bool:
    """Detect that the dataset is too small for meaningful fine-tuning.

    Triggers when:
      - n_pairs < 50 (hard floor — never enough for any task), OR
      - n_pairs < 200 AND last primary_metric < acceptable threshold
        (model couldn't reach minimum quality with so few examples)
    """
    n_pairs = int((state.get("data_result") or {}).get("n_pairs", 0) or 0)
    if n_pairs == 0:
        # Fall back to target_n_pairs if data_result not populated yet
        return False
    if n_pairs < 50:
        return True
    if n_pairs < 200 and improvement_log:
        last = improvement_log[-1]
        primary_value = last.get("primary_metric_value", last.get("rouge1", 0))
        if primary_value < primary_acceptable:
            return True
    return False


def _detect_model_mismatch(
    improvement_log: list[dict],
    primary_acceptable: float,
) -> bool:
    """Detect when the chosen model architecture is inadequate for the task.

    Triggers when:
      - At least 2 training iterations have happened with DIFFERENT hparams
      - AND the latest primary_metric is still far below acceptable (< acceptable * 0.5)
    This means hyperparameter tuning is not the bottleneck — the model itself is.
    """
    if len(improvement_log) < 2:
        return False
    seen_hparams: set[str] = set()
    for e in improvement_log:
        h = e.get("hparam_overrides") or {}
        seen_hparams.add(json.dumps(h, sort_keys=True, default=str))
    if len(seen_hparams) < 2:
        return False
    last = improvement_log[-1]
    primary_value = last.get("primary_metric_value", last.get("rouge1", 0))
    return primary_value < primary_acceptable * 0.5


def _build_system_prompt(state: OrchestratorState) -> str:
    from data.artifact_store import domain_path
    improvement_log = state.get("improvement_log") or []
    stagnating      = _detect_stagnation(improvement_log)
    overfitting     = _detect_overfitting(improvement_log)
    underfitting    = _detect_underfitting(improvement_log)
    intent          = state.get("user_intent") or {}
    task            = intent.get("task", "question-answering")

    # Domain: priority → domain.json → intent → state → unknown
    domain = state.get("domain") or intent.get("domain") or ""
    try:
        _dp = domain_path(state["job_id"])
        if _dp.exists():
            detected = json.loads(_dp.read_text(encoding="utf-8")).get("domain", "")
            if detected:
                domain = detected
    except Exception:
        pass
    if not domain:
        domain = "unknown"

    ds = state.get("dataset_size") or {}
    ds_str = (
        f"{ds.get('total_words', '?')} mots / {ds.get('total_chars', '?')} chars / "
        f"{ds.get('file_count', '?')} fichier(s)"
    ) if ds else "non calculé"

    # Eval strategy: use detected strategy if available, else defaults
    eval_strat = state.get("eval_strategy") or {}
    primary_metric       = eval_strat.get("primary_metric", "rouge1")
    primary_good         = eval_strat.get("good_threshold", 0.45)
    primary_acceptable   = eval_strat.get("acceptable_threshold", 0.28)
    primary_low          = round(primary_acceptable * 0.5, 3)
    strategy_name        = eval_strat.get("strategy", "generation")
    metrics_list         = ", ".join(eval_strat.get("metrics", ["rouge1", "rouge2", "rougeL", "bleu4"]))
    strategy_rationale   = eval_strat.get("rationale", "Text generation evaluated with ROUGE.")

    # Insufficient data + model mismatch use the acceptable threshold for the check
    insufficient_data   = _detect_insufficient_data(state, improvement_log, primary_acceptable)
    model_mismatch      = _detect_model_mismatch(improvement_log, primary_acceptable)
    swift_fatal_error   = bool((state.get("training_result") or {}).get("swift_fatal_error"))
    auto_fill_attempted = bool(
        ((state.get("data_result") or {}).get("auto_fill")) or state.get("target_n_pairs")
    )

    last_primary = round(
        improvement_log[-1].get("primary_metric_value", improvement_log[-1].get("rouge1", 0.0))
        if improvement_log else 0.0,
        3,
    )

    return _SYSTEM_TEMPLATE.format(
        job_id=state["job_id"],
        user_goal=state.get("user_goal", ""),
        filenames=", ".join(state.get("filenames", [])),
        gpu_vram_gb=state.get("gpu_vram_gb", 4),
        language=state.get("language", "en"),
        domain=domain,
        objective=state.get("objective", "balanced"),
        dataset_size=ds_str,
        task=task,
        improvement_iteration=state.get("improvement_iteration", 0),
        hard_max=HARD_MAX_ITERATIONS,
        stagnating=stagnating,
        overfitting=overfitting,
        underfitting=underfitting,
        insufficient_data=insufficient_data,
        model_mismatch=model_mismatch,
        swift_fatal_error=swift_fatal_error,
        auto_fill_attempted=auto_fill_attempted,
        last_rouge1=last_primary,
        # strategy fields
        eval_strategy=strategy_name,
        primary_metric=primary_metric,
        primary_good=primary_good,
        primary_acceptable=primary_acceptable,
        primary_low=primary_low,
        metrics_list=metrics_list,
        strategy_rationale=strategy_rationale,
    )


# ── Tool definitions (OpenAI function-calling format) ─────────────────────────

PIPELINE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "profile_files",
            "description": (
                "Profile all uploaded files to understand their content, format, language, "
                "and size. Must be called first."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_domain_compatibility",
            "description": (
                "Verify that all uploaded files belong to the same domain. "
                "If files are from incompatible domains, the pipeline must be blocked. "
                "Call this after profile_files and before extract_intent."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_intent",
            "description": (
                "Extract the user's task intent (task type, domain, language) "
                "from their natural language goal description."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_eval_strategy",
            "description": (
                "Use an LLM to determine which evaluation metrics are appropriate for this task. "
                "Must be called after extract_intent and before check_feasibility. "
                "Returns the evaluation strategy (generation/classification/ner), "
                "the list of metrics to compute, the primary metric that drives the "
                "improvement loop, and the quality thresholds."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_feasibility",
            "description": (
                "Check if the uploaded data is suitable for fine-tuning. "
                "Returns status: GO, WARNING, or BLOCKED."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_model",
            "description": (
                "Select the best pre-trained model and PEFT method for the task "
                "using semantic search and LLM ranking."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_data",
            "description": (
                "Run the full data preparation pipeline: generate QA pairs from documents, "
                "deduplicate, evaluate quality, and export train/val/test splits."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "auto_fill_qa",
            "description": (
                "Generate additional QA pairs to increase the dataset size "
                "when the current dataset is too small. The user MUST confirm "
                "before the new pairs are appended."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_pairs": {
                        "type": "integer",
                        "description": "Target total number of QA pairs to reach",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "REQUIRED: explain WHY the dataset is insufficient "
                            "(cite n_pairs and primary_metric numbers), and how "
                            "adding pairs is expected to help."
                        ),
                    },
                },
                "required": ["target_pairs", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reselect_model",
            "description": (
                "Select a different pre-trained model when the current one performs poorly. "
                "Automatically excludes models already tried in this job."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "REQUIRED: explain WHY the current model is inadequate "
                            "(cite the failed iterations and the metrics that prove "
                            "tuning is not enough). State what kind of model you expect "
                            "to be better suited."
                        ),
                    },
                },
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_model",
            "description": (
                "Fine-tune the selected model using SWIFT. "
                "Optionally override specific hyperparameters. "
                "On any iteration > 0 the 'reasoning' argument is mandatory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hparam_overrides": {
                        "type": "object",
                        "description": (
                            "Hyperparameter overrides. Supported keys: "
                            "lora_rank, lora_alpha, num_train_epochs, "
                            "learning_rate, lora_dropout, weight_decay"
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "REQUIRED on any iteration > 0 OR when hparam_overrides "
                            "is provided. MUST explain: (1) the diagnosis based on "
                            "observed signals (cite train_loss/eval_loss/primary_metric "
                            "numbers), (2) what you are changing and from what value "
                            "to what value, (3) why those specific values are expected "
                            "to fix the problem. Example: 'Overfitting detected: "
                            "eval_loss=2.38 vs train=0.86. Reducing num_train_epochs "
                            "8→4 and increasing lora_dropout 0.05→0.15 to force "
                            "regularization.'"
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_model",
            "description": (
                "Evaluate the trained model on the held-out test set. "
                "Returns ROUGE-1, ROUGE-2, ROUGE-L, and BLEU-4 metrics."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Conclude the pipeline. Call this when quality is satisfactory "
                "or when no further improvement is possible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["trained", "trained_low_quality", "blocked", "error", "done"],
                        "description": "Final pipeline status",
                    },
                    "summary": {
                        "type": "string",
                        "description": "One sentence summary of what was accomplished",
                    },
                },
                "required": ["status", "summary"],
            },
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────────

async def _tool_profile_files(state: OrchestratorState) -> tuple[dict, dict]:
    from data.artifact_store import input_file_path, profile_path, raw_extraction_path, write_json
    from agents.data_agent.services.profiling import profile_file

    ensure_job_dirs(state["job_id"])
    profiles: list[dict] = []
    for filename in state["filenames"]:
        path = input_file_path(state["job_id"], filename)
        if not path.exists():
            continue
        pp = profile_path(state["job_id"], filename)
        if pp.exists():
            try:
                profiles.append(json.loads(pp.read_text(encoding="utf-8")))
                logger.info("profile_cache_hit orchestrator file=%s", filename)
                continue
            except Exception:
                pass
        try:
            result = profile_file(path)
            write_json(pp, result)
            if result.get("file_kind") in ("pdf", "txt") and result.get("full_text"):
                ep = raw_extraction_path(state["job_id"], filename)
                ep.parent.mkdir(parents=True, exist_ok=True)
                ep.write_text(result["full_text"], encoding="utf-8")
            profiles.append(result)
        except Exception as exc:
            logger.warning("profile_failed file=%s: %s", filename, exc)
            profiles.append({"filename": filename, "error": str(exc)})

    # Compute aggregated dataset size across all files
    total_words = 0
    total_chars = 0
    total_rows  = 0
    for p in profiles:
        meta = p.get("metadata") or {}
        total_words += meta.get("word_count", 0) or 0
        total_chars += meta.get("character_count", 0) or meta.get("char_count", 0) or 0
        total_rows  += meta.get("row_count", 0) or 0

    dataset_size = {
        "total_words": total_words,
        "total_chars": total_chars,
        "total_rows":  total_rows or None,
        "file_count":  len(profiles),
    }

    summary = [
        {
            "filename": p.get("filename"),
            "file_kind": p.get("file_kind"),
            "language": p.get("language"),
            "size_chars": (p.get("metadata") or {}).get("character_count") or (p.get("metadata") or {}).get("char_count"),
            "rows": (p.get("metadata") or {}).get("row_count"),
        }
        for p in profiles
    ]
    await _notify(state, (
        f"Profiling terminé — {len(profiles)} fichier(s) analysé(s) : "
        f"{total_words:,} mots, {len(profiles)} source(s)."
    ))
    return (
        {"profiles": summary, "count": len(profiles), "dataset_size": dataset_size},
        {"file_profiles": profiles, "dataset_size": dataset_size},
    )


async def _tool_check_domain_compatibility(state: OrchestratorState) -> tuple[dict, dict]:
    from services.run_manager import get_run_manager

    filenames = state.get("filenames") or []
    profiles  = state.get("file_profiles") or []

    if len(filenames) <= 1:
        logger.info("domain_compatibility: single file — skipping check")
        return {"compatible": True, "domain": "general", "incompatible_files": [], "reason": "Single file — no inter-file compatibility check needed"}, {}

    if not profiles:
        return {"compatible": True, "domain": "general", "incompatible_files": [], "reason": "No profiles available — assuming compatible"}, {}

    # Build a short summary for each file
    summaries: list[str] = []
    for p in profiles:
        filename = p.get("filename", "?")
        lang     = p.get("language", "?")
        size     = (p.get("metadata") or {}).get("char_count", 0) or (p.get("metadata") or {}).get("character_count", 0)
        snippets = p.get("samples") or {}
        raw_snippet = " ".join(filter(None, [snippets.get("head", ""), snippets.get("tail", "")]))
        sample   = (p.get("semantic_summary") or raw_snippet or p.get("full_text", ""))[:1500]
        summaries.append(f"Fichier: {filename}\nLangue: {lang}\nTaille: {size} caractères\nExtrait:\n{sample}")

    profiles_text = "\n\n---\n".join(summaries)

    result: dict = {"compatible": True, "domain": "general", "incompatible_files": [], "reason": ""}
    try:
        raw = get_llm_client().complete(
            system="You are an expert at classifying documents by domain (medical, legal, financial, technical, educational, etc.).",
            user=(
                f"Analyse ces {len(profiles)} fichiers et détermine s'ils appartiennent tous au même domaine.\n\n"
                f"{profiles_text}\n\n"
                "Réponds UNIQUEMENT en JSON:\n"
                '{"compatible": true/false, "domain": "domaine_détecté", '
                '"incompatible_files": ["nom_fichier", ...], "reason": "explication courte"}\n'
                "Si tous les fichiers partagent le même domaine → compatible=true, incompatible_files=[].\n"
                "Si des fichiers sont de domaines différents → compatible=false et liste les fichiers incompatibles."
            ),
        )
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            result = json.loads(m.group(0))
        else:
            result = {"compatible": True, "domain": "general", "incompatible_files": [], "reason": "Réponse LLM non parsable — compatibilité supposée"}
    except Exception as exc:
        logger.warning("check_domain_compatibility LLM failed: %s", exc)
        result = {"compatible": True, "domain": "general", "incompatible_files": [], "reason": f"Vérification échouée ({exc}) — compatibilité supposée"}

    compatible = bool(result.get("compatible", True))
    logger.info("domain_compatibility compatible=%s domain=%s files=%s", compatible, result.get("domain"), result.get("incompatible_files"))

    if compatible:
        await _notify(state, f"Compatibilité des domaines vérifiée — domaine détecté : {result.get('domain', 'inconnu')}.")
    else:
        job_id = state["job_id"]
        handle = get_run_manager().get(job_id)
        if handle and hasattr(handle, "queue") and handle.queue is not None:
            msg = (
                f"⚠ INCOMPATIBILITÉ DE DOMAINE : Les fichiers importés n'appartiennent pas au même domaine. "
                f"Fichiers incompatibles : {', '.join(result.get('incompatible_files', []))}. "
                f"Raison : {result.get('reason', '')}. "
                "Veuillez importer uniquement des documents du même domaine et relancer le pipeline."
            )
            await handle.queue.put(msg)

    return result, {}


async def _tool_extract_intent(state: OrchestratorState) -> tuple[dict, dict]:
    from agents.orchestrator.services.intent import extract_intent, _VALID_TASKS

    intent = extract_intent(state["user_goal"], get_llm_client())
    d = intent.to_dict()
    hint = state.get("task_hint") or ""
    if hint and hint in _VALID_TASKS:
        d["task"] = hint
    return d, {"user_intent": d}


async def _tool_detect_eval_strategy(state: OrchestratorState) -> tuple[dict, dict]:
    from services.metric_thresholds import detect_eval_strategy

    intent      = state.get("user_intent") or {}
    task        = intent.get("task", "question-answering")
    domain      = state.get("domain") or intent.get("domain") or "general"
    user_goal   = state.get("user_goal", "")

    strategy = detect_eval_strategy(
        task=task,
        domain=domain,
        user_goal=user_goal,
        llm_client=get_llm_client(),
    )
    d = strategy.to_dict()

    await _notify(state, (
        f"Stratégie d'évaluation détectée : {d['strategy']} — "
        f"Métrique principale : {d['primary_metric']} "
        f"(seuil minimum : {d['acceptable_threshold']} | objectif : {d['good_threshold']}). "
        f"Justification : {d['rationale']}"
    ))
    return d, {"eval_strategy": d}


async def _tool_check_feasibility(state: OrchestratorState) -> tuple[dict, dict]:
    from data.feasibility import evaluate_feasibility

    profiles = state.get("file_profiles") or []
    result   = evaluate_feasibility(profiles)

    # WARNING is non-blocking — the orchestrator LLM already receives the warnings
    # and can decide.  An extra LLM call here produced too many false-positive blocks
    # (e.g. OCR noise on an otherwise complete PDF extraction).

    # Never escalate WARNING → BLOCKED via LLM: the PDF was successfully extracted
    # even if some pages had OCR noise. The orchestrator LLM handles warnings itself.

    d = result.to_dict()
    verdict = d.get("status", "GO")
    if verdict == "BLOCKED":
        await _notify(state, f"Vérification de faisabilité : BLOQUÉ — {'; '.join(d.get('blocking_reasons', []))}")
    elif verdict == "WARNING":
        await _notify(state, f"Vérification de faisabilité : avertissements détectés — pipeline continue. ({'; '.join(d.get('warnings', [])[:2])})")
    else:
        await _notify(state, "Vérification de faisabilité : données utilisables — pipeline continue.")
    return d, {"feasibility": d}


async def _tool_select_model(state: OrchestratorState) -> tuple[dict, dict]:
    from services.model_selection import ModelSelector, SelectionInput

    intent   = state.get("user_intent") or {}
    ds       = state.get("dataset_size") or {}

    total_rows  = ds.get("total_rows") or None
    total_chars = ds.get("total_chars") or None

    # Domain: domain.json > intent > state (never fall back to "general")
    from data.artifact_store import domain_path
    domain = intent.get("domain") or state.get("domain") or ""
    try:
        _dp = domain_path(state["job_id"])
        if _dp.exists():
            detected = json.loads(_dp.read_text(encoding="utf-8")).get("domain", "")
            if detected:
                domain = detected
    except Exception:
        pass

    inp = SelectionInput(
        task=intent.get("task", "question-answering"),
        domain=domain or "unknown",
        description=intent.get("description", state["user_goal"]),
        data_language=intent.get("language", "en"),
        modality="text",
        total_rows=total_rows,
        total_chars=total_chars,
        gpu_vram_gb=state.get("gpu_vram_gb", 4),
        objective=state.get("objective", "balanced"),
    )
    try:
        selector = ModelSelector()
        result   = await asyncio.get_event_loop().run_in_executor(None, selector.select, inp)
        d = result.to_dict()
        already_tried = list(state.get("attempted_models") or [])
        if result.model_id not in already_tried:
            already_tried = already_tried + [result.model_id]
        await _notify(state, (
            f"Modèle sélectionné : {result.model_id} "
            f"(méthode : {result.peft_method}, objectif : {inp.objective}). "
            f"Raison : {result.reasoning}"
        ))
        return d, {
            "selection_result": d,
            "target_model":     result.model_id,
            "attempted_models": already_tried,
        }
    except Exception as exc:
        err = {"error": str(exc)}
        return err, {"selection_result": err}


async def _tool_reselect_model(state: OrchestratorState, args: dict | None = None) -> tuple[dict, dict]:
    from services.model_selection import ModelSelector, SelectionInput
    from data.artifact_store import domain_path

    args = args or {}
    intent        = state.get("user_intent") or {}
    ds            = state.get("dataset_size") or {}
    already_tried = list(state.get("attempted_models") or [])
    llm_reasoning = (args.get("reasoning") or "").strip()

    if not llm_reasoning:
        err = {
            "error": (
                "reselect_model requires the 'reasoning' argument. Explain WHY "
                "the current model is inadequate and what type of model you expect "
                "to perform better."
            )
        }
        return err, {}

    total_rows  = ds.get("total_rows") or None
    total_chars = ds.get("total_chars") or None

    # Domain: domain.json > intent > state (same priority as _tool_select_model)
    domain = intent.get("domain") or state.get("domain") or ""
    try:
        _dp = domain_path(state["job_id"])
        if _dp.exists():
            detected = json.loads(_dp.read_text(encoding="utf-8")).get("domain", "")
            if detected:
                domain = detected
    except Exception:
        pass

    # ── Notify user: structured decision event + status message ──────────────
    iteration       = state.get("improvement_iteration", 0)
    current_model   = (state.get("selection_result") or {}).get("model_id") or state.get("target_model") or "inconnu"
    improvement_log = state.get("improvement_log") or []
    last_rouge1     = improvement_log[-1].get("rouge1", 0) if improvement_log else 0

    await _emit_decision(
        state,
        diagnosis=(
            f"Modèle '{current_model}' inadéquat (dernière métrique = {last_rouge1:.3f})"
        ),
        action="reselect_model",
        reason=llm_reasoning,
        changes={
            "current_model": current_model,
            "excluded_models": ", ".join(already_tried) or "aucun",
        },
    )

    await _notify(state, (
        f"Décision itération {iteration} — Changement de modèle. "
        f"Modèle actuel '{current_model}' jugé inadéquat. "
        f"Justification : {llm_reasoning[:300]}. "
        f"Modèles exclus : {', '.join(already_tried) if already_tried else 'aucun'}."
    ))

    inp = SelectionInput(
        task=intent.get("task", "question-answering"),
        domain=domain or "unknown",
        description=intent.get("description", state["user_goal"]),
        data_language=intent.get("language", "en"),
        modality="text",
        total_rows=total_rows,
        total_chars=total_chars,
        gpu_vram_gb=state.get("gpu_vram_gb", 4),
        objective=state.get("objective", "balanced"),
        exclude_models=already_tried,
    )
    try:
        selector = ModelSelector()
        result   = await asyncio.get_event_loop().run_in_executor(None, selector.select, inp)
        d = result.to_dict()
        new_tried = already_tried + ([result.model_id] if result.model_id else [])
        logger.info("reselect_model excluded=%s → %s", already_tried, result.model_id)
        await _notify(state, (
            f"Nouveau modèle sélectionné : {result.model_id} "
            f"(méthode : {result.peft_method}). Raison : {result.reasoning}"
        ))
        return d, {
            "selection_result": d,
            "target_model":     result.model_id,
            "attempted_models": new_tried,
        }
    except Exception as exc:
        err = {"error": str(exc)}
        return err, {"selection_result": err}


async def _tool_prepare_data(state: OrchestratorState) -> tuple[dict, dict]:
    from agents.data_agent.agent import DataAgentInput, run_data_agent

    intent: dict    = state.get("user_intent") or {}
    selection: dict = state.get("selection_result") or {}
    task            = intent.get("task", "question-answering")
    # Use the auto-selected model_id; fall back to manual target_model if none
    model_id        = selection.get("model_id") or state.get("target_model") or ""

    data_input: DataAgentInput = {
        "job_id":       state["job_id"],
        "filenames":    state["filenames"],
        "task":         task,
        "domain":       intent.get("domain", "general"),
        "language":     intent.get("language", "en"),
        "target_model": model_id,
        "gpu_vram_gb":  state.get("gpu_vram_gb", 4),
        "user_goal":    state.get("user_goal", ""),
    }
    prep_msg = (
        "Préparation des données démarrée — génération des résumés en cours…"
        if task == "summarization"
        else "Préparation des données démarrée — génération des paires QA en cours…"
    )
    await _notify(state, prep_msg)
    result = await run_data_agent(data_input)
    n_pairs = result.get("n_pairs", 0)
    done_msg = (
        f"Données prêtes — {n_pairs} paires (document → résumé) générées et exportées."
        if task == "summarization"
        else f"Données prêtes — {n_pairs} paires QA générées et exportées."
    )
    await _notify(state, done_msg)
    return result, {"data_result": result}


async def _tool_auto_fill_qa(state: OrchestratorState, args: dict) -> tuple[dict, dict]:
    from agents.data_agent.mcp_server.generation import (
        tool_auto_fill_qa, tool_auto_fill_qa_preview,
    )
    from agents.data_agent.mcp_server.export import tool_finish
    from agents.data_agent.agent import DataAgentInput, run_data_agent
    from services.run_manager import get_run_manager

    job_id       = state["job_id"]
    intent       = state.get("user_intent") or {}
    selection    = state.get("selection_result") or {}
    task         = intent.get("task", "question-answering")
    target_model = selection.get("model_id") or state.get("target_model") or ""
    peft_method  = selection.get("peft_method") or "qlora"
    target_pairs = int(args.get("target_pairs", 500))
    current_pairs = int((state.get("data_result") or {}).get("n_pairs", 0))
    llm_reasoning = (args.get("reasoning") or "").strip()

    if not llm_reasoning:
        err = {
            "error": (
                "auto_fill_qa requires the 'reasoning' argument. Explain WHY the "
                "current dataset is insufficient (cite n_pairs and primary_metric)."
            )
        }
        return err, {}

    # ── Notify user + structured decision event ──────────────────────────────
    await _emit_decision(
        state,
        diagnosis=(
            f"Dataset insuffisant — {current_pairs} paires (cible : {target_pairs})"
        ),
        action="auto_fill_qa",
        reason=llm_reasoning,
        changes={
            "current_pairs": str(current_pairs),
            "target_pairs":  str(target_pairs),
        },
    )
    await _notify(state, (
        f"Diagnostic — Dataset insuffisant : {current_pairs} paire(s) disponible(s), "
        f"cible : {target_pairs}. Justification : {llm_reasoning[:300]}. "
        f"Génération d'un aperçu pour confirmation utilisateur…"
    ))

    # ── Generate a real preview of pairs that WILL be added ──────────────────
    sample_pairs: list[dict] = []
    try:
        loop = asyncio.get_event_loop()
        preview = await loop.run_in_executor(
            None,
            lambda: tool_auto_fill_qa_preview(
                job_id=job_id, target_model=target_model, task=task,
                target_peft=peft_method, max_samples=5,
            ),
        )
        for s in (preview.get("samples") or []):
            sample_pairs.append({
                "q": s.get("question", ""),
                "a": s.get("answer", ""),
                "source": s.get("source_file", ""),
            })
    except Exception as exc:
        logger.warning("auto_fill_preview_failed job=%s: %s", job_id, exc)

    # ── Ask user for confirmation, showing the actual generated content ──────
    handle = get_run_manager().get(job_id)
    if handle is not None:
        handle.confirm_event    = asyncio.Event()
        handle.confirm_decision = None
        handle.additional_files = []

        confirm_msg = json.dumps({
            "__confirm__":   True,
            "action":        "auto_fill_qa",
            "current_pairs": current_pairs,
            "target_pairs":  target_pairs,
            "sample_pairs":  sample_pairs,
            "reasoning":     llm_reasoning,
        }, ensure_ascii=False)
        await handle.queue.put(confirm_msg)
        logger.info("auto_fill_qa: waiting for user confirmation job=%s", job_id)

        try:
            await asyncio.wait_for(handle.confirm_event.wait(), timeout=600)
        except asyncio.TimeoutError:
            handle.confirm_decision = "approve"
            logger.info("auto_fill_qa: confirmation timeout → auto-approve job=%s", job_id)

        decision        = handle.confirm_decision
        new_filenames   = list(handle.additional_files)
        handle.confirm_event    = None
        handle.confirm_decision = None
        handle.additional_files = []

        # ── User uploaded additional files — restart data agent ───────────────
        if decision == "upload" and new_filenames:
            all_filenames = list(state.get("filenames") or []) + new_filenames
            await _notify(state, (
                f"Nouveaux fichiers reçus ({len(new_filenames)}) — "
                f"re-traitement avec {len(all_filenames)} fichier(s) au total…"
            ))
            data_input: DataAgentInput = {
                "job_id":       job_id,
                "filenames":    all_filenames,
                "task":         task,
                "domain":       intent.get("domain", "general"),
                "language":     intent.get("language", "en"),
                "target_model": target_model,
                "gpu_vram_gb":  state.get("gpu_vram_gb", 4),
                "user_goal":    state.get("user_goal", ""),
            }
            result = await run_data_agent(data_input)
            n_new = result.get("n_pairs", 0)
            await _notify(state, (
                f"Données prêtes — {n_new} paire(s) QA au total après ajout des nouveaux fichiers."
            ))
            data_result = dict(state.get("data_result") or {})
            data_result.update(result)
            data_result["n_pairs"] = n_new
            return result, {"data_result": data_result, "filenames": all_filenames}

        # ── User refused without uploading — end pipeline ─────────────────────
        if decision == "refuse":
            logger.info("auto_fill_qa: refused by user job=%s", job_id)
            msg = "Génération de données annulée par l'utilisateur. Importez d'autres fichiers et relancez."
            return {"refused": True, "message": msg}, {"error": msg}

    # ── User approved (or no handle / timeout) — proceed with auto-fill ──────
    loop = asyncio.get_event_loop()
    fill = await loop.run_in_executor(
        None,
        lambda: tool_auto_fill_qa(job_id=job_id, target_model=target_model,
                                   task=task, target_peft=peft_method),
    )
    export = await loop.run_in_executor(
        None,
        lambda: tool_finish(job_id=job_id, task=task, target_model=target_model),
    )

    data_result = dict(state.get("data_result") or {})
    data_result["auto_fill"] = fill
    data_result["n_pairs"]   = fill.get("total", data_result.get("n_pairs", 0))
    if "splits" in export:
        data_result["splits"] = export["splits"]

    result = {"auto_fill": fill, "export": export, "total_pairs": fill.get("total")}
    return result, {"data_result": data_result, "target_n_pairs": target_pairs}


async def _tool_train_model(state: OrchestratorState, args: dict) -> tuple[dict, dict]:
    from services.training.trainer import run_training, run_hpo

    selection    = state.get("selection_result") or {}
    model_id     = selection.get("model_id") or state.get("target_model") or ""
    peft_method  = selection.get("peft_method") or "qlora"
    swift_config = selection.get("swift_config") or {}
    intent       = state.get("user_intent") or {}
    data_result  = state.get("data_result") or {}

    n_pairs = int(data_result.get("n_pairs") or data_result.get("total_pairs") or 0)
    task    = intent.get("task", "question-answering")
    vram_gb = state.get("gpu_vram_gb", 4)

    # Fallback: count train.jsonl lines directly if n_pairs not reported
    if n_pairs == 0:
        from data.artifact_store import train_path
        tp = train_path(state["job_id"])
        if tp.exists():
            try:
                n_pairs = sum(1 for _ in tp.open(encoding="utf-8"))
                logger.info("train_model: n_pairs fallback from file count = %d", n_pairs)
            except Exception:
                pass

    # Merge: previous overrides + new overrides from LLM
    new_overrides = args.get("hparam_overrides") or {}
    overrides = dict(state.get("hparam_overrides") or {})
    overrides.update(new_overrides)

    if not model_id:
        err = {"error": "No model selected — call select_model first"}
        return err, {}

    iteration = state.get("improvement_iteration", 0)
    llm_reasoning = (args.get("reasoning") or "").strip()

    # Reject silent retries: from iteration 1 onward, reasoning is mandatory
    if iteration > 0 and not llm_reasoning:
        err = {
            "error": (
                "train_model requires the 'reasoning' argument on iteration > 0. "
                "You must explain: (1) the diagnosis (cite numbers), "
                "(2) what hyperparameters are changing and to what values, "
                "(3) why those values are expected to fix the problem."
            )
        }
        return err, {}

    # ── Notify user with structured diagnostic before training ───────────────
    if iteration == 0:
        await _notify(state, (
            f"Lancement de l'entraînement — modèle : {model_id}, "
            f"méthode PEFT : {peft_method}, dataset : {n_pairs} paires."
        ))
    else:
        # Build a "before → after" diff for each changed hyperparameter
        prev_log = (state.get("improvement_log") or [{}])[-1]
        prev_overrides = prev_log.get("hparam_overrides") or {}
        changes: dict[str, str] = {}
        for k, v in new_overrides.items():
            prev_val = prev_overrides.get(k, "default")
            changes[k] = f"{prev_val} → {v}"

        diagnosis = llm_reasoning.split(".")[0][:200] if llm_reasoning else (
            f"Itération {iteration} — réajustement des hyperparamètres"
        )

        # Push structured decision event for the frontend audit panel
        await _emit_decision(
            state,
            diagnosis=diagnosis,
            action=f"train_model (iteration {iteration})",
            reason=llm_reasoning,
            changes=changes,
        )

        await _notify(state, (
            f"Décision itération {iteration} — {diagnosis}. "
            f"Changements : {', '.join(f'{k}: {v}' for k, v in changes.items()) or 'aucun'}. "
            f"Justification : {llm_reasoning[:300]}"
        ))

    # ── Single-phase HPO or direct training ──────────────────────────────────
    try:
        loop = asyncio.get_event_loop()

        if HPO_ENABLED and iteration == 0:
            # Iteration 0: LLM determines n_trials, run_hpo() IS the full training
            from services.training.hparam_advisor import recommend_n_trials
            llm_for_trials = get_llm_client()
            n_trials = recommend_n_trials(
                model_name=model_id,
                peft_method=peft_method,
                n_pairs=int(n_pairs),
                vram_gb=vram_gb,
                llm_client=llm_for_trials,
                objective=state.get("objective", "balanced"),
            )
            await _notify(state, (
                f"HPO — {n_trials} trials Optuna (entraînement complet par trial, "
                f"époques optimisées anti-overfitting). "
                f"Modèle : {model_id}, méthode : {peft_method}."
            ))
            report = await loop.run_in_executor(
                None,
                lambda: run_hpo(
                    job_id=state["job_id"],
                    model_id=model_id,
                    peft_method=peft_method,
                    n_pairs=int(n_pairs),
                    vram_gb=vram_gb,
                    task=task,
                    n_trials=n_trials,
                    objective=state.get("objective", "balanced"),
                ),
            )
            best_trial = (report.get("hpo_report") or {}).get("best_trial", 0)
            best_loss  = (report.get("hpo_report") or {}).get("best_eval_loss")
            await _notify(state, (
                f"HPO terminé — {n_trials} trials, "
                f"meilleur trial #{best_trial}, eval_loss={best_loss:.4f}."
            ) if best_loss is not None else f"HPO terminé — {n_trials} trials complétés.")

        else:
            # Iterations 1+: direct training with LLM-adjusted overrides
            objective = state.get("objective", "balanced")
            report = await loop.run_in_executor(
                None,
                lambda: run_training(
                    job_id=state["job_id"],
                    model_id=model_id,
                    peft_method=peft_method,
                    swift_config=swift_config,
                    n_pairs=int(n_pairs),
                    vram_gb=vram_gb,
                    task=task,
                    hparam_overrides=overrides or None,
                    objective=objective,
                ),
            )
            await _notify(state, (
                f"Entraînement terminé — modèle : {model_id}, "
                f"méthode : {report.get('peft_method', peft_method)}, "
                f"{int(n_pairs)} paires."
            ))

        # ── Fatal SWIFT CLI error detection ───────────────────────────────────
        result_inner = report.get("result") or {}
        if not result_inner.get("success"):
            swift_err_text = (
                result_inner.get("swift_error", "")
                or result_inner.get("error", "")
            )
            if _is_fatal_swift_error(swift_err_text):
                report["swift_fatal_error"] = True
                attempted = list(state.get("attempted_models") or [model_id])
                if len(attempted) >= 2:
                    await _notify(state, (
                        f"Erreur fatale SWIFT sur {len(attempted)} modèles différents "
                        f"({', '.join(attempted[-2:])}). "
                        "Aucun modèle compatible trouvé — le pipeline va s'arrêter."
                    ))
                else:
                    await _notify(state, (
                        f"Erreur fatale SWIFT pour {model_id} "
                        f"(argument CLI rejeté : {swift_err_text[:120]}). "
                        "Reselection du modèle nécessaire."
                    ))

        return report, {"training_result": report, "hparam_overrides": overrides or None}
    except Exception as exc:
        err = {"error": str(exc)}
        return err, {"training_result": err, "error": str(exc)}


async def _tool_evaluate_model(state: OrchestratorState) -> tuple[dict, dict]:
    from services.training.evaluator import evaluate
    from services.vram_cleanup import cleanup_vram

    training  = state.get("training_result") or {}
    selection = state.get("selection_result") or {}
    model_id  = selection.get("model_id") or state.get("target_model") or ""
    output_dir = (training.get("result") or {}).get("output_dir", "")

    adapter_path = _find_best_checkpoint(output_dir)
    if not adapter_path and output_dir:
        base = Path(output_dir)
        for sub in (sorted(base.iterdir()) if base.exists() else []):
            if sub.is_dir():
                adapter_path = _find_best_checkpoint(str(sub))
                if adapter_path:
                    break

    if not adapter_path or not model_id:
        err = {"error": "No adapter found — train_model must succeed before evaluate_model"}
        return err, {}

    if not (training.get("result") or {}).get("success"):
        err = {"error": "Training did not succeed — cannot evaluate"}
        return err, {}

    # ── Niveau 1 — free VRAM between training and evaluation ────────────────
    # Training leaves SWIFT workers, MCP servers, and cached CUDA pages behind.
    # Cleanup them here so the evaluator has the maximum possible VRAM available
    # before attempting to load the base model.
    cleanup_report = cleanup_vram(label=f"before_eval_{state['job_id']}")
    logger.info(
        "eval pre-flight cleanup: killed=%d  vram_before=%s MB  vram_after=%s MB",
        len(cleanup_report.get("killed", [])),
        cleanup_report.get("vram_before_mb"),
        cleanup_report.get("vram_after_mb"),
    )

    try:
        loop        = asyncio.get_event_loop()
        iteration   = state.get("improvement_iteration", 0)
        eval_strat  = state.get("eval_strategy") or {}

        eval_report = await loop.run_in_executor(
            None,
            lambda: evaluate(
                job_id=state["job_id"],
                adapter_path=adapter_path,
                base_model_id=model_id,
                n_samples=50,
                use_llm_judge=(iteration == 0),
                eval_strategy=eval_strat or None,
            ),
        )
        metrics = eval_report.get("metrics", {})

        # Primary metric drives quality tier and improvement loop
        primary_metric       = eval_strat.get("primary_metric", "rouge1")
        primary_good         = eval_strat.get("good_threshold", 0.45)
        primary_acceptable   = eval_strat.get("acceptable_threshold", 0.28)
        primary_value        = metrics.get(primary_metric, metrics.get("rouge1", 0))

        # Quality tier based on primary metric thresholds
        if primary_value >= primary_good:
            quality_tier = "good"
        elif primary_value >= primary_acceptable:
            quality_tier = "acceptable"
        else:
            quality_tier = "poor"

        tier_fr = {"good": "Bonne qualité", "acceptable": "Qualité acceptable"}.get(
            quality_tier, "Qualité insuffisante"
        )

        logger.info(
            "react_eval iter=%d strategy=%s %s=%.3f tier=%s",
            iteration, eval_strat.get("strategy", "generation"),
            primary_metric, primary_value, quality_tier,
        )

        # Build recommendation message
        n_pairs_current = (state.get("data_result") or {}).get("n_pairs", 0)
        suspicious_threshold = primary_good * 1.9  # sanity upper bound
        trial_log = list(state.get("improvement_log") or []) + [{"primary_metric_value": primary_value}]

        if primary_value > suspicious_threshold:
            recommendation = (
                f"⚠ Score anormalement élevé ({primary_metric}={primary_value:.3f} > {suspicious_threshold:.2f}) — "
                "risque de fuite de données (data leakage). Vérification recommandée avant export."
            )
        elif quality_tier == "good":
            recommendation = "Objectif atteint — le modèle sera exporté."
        elif quality_tier == "acceptable" and _detect_stagnation(trial_log):
            recommendation = "Qualité acceptable et stagnation détectée — pipeline terminé."
        elif n_pairs_current < 300 or primary_value < primary_acceptable * 0.5:
            recommendation = (
                f"Score {primary_metric} ({primary_value:.3f}) très bas avec {n_pairs_current} paires. "
                "Action prévue : génération de paires supplémentaires."
            )
        elif primary_value < primary_acceptable:
            recommendation = (
                f"Score {primary_metric} ({primary_value:.3f}) inférieur au minimum requis ({primary_acceptable}). "
                "Action prévue : ajustement des hyperparamètres."
            )
        elif primary_value < primary_good:
            recommendation = (
                f"Score {primary_metric} ({primary_value:.3f}) au-dessus du minimum ({primary_acceptable}) "
                f"mais inférieur à l'objectif ({primary_good}). "
                "Action prévue : continuation avec learning rate réduit."
            )
        else:
            recommendation = "Analyse en cours."

        # Build metrics display string (show all computed metrics)
        metrics_display = " | ".join(
            f"{m}: {metrics[m]:.3f}" for m in (eval_strat.get("metrics") or ["rouge1", "rougeL", "bleu4"])
            if m in metrics
        )

        await _notify(state, (
            f"Évaluation itération {iteration + 1} [{eval_strat.get('strategy', 'generation')}] — "
            f"{metrics_display} → {tier_fr}. "
            f"(Métrique principale : {primary_metric} | minimum : {primary_acceptable} | objectif : {primary_good} | "
            f"plus le score est élevé, meilleur est le modèle). "
            f"{recommendation}"
        ))

        swift_metrics = (
            (state.get("training_result") or {}).get("result") or {}
        ).get("metrics", {})
        _train_loss = swift_metrics.get("train_loss") or swift_metrics.get("loss")
        _eval_loss  = swift_metrics.get("eval_loss")

        log_entry = {
            "iteration":            iteration,
            "primary_metric":       primary_metric,
            "primary_metric_value": primary_value,
            "rouge1":               metrics.get("rouge1", 0),
            "rougeL":               metrics.get("rougeL", 0),
            "bleu4":                metrics.get("bleu4", 0),
            "f1_token":             metrics.get("f1_token", 0),
            "bertscore_f1":         metrics.get("bertscore_f1", 0),
            "g_eval_overall":       metrics.get("g_eval_overall", 0),
            "faithfulness":         metrics.get("faithfulness", 0),
            "quality_tier":         quality_tier,
            "model_id":             model_id,
            "hparam_overrides":     state.get("hparam_overrides"),
            "n_pairs":              n_pairs_current,
            "train_loss":           round(_train_loss, 4) if _train_loss is not None else None,
            "eval_loss":            round(_eval_loss,  4) if _eval_loss  is not None else None,
            "action_taken":         None,
        }
        # Compute overfitting flag on the log including this new entry
        log_entry["overfitting_detected"] = _detect_overfitting(
            list(state.get("improvement_log") or []) + [log_entry]
        )
        improvement_log = list(state.get("improvement_log") or [])
        improvement_log.append(log_entry)

        return eval_report, {
            "eval_result":           eval_report,
            "improvement_iteration": iteration + 1,
            "improvement_log":       improvement_log,
        }
    except Exception as exc:
        err = {"error": str(exc)}
        return err, {"eval_result": err}


async def _tool_finish(_state: OrchestratorState, args: dict) -> tuple[dict, dict]:
    return {"status": args.get("status", "done"), "summary": args.get("summary", "")}, {}


# ── Tool dispatcher ───────────────────────────────────────────────────────────

async def _run_tool(
    name: str, args: dict, state: OrchestratorState
) -> tuple[dict, dict]:
    """Dispatch tool name → implementation. Returns (result, state_updates)."""
    try:
        if name == "profile_files":
            return await _tool_profile_files(state)
        if name == "check_domain_compatibility":
            return await _tool_check_domain_compatibility(state)
        if name == "extract_intent":
            return await _tool_extract_intent(state)
        if name == "detect_eval_strategy":
            return await _tool_detect_eval_strategy(state)
        if name == "check_feasibility":
            return await _tool_check_feasibility(state)
        if name == "select_model":
            return await _tool_select_model(state)
        if name == "reselect_model":
            return await _tool_reselect_model(state, args)
        if name == "prepare_data":
            return await _tool_prepare_data(state)
        if name == "auto_fill_qa":
            return await _tool_auto_fill_qa(state, args)
        if name == "train_model":
            return await _tool_train_model(state, args)
        if name == "evaluate_model":
            return await _tool_evaluate_model(state)
        if name == "finish":
            return await _tool_finish(state, args)
        return {"error": f"Unknown tool: {name}"}, {}
    except Exception as exc:
        logger.error("tool_error tool=%s: %s", name, exc)
        return {"error": str(exc)}, {}


# ── LangGraph nodes ───────────────────────────────────────────────────────────

async def _react_agent(state: OrchestratorState) -> dict:
    """LLM observes current context and decides the next tool to call."""
    messages = list(state.get("messages") or [])
    system   = _build_system_prompt(state)

    response = get_llm_client().call_with_tools(
        messages=messages,
        tools=PIPELINE_TOOLS,
        system=system,
    )
    logger.info(
        "react_agent tool_calls=%s",
        [tc["function"]["name"] for tc in response.get("tool_calls", [])],
    )
    return {"messages": messages + [response]}


async def _tool_executor(state: OrchestratorState) -> dict:
    """Execute the tool chosen by the LLM and append results to messages."""
    messages = list(state.get("messages") or [])
    last_msg = messages[-1]
    tool_calls = last_msg.get("tool_calls", [])

    tool_results: list[dict] = []
    state_updates: dict = {}

    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}

        logger.info("react_execute tool=%s args=%s", name, args)
        result, updates = await _run_tool(name, args, state)

        # Merge state updates (later tools can see earlier updates via state)
        state_updates.update(updates)

        tool_results.append({
            "role":         "tool",
            "tool_call_id": tc["id"],
            "content":      json.dumps(result, ensure_ascii=False, default=str),
        })
        logger.info("react_result tool=%s ok=%s", name, "error" not in result)

    state_updates["messages"] = messages + tool_results
    return state_updates


_METRIC_LABELS = {
    "rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
    "bleu4": "BLEU-4", "f1_token": "F1-token", "exact_match": "Exact-Match",
    "bertscore_f1": "BERTScore-F1", "g_eval_overall": "G-Eval",
    "faithfulness": "Faithfulness", "f1_macro": "F1-macro",
    "accuracy": "Accuracy", "entity_f1": "Entity-F1",
}


def _format_metric_line(entry: dict) -> str:
    """Build a metric display string: primary metric first, then up to 3 available
    secondary metrics. Works for both improvement_log entries and eval_report metrics.
    """
    primary = entry.get("primary_metric", "rouge1")
    pv = entry.get("primary_metric_value")
    if pv is None:
        pv = entry.get(primary, entry.get("rouge1", 0)) or 0
    parts = [f"{_METRIC_LABELS.get(primary, primary)} = {float(pv):.3f}"]
    for m in ("bertscore_f1", "g_eval_overall", "faithfulness", "rougeL", "bleu4", "f1_token"):
        if len(parts) >= 4:
            break
        if m != primary and entry.get(m):
            parts.append(f"{_METRIC_LABELS.get(m, m)} = {float(entry[m]):.3f}")
    return " | ".join(parts)


def _build_decision_journal(
    state: OrchestratorState,
    finish_status: str,
    finish_summary: str,
) -> list[dict]:
    """Build a human-readable audit trail of every orchestrator decision."""
    from data.artifact_store import domain_path as _domain_path
    journal: list[dict] = []
    step = 1

    # 1. Files
    filenames = state.get("filenames") or []
    ds = state.get("dataset_size") or {}
    journal.append({
        "étape": f"{step}. Analyse des fichiers",
        "décision": f"{len(filenames)} fichier(s) : {', '.join(filenames) or 'aucun'}",
        "justification": (
            f"Taille estimée : {ds.get('total_words', '?')} mots, "
            f"{ds.get('file_count', '?')} fichier(s). Profiling automatique du contenu et du format."
        ),
        "statut": "ok",
    })
    step += 1

    # 2. Domain
    domain = state.get("domain") or (state.get("user_intent") or {}).get("domain") or "inconnu"
    try:
        _dp = _domain_path(state["job_id"])
        if _dp.exists():
            detected = json.loads(_dp.read_text(encoding="utf-8")).get("domain", "")
            if detected:
                domain = detected
    except Exception:
        pass
    journal.append({
        "étape": f"{step}. Détection du domaine",
        "décision": f"Domaine : {domain}",
        "justification": (
            "Analyse sémantique du contenu pour identifier le domaine métier "
            "et adapter les seuils d'évaluation ROUGE/BLEU."
        ),
        "statut": "ok",
    })
    step += 1

    # 3. Intent
    intent = state.get("user_intent") or {}
    journal.append({
        "étape": f"{step}. Extraction de l'intention",
        "décision": f"Tâche : {intent.get('task', '?')} | Langue : {state.get('language', '?')}",
        "justification": f"Objectif déclaré par l'utilisateur : \"{state.get('user_goal', '')}\".",
        "statut": "ok",
    })
    step += 1

    # 4. Feasibility
    feasibility = state.get("feasibility") or {}
    feas_status = feasibility.get("status", "?")
    feas_reason = (
        feasibility.get("reason")
        or feasibility.get("message")
        or "Données conformes aux critères minimaux de fine-tuning."
    )
    journal.append({
        "étape": f"{step}. Vérification de faisabilité",
        "décision": f"Statut : {feas_status}",
        "justification": feas_reason,
        "statut": "ok" if feas_status == "GO" else ("warn" if feas_status == "WARNING" else "err"),
    })
    step += 1

    # 5. Model selection
    selection = state.get("selection_result") or {}
    if selection:
        journal.append({
            "étape": f"{step}. Sélection du modèle",
            "décision": (
                f"Modèle : {selection.get('model_id', '?')} | "
                f"Méthode PEFT : {selection.get('peft_method', '?')}"
            ),
            "justification": (
                selection.get("reasoning")
                or "Sélection par similarité vectorielle FAISS + classement LLM "
                   "selon VRAM disponible, tâche, domaine et objectif utilisateur."
            ),
            "statut": "ok",
        })
        step += 1

    # 6. Data preparation
    data = state.get("data_result") or {}
    if data:
        journal.append({
            "étape": f"{step}. Préparation des données",
            "décision": (
                f"{data.get('n_pairs', '?')} paires QA générées "
                f"(format {data.get('format', '?')})"
            ),
            "justification": (
                "Génération automatique de paires question-réponse depuis les documents. "
                "Déduplication et découpage stratifié train / validation / test."
            ),
            "statut": "ok",
        })
        step += 1

    # 7+. One entry per training iteration (from improvement_log)
    improvement_log = state.get("improvement_log") or []
    tier_labels = {
        "good":       "Bonne qualité",
        "acceptable": "Qualité acceptable",
        "poor":       "Qualité insuffisante",
    }
    for i, entry in enumerate(improvement_log):
        tier    = entry.get("quality_tier", "poor")
        overrides = entry.get("hparam_overrides") or {}
        n_pairs = entry.get("n_pairs", "?")
        model_used = entry.get("model_id", "?")
        overrides_str = (
            f"Hyperparamètres ajustés : {json.dumps(overrides, ensure_ascii=False)}"
            if overrides else "Hyperparamètres par défaut."
        )
        journal.append({
            "étape": f"{step}. Entraînement — itération {i + 1}",
            "décision": (
                f"{_format_metric_line(entry)} → {tier_labels.get(tier, tier)}"
            ),
            "justification": (
                f"Modèle utilisé : {model_used}. "
                f"Paires d'entraînement : {n_pairs}. {overrides_str}"
            ),
            "statut": "ok" if tier == "good" else ("warn" if tier == "acceptable" else "info"),
        })
        step += 1

    # Final evaluation
    eval_res     = state.get("eval_result") or {}
    eval_metrics = eval_res.get("metrics") or {}
    if eval_metrics:
        tier_final  = eval_metrics.get("quality_tier", "poor")
        tier_label  = tier_labels.get(tier_final, "Qualité insuffisante")
        quality_sum = eval_metrics.get("quality_summary", "")
        journal.append({
            "étape": f"{step}. Évaluation finale",
            "décision": (
                f"{_format_metric_line(eval_metrics)} → {tier_label}"
            ),
            "justification": (
                quality_sum
                or "Évaluation sur jeu de test réservé, jamais vu pendant l'entraînement."
            ),
            "statut": (
                "ok" if tier_final == "good"
                else "warn" if tier_final == "acceptable"
                else "err"
            ),
        })
        step += 1

    # Final pipeline decision
    status_labels = {
        "trained":             "Modèle fine-tuné avec succès — adaptateur exporté",
        "trained_low_quality": "Fine-tuning terminé — qualité limitée malgré les tentatives d'amélioration",
        "blocked":             "Pipeline bloqué — préparation des données impossible",
        "done":                "Pipeline terminé",
        "error":               "Erreur critique — pipeline interrompu",
    }
    journal.append({
        "étape": f"{step}. Décision finale",
        "décision": status_labels.get(finish_status, finish_status),
        "justification": (
            finish_summary
            or "Pipeline terminé selon les critères de qualité et les tentatives d'amélioration effectuées."
        ),
        "statut": (
            "ok" if finish_status in ("trained", "done")
            else "warn" if finish_status == "trained_low_quality"
            else "err"
        ),
    })

    return journal


async def _finalize(state: OrchestratorState) -> dict:
    """Build and persist the final output after the LLM calls finish."""
    # Auto-evaluate if training succeeded but LLM skipped evaluate_model
    training = state.get("training_result") or {}
    if (training.get("result") or {}).get("success") and not state.get("eval_result"):
        logger.info("finalize: auto-triggering evaluate_model (LLM skipped it)")
        try:
            _, eval_updates = await _tool_evaluate_model(state)
            state = {**state, **eval_updates}  # type: ignore[assignment]
        except Exception as exc:
            logger.warning("finalize: auto-eval failed: %s", exc)

    feasibility = state.get("feasibility") or {}
    selection   = state.get("selection_result") or {}
    data        = state.get("data_result") or {}
    training    = state.get("training_result") or {}
    eval_res    = state.get("eval_result") or {}
    intent      = state.get("user_intent") or {}
    messages    = state.get("messages") or []

    # Extract finish args from last finish tool call
    finish_status  = "done"
    finish_summary = ""
    for msg in reversed(messages):
        for tc in msg.get("tool_calls", []):
            if tc["function"]["name"] == "finish":
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                    finish_status  = args.get("status", "done")
                    finish_summary = args.get("summary", "")
                except Exception:
                    pass
                break

    decision_journal = _build_decision_journal(state, finish_status, finish_summary)

    output = {
        "job_id":            state["job_id"],
        "status":            finish_status,
        "summary":           finish_summary,
        "task":              intent.get("task"),
        "domain":            intent.get("domain"),
        "feasibility":       feasibility,
        "selected_model":    selection,
        "dataset":           data,
        "training":          training,
        "evaluation":        eval_res,
        "improvement_log":   state.get("improvement_log") or [],
        "decision_journal":  decision_journal,
        "error":             state.get("error"),
    }
    write_json(Path(state["artifact_dir"]) / "final_output.json", output)
    return {"final_output": output}


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_after_agent(state: OrchestratorState) -> str:
    messages = state.get("messages") or []
    if not messages:
        return "finalize"

    last = messages[-1]
    tool_calls = last.get("tool_calls", [])

    if not tool_calls:
        return "finalize"

    # If LLM called finish → skip tool_executor, go straight to finalize
    if tool_calls[0]["function"]["name"] == "finish":
        return "finalize"

    return "tool_executor"


# ── Checkpoint helper ─────────────────────────────────────────────────────────

def _find_best_checkpoint(output_dir: str) -> str | None:
    if not output_dir:
        return None
    base = Path(output_dir)
    if not base.exists():
        return None

    best_link = base / "best_model"
    if best_link.exists():
        return str(best_link.resolve())

    checkpoints = sorted(
        base.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        if (base / "adapter_config.json").exists():
            return str(base)
        return None

    best_path, best_loss = None, float("inf")
    for cp in checkpoints:
        ts = cp / "trainer_state.json"
        if ts.exists():
            try:
                data = json.loads(ts.read_text(encoding="utf-8"))
                for entry in data.get("log_history", []):
                    if "eval_loss" in entry and entry["eval_loss"] < best_loss:
                        best_loss = entry["eval_loss"]
                        best_path = str(cp)
            except Exception:
                pass

    return best_path or str(checkpoints[-1])


# ── Graph assembly ────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(OrchestratorState)

    g.add_node("react_agent",   _react_agent)
    g.add_node("tool_executor", _tool_executor)
    g.add_node("finalize",      _finalize)

    g.set_entry_point("react_agent")
    g.add_conditional_edges(
        "react_agent",
        _route_after_agent,
        {
            "tool_executor": "tool_executor",
            "finalize":      "finalize",
        },
    )
    g.add_edge("tool_executor", "react_agent")
    g.add_edge("finalize",      END)

    return g


_COMPILED_GRAPH = _build_graph().compile()


async def run_orchestrator(state: OrchestratorState) -> OrchestratorState:
    return await _COMPILED_GRAPH.ainvoke(state, config={"recursion_limit": 100})
