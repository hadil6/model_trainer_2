"""LangGraph state definitions shared across all agents."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _merge_dicts(a, b) -> dict:
    return {**(a or {}), **(b or {})}


def _merge_lists(a: list, b: list) -> list:
    """Concatène deux listes sans doublons."""
    seen = {id(x) for x in a}
    return a + [x for x in b if id(x) not in seen]


class DataAgentState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────
    job_id: str
    files: list[str]
    task: str
    domain: str
    language: str
    target_model: str
    user_context: str
    tone: str

    # ── Messages ──────────────────────────────────────────────────────────
    messages: Annotated[list, operator.add]

    # ── Runtime state ─────────────────────────────────────────────────────
    completed_steps: Annotated[list[str], operator.add]           # ← reducer add
    current_file_profiles: Annotated[dict[str, Any], _merge_dicts]  # ← merge
    file_summaries: Annotated[dict[str, str], _merge_dicts]         # ← merge
    cleaned_texts: Annotated[dict[str, str], _merge_dicts]          # ← merge
    coherence_score: float
    coherence_report: Annotated[dict[str, Any], _merge_dicts]       # ← merge
    qa_pairs: Annotated[list[dict], operator.add]                   # ← add
    dropped_pairs: Annotated[list[dict], operator.add]              # ← add
    generate_qa_passes: Annotated[dict[str, int], _merge_dicts]     # ← merge
    iteration: int
    max_iterations: int

    # ── Per-step scratchpad ───────────────────────────────────────────────
    _last_tool: str
    _last_args: dict[str, Any]
    _last_obs: str

    # ── Quality signals ───────────────────────────────────────────────────
    eval_scores: Annotated[dict[str, Any], _merge_dicts]            # ← merge
    judge_verdicts: Annotated[list[dict], operator.add]             # ← add

    # ── Output ────────────────────────────────────────────────────────────
    output_path: str
    readiness_status: str
    readiness_report: Annotated[dict[str, Any], _merge_dicts]       # ← merge
    final_report: Annotated[dict[str, Any], _merge_dicts]           # ← merge
    error: str


class OrchestratorState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────
    user_goal: str
    files: list[str]
    domain: str
    language: str
    target_model: str
    output_path: str

    # ── Messages ──────────────────────────────────────────────────────────
    messages: Annotated[list, operator.add]

    # ── Routing ───────────────────────────────────────────────────────────
    next_agent: str
    routing_rationale: str
    route_count: int
    job_id: str
    gpu_vram_gb: int
    owner_email: str  # email of the run's owner — used for isolation in /api/runs

    # ── Metadata ──────────────────────────────────────────────────────────
    user_intent: Annotated[dict[str, Any], _merge_dicts]
    dataset_summary: Annotated[dict[str, Any], _merge_dicts]
    constraints: Annotated[dict[str, Any], _merge_dicts]
    orchestrator_output: Annotated[dict[str, Any], _merge_dicts]

    # ── Sub-agent results ─────────────────────────────────────────────────
    selection_result: Annotated[dict[str, Any], _merge_dicts]
    data_agent_result: Annotated[dict[str, Any], _merge_dicts]

    # ── Output ────────────────────────────────────────────────────────────
    final_output: Annotated[dict[str, Any], _merge_dicts]
    error: str