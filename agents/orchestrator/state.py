"""Orchestrator LangGraph state."""

from __future__ import annotations

from typing_extensions import TypedDict


class OrchestratorState(TypedDict):
    job_id: str
    artifact_dir: str
    filenames: list[str]
    user_goal: str
    language: str
    domain: str
    objective: str                     # "speed" | "balanced" | "performance"
    target_model: str | None
    gpu_vram_gb: int
    file_profiles: list[dict]
    dataset_size: dict | None          # {total_words, total_chars, total_rows, file_count}
    user_intent: dict | None
    feasibility: dict | None
    selection_result: dict | None
    data_result: dict | None
    training_result: dict | None
    eval_result: dict | None
    eval_strategy: dict | None          # detected by detect_eval_strategy tool
    improvement_iteration: int
    improvement_log: list[dict]
    hparam_overrides: dict | None
    target_n_pairs: int | None
    attempted_models: list[str]
    messages: list[dict]
    final_output: dict | None
    error: str | None
    task_hint: str | None               # explicit task selected by user on the frontend
