"""DataAgent LangGraph state."""

from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class DataAgentState(TypedDict):
    job_id: str
    filenames: list[str]
    task: str
    domain: str
    language: str
    target_model: str | None
    user_goal: str
    # Progress — only scalars + step markers; full data lives on disk
    completed_steps: Annotated[list[str], operator.add]
    iteration: int
    qa_count: int
    coherence_passed: bool | None
    readiness_verdict: str | None
    messages: Annotated[list[BaseMessage], add_messages]
    result: dict | None
    error: str | None


class DataAgentInput(TypedDict):
    job_id: str
    filenames: list[str]
    task: str
    domain: str
    language: str
    target_model: str | None
    gpu_vram_gb: int
    user_goal: str
