"""RunHandle registry and SSE queue management.

Public API
----------
RunHandle         — per-run state (queue, result, status)
RunManager        — singleton registry; get_or_create(), get(), push_event(), set_result()
get_run_manager() — module-level singleton accessor
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunHandle:
    job_id: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    result: dict | None = None
    status: str = "running"  # running | done | error
    confirm_event: asyncio.Event | None = None    # set when waiting for user approval
    confirm_decision: str | None = None            # "approve" | "refuse" | "upload"
    additional_files: list = field(default_factory=list)  # filenames uploaded by user on refuse


class RunManager:
    """Thread-safe registry of active runs with SSE queues."""

    def __init__(self) -> None:
        self._runs: dict[str, RunHandle] = {}

    def create(self, job_id: str) -> RunHandle:
        handle = RunHandle(job_id=job_id)
        self._runs[job_id] = handle
        return handle

    def get(self, job_id: str) -> RunHandle | None:
        return self._runs.get(job_id)

    async def push_event(self, job_id: str, event_type: str, data: Any) -> None:
        handle = self._runs.get(job_id)
        if handle is None:
            return
        payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
        await handle.queue.put(payload)

    def set_result(self, job_id: str, result: dict) -> None:
        handle = self._runs.get(job_id)
        if handle:
            handle.result = result
            handle.status = "done" if not result.get("error") else "error"

    def cleanup(self, job_id: str) -> None:
        self._runs.pop(job_id, None)


_MANAGER: RunManager | None = None


def get_run_manager() -> RunManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = RunManager()
    return _MANAGER
