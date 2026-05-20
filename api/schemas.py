"""API-layer request/response types."""

from __future__ import annotations

from pydantic import BaseModel


class RunResult(BaseModel):
    status: str             # running | done | error
    error: str | None = None
    result: dict | None = None


class DatasetResponse(BaseModel):
    count: int
    format: str
    pairs: list[dict]
