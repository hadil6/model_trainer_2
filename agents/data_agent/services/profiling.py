"""Profiler dispatcher for uploaded files.

Public API
----------
profile_file(path: Path) -> dict
summarize_profile(profile: dict, task: str, llm_client: LLMClient) -> str
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agents.llm_client import LLMClient
from data.profilers.csv_profiler import CsvProfiler
from data.profilers.excel_profiler import ExcelProfiler
from data.profilers.pdf_profiler import PdfProfiler
from data.profilers.txt_profiler import TxtProfiler

logger = logging.getLogger(__name__)

_EXT_TO_KIND: dict[str, str] = {
    ".csv":  "csv",
    ".pdf":  "pdf",
    ".txt":  "txt",
    ".xlsx": "excel",
    ".xls":  "excel",
    ".xlsm": "excel",
}

_CSV_PROFILER   = CsvProfiler()
_EXCEL_PROFILER = ExcelProfiler()
_PDF_PROFILER   = PdfProfiler()
_TXT_PROFILER   = TxtProfiler()


def profile_file(path: Path) -> dict:
    """Profile a file and return the profile dict with an added `file_kind` key."""
    kind = _EXT_TO_KIND.get(path.suffix.lower(), "unsupported")
    if kind == "unsupported":
        raise ValueError(f"Unsupported file type: {path.name}")

    content = path.read_bytes()

    if kind == "csv":
        payload = _CSV_PROFILER.profile(path.name, content)
    elif kind == "excel":
        payload = _EXCEL_PROFILER.profile(path.name, content)
    elif kind == "pdf":
        payload = _PDF_PROFILER.profile(path.name, content)
    else:
        payload = _TXT_PROFILER.profile(path.name, content)

    payload["file_kind"] = kind
    payload["filename"]  = path.name
    return payload


def summarize_profile(profile: dict, task: str, llm_client: LLMClient) -> str:
    """Ask the LLM for a semantic summary of a file profile."""
    prompt_content = json.dumps(profile, default=str)[:5000]
    return llm_client.complete(
        system=(
            "You are a data analyst. Given a file profile, write a concise semantic summary "
            "covering: likely modeling use cases for the stated task, notable risks, and "
            "overall readiness. 3-5 sentences maximum."
        ),
        user=f"Task: {task}\n\nProfile:\n{prompt_content}",
    )
