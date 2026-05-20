"""Centralized artifact I/O for all MCP tools.

All tools use this module to read/write artifacts so path construction
is never duplicated across tool handlers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.config import get_settings

logger = logging.getLogger(__name__)


def _job_dir(job_id: str) -> Path:
    return Path(get_settings().artifacts_dir) / job_id


def input_file_path(job_id: str, filename: str) -> Path:
    return _job_dir(job_id) / "input_files" / filename


def profile_path(job_id: str, filename: str) -> Path:
    return _job_dir(job_id) / "profiles" / f"{filename}.json"


def summary_path(job_id: str, filename: str) -> Path:
    return _job_dir(job_id) / "summaries" / f"{filename}.json"


def cleaned_path(job_id: str, filename: str) -> Path:
    return _job_dir(job_id) / "cleaned" / f"{filename}.txt"


def ocr_review_path(job_id: str, filename: str) -> Path:
    return _job_dir(job_id) / "ocr_review" / f"{filename}.json"


def qa_path(job_id: str, filename: str) -> Path:
    return _job_dir(job_id) / "qa" / f"{filename}.jsonl"


def qa_final_path(job_id: str) -> Path:
    return _job_dir(job_id) / "qa_final.jsonl"


def coherence_path(job_id: str) -> Path:
    return _job_dir(job_id) / "coherence.json"


def readiness_path(job_id: str) -> Path:
    return _job_dir(job_id) / "readiness.json"


def readiness_report_path(job_id: str) -> Path:
    return _job_dir(job_id) / "readiness_report.json"


def dataset_path(job_id: str) -> Path:
    return _job_dir(job_id) / "dataset.jsonl"


def train_path(job_id: str) -> Path:
    return _job_dir(job_id) / "train.jsonl"


def val_path(job_id: str) -> Path:
    return _job_dir(job_id) / "val.jsonl"


def test_path(job_id: str) -> Path:
    return _job_dir(job_id) / "test.jsonl"


def chunks_path(job_id: str, filename: str) -> Path:
    return _job_dir(job_id) / "chunks" / f"{filename}.jsonl"


def raw_extraction_path(job_id: str, filename: str) -> Path:
    return _job_dir(job_id) / "extractions" / f"{filename}.md"


def merged_corpus_path(job_id: str) -> Path:
    return _job_dir(job_id) / "merged_corpus.txt"


def dropped_qa_path(job_id: str) -> Path:
    return _job_dir(job_id) / "dropped_qa.json"


def eval_qa_path(job_id: str) -> Path:
    return _job_dir(job_id) / "eval_qa.jsonl"


def domain_path(job_id: str) -> Path:
    return _job_dir(job_id) / "domain.json"


def agent_log_path(job_id: str) -> Path:
    return _job_dir(job_id) / "agent_log.json"


def training_output_path(job_id: str) -> Path:
    return _job_dir(job_id) / "training_output"


def training_report_path(job_id: str) -> Path:
    return _job_dir(job_id) / "training_report.json"


def eval_report_path(job_id: str) -> Path:
    return _job_dir(job_id) / "eval_report.json"


# ── I/O helpers ──────────────────────────────────────────────────────────────

def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line in %s", path)
    return records


def read_all_profiles(job_id: str) -> list[dict]:
    """Load all written profiles for a job."""
    profiles_dir = _job_dir(job_id) / "profiles"
    if not profiles_dir.exists():
        return []
    profiles = []
    for p in sorted(profiles_dir.glob("*.json")):
        try:
            profiles.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Failed to read profile %s: %s", p, exc)
    return profiles


def read_all_summaries(job_id: str) -> dict[str, str]:
    """Load all written summaries for a job. Returns {filename: summary_text}."""
    summaries_dir = _job_dir(job_id) / "summaries"
    if not summaries_dir.exists():
        return {}
    result = {}
    for p in sorted(summaries_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            filename = data.get("filename", p.stem)
            result[filename] = data.get("summary", "")
        except Exception as exc:
            logger.warning("Failed to read summary %s: %s", p, exc)
    return result


def read_all_qa(job_id: str) -> list[dict]:
    """Merge all per-file QA JSONL files into a single list."""
    qa_dir = _job_dir(job_id) / "qa"
    if not qa_dir.exists():
        return []
    all_pairs = []
    for p in sorted(qa_dir.glob("*.jsonl")):
        all_pairs.extend(read_jsonl(p))
    return all_pairs


def ensure_job_dirs(job_id: str) -> None:
    for subdir in ("input_files", "profiles", "summaries", "cleaned", "ocr_review", "qa", "extractions"):
        (_job_dir(job_id) / subdir).mkdir(parents=True, exist_ok=True)
