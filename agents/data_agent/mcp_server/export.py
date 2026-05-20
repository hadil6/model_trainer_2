"""Export tool implementation (plain function — registered on MCP app in server.py)."""

from __future__ import annotations

import json
import logging
import math
import random

from agents.data_agent.services.qa import QAPair, export_dataset
from data.artifact_store import (
    agent_log_path,
    dataset_path,
    dropped_qa_path,
    qa_final_path,
    read_jsonl,
    test_path,
    train_path,
    val_path,
    write_json,
)

logger = logging.getLogger(__name__)


def tool_finish(job_id: str, task: str, target_model: str = "") -> dict:
    """Export final dataset: dataset.jsonl + train/val/test splits."""
    raw = read_jsonl(qa_final_path(job_id))
    if not raw:
        return {"error": "No Q&A pairs found — run generate_qa and deduplicate_qa first"}

    pairs = [QAPair(question=r["question"], answer=r["answer"],
                    source_file=r.get("source_file", ""), confidence=r.get("confidence", 0.5))
             for r in raw]

    records = export_dataset(pairs, task=task, model_id=target_model)
    n = len(records)

    out = dataset_path(job_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    splits = _split_dataset(records)
    for path_fn, key in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
        p = path_fn(job_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in splits[key]) + "\n",
            encoding="utf-8",
        )

    dropped_count = len(read_jsonl(dropped_qa_path(job_id)))
    result = {
        "dataset_path": str(out),
        "count": n,
        "n_pairs": n,
        "format": "jsonl",
        "task": task,
        "target_model": target_model,
        "splits": {k: len(v) for k, v in splits.items()},
        "dropped_pairs": dropped_count,
        "verdict": "done",
    }
    write_json(agent_log_path(job_id), {"job_id": job_id, "result": result})
    logger.info("finish job=%s count=%d target=%s", job_id, n, target_model)
    return result


def _split_dataset(records: list[dict]) -> dict[str, list[dict]]:
    n = len(records)
    if n < 3:
        return {"train": records, "val": [], "test": []}
    shuffled = records.copy()
    random.shuffle(shuffled)
    val_n  = max(1, math.floor(n * 0.10))
    test_n = max(1, math.floor(n * 0.10))
    return {
        "train": shuffled[: n - val_n - test_n],
        "val":   shuffled[n - val_n - test_n : n - test_n],
        "test":  shuffled[n - test_n :],
    }
