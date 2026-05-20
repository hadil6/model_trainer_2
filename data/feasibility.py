"""Deterministic feasibility rules that run on actual profiler output.

Design principle:
- Rules can only *escalate* severity (GO → WARNING → BLOCKED), never downgrade.
- Rules enforce hard physical impossibilities only.
  Everything context-dependent (task difficulty, balance, etc.) is left to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FeasibilityStatus(str, Enum):
    GO = "GO"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"


_SEVERITY = {
    FeasibilityStatus.GO: 0,
    FeasibilityStatus.WARNING: 1,
    FeasibilityStatus.BLOCKED: 2,
}


@dataclass
class FeasibilityOverride:
    status: FeasibilityStatus = FeasibilityStatus.GO
    blocking_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "blocking_reasons": self.blocking_reasons,
            "warnings": self.warnings,
        }


# ── Individual rules ──────────────────────────────────────────────────────────

def _check_empty_dataset(profile: dict) -> FeasibilityOverride:
    override = FeasibilityOverride()
    file_kind = profile.get("file_kind", "")

    # Skip empty-check when timeout already explains the missing content
    if "profile_extraction_timed_out" in (profile.get("risk_flags") or []):
        return override

    if file_kind in ("csv", "excel"):
        row_count = profile.get("metadata", {}).get("row_count", 0) or 0
        if row_count == 0:
            override.status = FeasibilityStatus.BLOCKED
            override.blocking_reasons.append("Dataset is empty (0 rows)")
    elif file_kind in ("txt", "pdf"):
        word_count = profile.get("metadata", {}).get("word_count", 0) or 0
        if word_count == 0:
            override.status = FeasibilityStatus.BLOCKED
            override.blocking_reasons.append("Document is empty (0 words)")

    return override


def _check_row_count(profile: dict) -> FeasibilityOverride:
    """Block CSV datasets with ≤50 rows — too small for any fine-tuning signal."""
    override = FeasibilityOverride()
    if profile.get("file_kind") not in ("csv", "excel"):
        return override

    row_count = profile.get("metadata", {}).get("row_count")
    if row_count is None:
        return override

    if row_count <= 50:
        override.status = FeasibilityStatus.BLOCKED
        override.blocking_reasons.append(
            f"Dataset has only {row_count} rows — minimum 51 required for any fine-tuning"
        )
    return override


def _check_null_ratio(profile: dict) -> FeasibilityOverride:
    """Block when the majority of columns are >50% null."""
    override = FeasibilityOverride()
    if profile.get("file_kind") not in ("csv", "excel"):
        return override

    columns = profile.get("columns", [])
    if not columns:
        return override

    mostly_null = [c for c in columns if (c.get("null_ratio") or 0) > 0.5]
    if len(mostly_null) > len(columns) / 2:
        override.status = FeasibilityStatus.BLOCKED
        override.blocking_reasons.append(
            f"{len(mostly_null)}/{len(columns)} columns are >50% null — data is unusable"
        )
    return override


def _check_high_duplicate_ratio(profile: dict) -> FeasibilityOverride:
    """Block >80% duplicates; warn >50%."""
    override = FeasibilityOverride()
    if profile.get("file_kind") not in ("csv", "excel"):
        return override

    dup_ratio = profile.get("quality_checks", {}).get("duplicate_row_ratio") or 0.0

    if dup_ratio > 0.80:
        override.status = FeasibilityStatus.BLOCKED
        override.blocking_reasons.append(
            f"{dup_ratio:.0%} of rows are duplicates — model will memorise a single example"
        )
    elif dup_ratio > 0.50:
        override.status = FeasibilityStatus.WARNING
        override.warnings.append(
            f"{dup_ratio:.0%} of rows are duplicates — training distribution will be heavily biased"
        )
    return override


def _check_near_constant_columns(profile: dict) -> FeasibilityOverride:
    """Warn when most columns carry no variance."""
    override = FeasibilityOverride()
    if profile.get("file_kind") not in ("csv", "excel"):
        return override

    near_constant = profile.get("quality_checks", {}).get("near_constant_columns") or []
    columns = profile.get("columns", [])
    if not columns:
        return override

    if len(near_constant) > len(columns) / 2:
        override.status = FeasibilityStatus.WARNING
        override.warnings.append(
            f"{len(near_constant)}/{len(columns)} columns have near-zero variance — "
            "the model has very little signal to learn from"
        )
    return override


def _check_text_quality(profile: dict) -> FeasibilityOverride:
    """Block >30% garbled; warn >20% garbled or >50% boilerplate."""
    override = FeasibilityOverride()
    if profile.get("file_kind") not in ("txt", "pdf"):
        return override

    quality = profile.get("quality_checks", {}) or {}
    garbled = quality.get("garbled_ratio") or 0
    boilerplate = quality.get("boilerplate_ratio") or 0

    if garbled > 0.3:
        override.status = FeasibilityStatus.BLOCKED
        override.blocking_reasons.append(
            f"Text is predominantly garbled ({garbled:.0%} non-printable characters)"
        )
    elif garbled > 0.2:
        override.status = FeasibilityStatus.WARNING
        override.warnings.append(f"Significant garbled content ({garbled:.0%})")

    if boilerplate > 0.5:
        if _SEVERITY[override.status] < _SEVERITY[FeasibilityStatus.WARNING]:
            override.status = FeasibilityStatus.WARNING
        override.warnings.append(
            f"High boilerplate ratio ({boilerplate:.0%} repeated lines)"
        )
    return override


def _check_risk_flags(profile: dict) -> FeasibilityOverride:
    override = FeasibilityOverride()
    flags = set(profile.get("risk_flags") or [])

    if "profile_extraction_timed_out" in flags:
        override.status = FeasibilityStatus.BLOCKED
        override.blocking_reasons.append(
            "PDF extraction timed out — the file is too large or image-heavy for the current timeout. "
            "Try a smaller file or increase pdf_extraction_timeout_seconds in settings."
        )
        return override

    warning_flags = {"empty_dataset", "high_garbled_text", "marker_timeout_fitz_fallback"}
    found = flags & warning_flags
    if found:
        if _SEVERITY[override.status] < _SEVERITY[FeasibilityStatus.WARNING]:
            override.status = FeasibilityStatus.WARNING
        for flag in sorted(found):
            override.warnings.append(f"Risk flag: {flag}")
    return override


_ALL_RULES = [
    _check_empty_dataset,
    _check_row_count,
    _check_null_ratio,
    _check_high_duplicate_ratio,
    _check_near_constant_columns,
    _check_text_quality,
    _check_risk_flags,
]


# ── Aggregation ───────────────────────────────────────────────────────────────

def _merge(overrides: list[FeasibilityOverride]) -> FeasibilityOverride:
    merged = FeasibilityOverride()
    for ov in overrides:
        if _SEVERITY[ov.status] > _SEVERITY[merged.status]:
            merged.status = ov.status
        merged.blocking_reasons.extend(ov.blocking_reasons)
        merged.warnings.extend(ov.warnings)
    return merged


def evaluate_feasibility(profiles: list[dict]) -> FeasibilityOverride:
    """Run all rules across multiple file profiles and return merged verdict."""
    if not profiles:
        return FeasibilityOverride()

    all_overrides: list[FeasibilityOverride] = []
    for profile in profiles:
        for rule in _ALL_RULES:
            all_overrides.append(rule(profile))

    return _merge(all_overrides)


def check_prompt_vagueness(user_prompt: str) -> FeasibilityOverride:
    override = FeasibilityOverride()
    if len(user_prompt.strip().split()) <= 2:
        override.status = FeasibilityStatus.BLOCKED
        override.blocking_reasons.append(
            "Prompt is too short — please describe what you want to do with your data"
        )
    return override
