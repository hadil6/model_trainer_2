from __future__ import annotations

import io
from typing import Any

import chardet
import polars as pl

from core.config import get_settings
from data.profilers.base import BaseProfiler

_SENTINEL_VALUES = {"n/a", "na", "null", "none", "unknown", "-", "--", "?", "nan", ""}


class CsvProfiler(BaseProfiler):
    def __init__(self) -> None:
        self.settings = get_settings()

    def _read_csv_best_effort(self, content: bytes) -> tuple[pl.DataFrame, str, str]:
        detected = chardet.detect(content) or {}
        encoding = (detected.get("encoding") or "utf-8").lower()
        delimiters = [",", ";", "\t", "|"]

        try:
            decoded = content.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            encoding = "utf-8"
            decoded = content.decode("utf-8", errors="replace")

        best: tuple[pl.DataFrame, str] | None = None
        last_error: Exception | None = None

        for delimiter in delimiters:
            try:
                dataframe = pl.read_csv(
                    io.StringIO(decoded),
                    separator=delimiter,
                    infer_schema_length=500,
                    quote_char='"',
                    ignore_errors=True,
                    truncate_ragged_lines=True,
                )
                if dataframe.width < 2:
                    continue
                if best is None or dataframe.width > best[0].width:
                    best = (dataframe, delimiter)
            except Exception as exc:
                last_error = exc

        if best is not None:
            return best[0], encoding, best[1]

        if last_error:
            raise ValueError(f"Unable to parse CSV content: {last_error}") from last_error
        raise ValueError("Unable to parse CSV content")

    def _infer_numeric_stats(self, series: pl.Series) -> dict[str, Any] | None:
        if series.dtype.is_numeric() and series.len() > 0:
            non_null = series.drop_nulls()
            if non_null.len() == 0:
                return None
            return {
                "min": non_null.min(),
                "max": non_null.max(),
                "mean": non_null.mean(),
                "std": non_null.std(),
                "q25": non_null.quantile(0.25),
                "q50": non_null.quantile(0.50),
                "q75": non_null.quantile(0.75),
            }
        return None

    def _count_outliers(self, series: pl.Series) -> int:
        if not series.dtype.is_numeric():
            return 0
        non_null = series.drop_nulls()
        if non_null.len() < 4:
            return 0
        mean = non_null.mean()
        std = non_null.std()
        if not std or std == 0:
            return 0
        return int(((non_null - mean).abs() > 3 * std).sum())

    def _detect_string_issues(self, series: pl.Series) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sentinel_count": 0,
            "has_whitespace_issues": False,
            "has_case_inconsistency": False,
            "avg_word_count": None,
        }
        if series.dtype != pl.Utf8:
            return result

        non_null = series.drop_nulls()
        if non_null.len() == 0:
            return result

        lower_stripped = non_null.str.strip_chars().str.to_lowercase()
        result["sentinel_count"] = int(lower_stripped.is_in(list(_SENTINEL_VALUES)).sum())

        stripped = non_null.str.strip_chars()
        result["has_whitespace_issues"] = bool((non_null != stripped).any())

        orig_unique = stripped.n_unique()
        lower_unique = lower_stripped.n_unique()
        result["has_case_inconsistency"] = orig_unique > lower_unique

        word_counts = non_null.str.split(" ").list.len()
        result["avg_word_count"] = round(float(word_counts.mean()), 2)

        return result

    def _build_quality_signals(
        self,
        row_count: int,
        duplicate_ratio: float,
        null_ratio_by_column: dict[str, float],
        unique_ratio_by_column: dict[str, float],
    ) -> tuple[list[str], list[str]]:
        warnings: list[str] = []
        risk_flags: list[str] = []

        if row_count == 0:
            warnings.append("csv_empty_dataset")
            risk_flags.append("empty_dataset")
            return warnings, risk_flags

        if row_count < 30:
            warnings.append(f"small_dataset_rows:{row_count}")

        if duplicate_ratio > 0:
            warnings.append(f"duplicate_rows_detected:{duplicate_ratio:.4f}")
        if duplicate_ratio >= 0.10:
            risk_flags.append("high_duplicate_row_ratio")

        for col_name, null_ratio in null_ratio_by_column.items():
            if null_ratio >= 0.10:
                warnings.append(f"missing_values:{col_name}:{null_ratio:.4f}")
            if null_ratio >= 0.30:
                risk_flags.append(f"high_missing_ratio:{col_name}")

        for col_name, unique_ratio in unique_ratio_by_column.items():
            if unique_ratio <= 0.01 and row_count >= 30:
                warnings.append(f"low_variance_column:{col_name}")
                risk_flags.append(f"near_constant_column:{col_name}")

        return sorted(set(warnings)), sorted(set(risk_flags))

    def profile(self, filename: str, content: bytes) -> dict[str, Any]:
        dataframe, encoding, delimiter = self._read_csv_best_effort(content)
        row_count = dataframe.height
        col_count = dataframe.width

        columns: list[dict[str, Any]] = []
        likely_id_columns: list[str] = []
        possible_target_candidates: list[str] = []
        null_ratio_by_column: dict[str, float] = {}
        unique_ratio_by_column: dict[str, float] = {}

        total_sentinel_values = 0
        columns_with_sentinel_values: list[str] = []
        columns_with_whitespace_issues: list[str] = []
        columns_with_case_inconsistency: list[str] = []
        short_text_columns: list[str] = []
        columns_with_outliers: list[dict[str, Any]] = []

        for col_name in dataframe.columns:
            series = dataframe[col_name]
            null_ratio = (series.null_count() / max(row_count, 1)) if row_count else 0.0
            unique = series.n_unique()
            unique_ratio = (unique / max(row_count, 1)) if row_count else 0.0
            null_ratio_by_column[col_name] = float(null_ratio)
            unique_ratio_by_column[col_name] = float(unique_ratio)

            numeric_stats = self._infer_numeric_stats(series)
            string_issues = self._detect_string_issues(series)

            if string_issues["sentinel_count"] > 0:
                total_sentinel_values += string_issues["sentinel_count"]
                columns_with_sentinel_values.append(col_name)
            if string_issues["has_whitespace_issues"]:
                columns_with_whitespace_issues.append(col_name)
            if string_issues["has_case_inconsistency"]:
                columns_with_case_inconsistency.append(col_name)
            avg_wc = string_issues["avg_word_count"]
            if avg_wc is not None and avg_wc < 5:
                short_text_columns.append(col_name)

            outlier_count = self._count_outliers(series)
            if outlier_count > 0:
                columns_with_outliers.append({"column": col_name, "outlier_count": outlier_count})

            if (
                unique_ratio >= 0.98
                and null_ratio <= 0.05
                and ("id" in col_name.lower() or series.dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32, pl.Utf8))
            ):
                likely_id_columns.append(col_name)

            if 1 < unique <= max(20, int(row_count * 0.2)):
                possible_target_candidates.append(col_name)

            columns.append({
                "name": col_name,
                "inferred_type": str(series.dtype),
                "null_ratio": round(float(null_ratio), 4),
                "unique_ratio": round(float(unique_ratio), 4),
                "numeric_stats": numeric_stats,
                "sentinel_count": string_issues["sentinel_count"],
                "has_whitespace_issues": string_issues["has_whitespace_issues"],
                "has_case_inconsistency": string_issues["has_case_inconsistency"],
                "avg_word_count": string_issues["avg_word_count"],
                "outlier_count": outlier_count if series.dtype.is_numeric() else None,
            })

        total_missing_values = sum(int(dataframe[col].null_count()) for col in dataframe.columns)

        duplicate_ratio = 0.0
        duplicate_row_count = 0
        if row_count:
            duplicate_row_count = int(dataframe.is_duplicated().sum())
            duplicate_ratio = duplicate_row_count / row_count

        warnings, risk_flags = self._build_quality_signals(
            row_count=row_count,
            duplicate_ratio=float(duplicate_ratio),
            null_ratio_by_column=null_ratio_by_column,
            unique_ratio_by_column=unique_ratio_by_column,
        )

        if total_sentinel_values > 0:
            warnings.append(f"sentinel_values_detected:{total_sentinel_values}")
        for col_name in columns_with_whitespace_issues:
            warnings.append(f"whitespace_issues:{col_name}")
        for col_name in columns_with_case_inconsistency:
            warnings.append(f"case_inconsistency:{col_name}")
        for item in columns_with_outliers:
            warnings.append(f"outliers_detected:{item['column']}:{item['outlier_count']}")

        warnings = sorted(set(warnings))

        sample_size = min(300, row_count)
        sample_rows: list[dict[str, Any]] = []
        if sample_size > 0:
            sample_rows = dataframe.sample(
                n=sample_size, seed=self.settings.sample_seed, shuffle=True,
            ).to_dicts()

        cleaned_rows = self._clean_rows(sample_rows)

        column_summaries: list[dict[str, Any]] = []
        for col in columns:
            summary: dict[str, Any] = {
                "name": col["name"],
                "type": col["inferred_type"],
                "null_ratio": col["null_ratio"],
            }
            if col["numeric_stats"]:
                s = col["numeric_stats"]
                summary["range"] = f"{s['min']} – {s['max']}"
                summary["mean"] = round(s["mean"], 2) if s["mean"] is not None else None
                summary["std"] = round(s["std"], 2) if s["std"] is not None else None
            else:
                try:
                    top = (
                        dataframe[col["name"]]
                        .drop_nulls()
                        .value_counts(sort=True)
                        .head(5)
                        .to_series(0)
                        .to_list()
                    )
                    summary["top_values"] = [str(v) for v in top]
                except Exception:
                    pass
            column_summaries.append(summary)

        return {
            "file_kind": "csv",
            "metadata": {
                "filename": filename,
                "size_bytes": len(content),
                "row_count": row_count,
                "column_count": col_count,
                "delimiter": delimiter,
                "encoding": encoding,
            },
            "quality_checks": {
                "total_missing_values": total_missing_values,
                "total_sentinel_values": total_sentinel_values,
                "duplicate_row_count": duplicate_row_count,
                "duplicate_row_ratio": round(float(duplicate_ratio), 4),
                "columns_with_sentinel_values": columns_with_sentinel_values,
                "columns_with_whitespace_issues": columns_with_whitespace_issues,
                "columns_with_case_inconsistency": columns_with_case_inconsistency,
                "short_text_columns": short_text_columns,
                "columns_with_outliers": columns_with_outliers,
                "likely_id_columns": likely_id_columns,
                "possible_target_candidates": sorted(set(possible_target_candidates)),
                "high_null_columns": sorted(
                    [name for name, ratio in null_ratio_by_column.items() if ratio >= 0.30]
                ),
                "near_constant_columns": sorted(
                    [name for name, ratio in unique_ratio_by_column.items() if ratio <= 0.01 and row_count >= 30]
                ),
            },
            "columns": columns,
            "column_summaries": column_summaries,
            "samples": {
                "random_rows_seed": self.settings.sample_seed,
                "rows": sample_rows,
                "cleaned_rows": cleaned_rows,
            },
            "warnings": warnings,
            "risk_flags": risk_flags,
        }

    @staticmethod
    def _clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sentinel_values = {"n/a", "na", "null", "none", "unknown", "-", "--", "?", "nan", ""}
        cleaned = []
        for row in rows:
            clean_row: dict[str, Any] = {}
            for k, v in row.items():
                if isinstance(v, str):
                    v = v.strip()
                    if v.lower() in sentinel_values:
                        v = None
                clean_row[k] = v
            cleaned.append(clean_row)
        return cleaned
