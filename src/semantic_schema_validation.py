"""
Validate observed data against semantic_schema.py.
"""

from __future__ import annotations

from pathlib import Path
from sys import path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
path.insert(0, str(PROJECT_ROOT))

from src.semantic_schema import SEMANTIC_SCHEMA, schema_columns

INPUTS = [
    PROJECT_ROOT / "data" / "raw" / "housing.csv",
    PROJECT_ROOT / "data" / "housing.csv",
]
OUTPUT_DIR = PROJECT_ROOT / "results" / "data_validation"


def find_input():
    for file_path in INPUTS:
        if file_path.exists():
            return file_path
    raise FileNotFoundError("Input CSV not found.")


def add_issue(issues, severity, rule, column, count, message):
    issues.append(
        {
            "severity": severity,
            "rule": rule,
            "column": column or "",
            "affected_count": int(count),
            "message": message,
        }
    )


def validate_data(data):
    issues = []
    observations = []

    expected = schema_columns()
    observed = list(data.columns)

    missing = [c for c in expected if c not in observed]
    unexpected = [c for c in observed if c not in expected]

    if missing:
        add_issue(
            issues,
            "ERROR",
            "required_columns",
            None,
            len(missing),
            "Missing columns: " + ", ".join(missing),
        )

    if unexpected:
        add_issue(
            issues,
            "WARNING",
            "unexpected_columns",
            None,
            len(unexpected),
            "Unexpected columns: " + ", ".join(unexpected),
        )

    for column, rules in SEMANTIC_SCHEMA.items():
        if column not in data.columns:
            continue

        series = data[column]
        expected_type = rules.get("expected_storage_type")

        if expected_type == "numeric":
            if not pd.api.types.is_numeric_dtype(series.dtype):
                add_issue(
                    issues,
                    "ERROR",
                    "storage_type",
                    column,
                    len(series),
                    f"Expected numeric; observed {series.dtype}.",
                )

        elif expected_type == "string":
            is_string_like = (
                pd.api.types.is_object_dtype(series.dtype)
                or pd.api.types.is_string_dtype(series.dtype)
                or isinstance(series.dtype, pd.CategoricalDtype)
            )
            if not is_string_like:
                add_issue(
                    issues,
                    "ERROR",
                    "storage_type",
                    column,
                    len(series),
                    f"Expected string; observed {series.dtype}.",
                )

        missing_count = int(series.isna().sum())
        if missing_count:
            severity = (
                "WARNING"
                if rules.get("nullable", False)
                else "ERROR"
            )
            add_issue(
                issues,
                severity,
                "missing_values",
                column,
                missing_count,
                (
                    "Missing values allowed with documented treatment."
                    if severity == "WARNING"
                    else "Missing values found in non-nullable column."
                ),
            )

        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric = pd.to_numeric(series, errors="coerce")

            minimum = rules.get("minimum")
            if minimum is not None:
                mask = numeric.notna() & (numeric < minimum)
                if int(mask.sum()):
                    add_issue(
                        issues,
                        "ERROR",
                        "minimum_bound",
                        column,
                        int(mask.sum()),
                        f"Values below minimum {minimum}.",
                    )
                    flagged = data.loc[mask].copy()
                    flagged.insert(0, "validation_rule", "minimum_bound")
                    flagged.insert(1, "validation_column", column)
                    observations.append(flagged)

            maximum = rules.get("maximum")
            if maximum is not None:
                mask = numeric.notna() & (numeric > maximum)
                if int(mask.sum()):
                    add_issue(
                        issues,
                        "ERROR",
                        "maximum_bound",
                        column,
                        int(mask.sum()),
                        f"Values above maximum {maximum}.",
                    )

            if rules.get("whole_number") is True:
                mask = (
                    numeric.notna()
                    & ~np.isclose(numeric, np.round(numeric))
                )
                if int(mask.sum()):
                    add_issue(
                        issues,
                        "ERROR",
                        "whole_number_requirement",
                        column,
                        int(mask.sum()),
                        "Non-whole-number values detected.",
                    )

        allowed = rules.get("allowed_values")
        if allowed:
            mask = (
                series.notna()
                & ~series.astype(str).isin(allowed)
            )
            if int(mask.sum()):
                invalid = sorted(
                    series.loc[mask].astype(str).unique()
                )
                add_issue(
                    issues,
                    "ERROR",
                    "allowed_values",
                    column,
                    int(mask.sum()),
                    "Unexpected values: " + ", ".join(invalid),
                )

    duplicates = int(data.duplicated().sum())
    if duplicates:
        add_issue(
            issues,
            "WARNING",
            "duplicate_rows",
            None,
            duplicates,
            "Duplicate rows detected.",
        )

    issues_df = pd.DataFrame(
        issues,
        columns=[
            "severity",
            "rule",
            "column",
            "affected_count",
            "message",
        ],
    )

    if observations:
        observations_df = pd.concat(
            observations,
            ignore_index=False,
        )
    else:
        observations_df = pd.DataFrame(
            columns=[
                "validation_rule",
                "validation_column",
                *data.columns,
            ]
        )

    return issues_df, observations_df


def main():
    input_file = find_input()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_file)
    issues, observations = validate_data(data)

    errors = (
        int((issues["severity"] == "ERROR").sum())
        if not issues.empty
        else 0
    )
    warnings = (
        int((issues["severity"] == "WARNING").sum())
        if not issues.empty
        else 0
    )

    status = (
        "FAILED"
        if errors
        else "PASSED WITH WARNINGS"
        if warnings
        else "PASSED"
    )

    summary = pd.DataFrame(
        [
            {
                "observations": len(data),
                "columns": len(data.columns),
                "validation_errors": errors,
                "validation_warnings": warnings,
                "duplicate_rows": int(data.duplicated().sum()),
                "total_missing_values": int(data.isna().sum().sum()),
                "status": status,
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR / "validation_summary.csv",
        index=False,
    )
    issues.to_csv(
        OUTPUT_DIR / "validation_issues.csv",
        index=False,
    )
    observations.to_csv(
        OUTPUT_DIR / "validation_observations.csv",
        index=True,
    )
    (OUTPUT_DIR / "validation_status.txt").write_text(
        f"Validation status: {status}\n",
        encoding="utf-8",
    )

    print("=" * 90)
    print("SEMANTIC SCHEMA VALIDATION")
    print("=" * 90)
    print(summary.to_string(index=False))
    print("\nIssues:")
    print(
        "No issues."
        if issues.empty
        else issues.to_string(index=False)
    )

    if status == "FAILED":
        raise SystemExit(
            "Validation failed. Review validation_issues.csv."
        )


if __name__ == "__main__":
    main()
