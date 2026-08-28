"""
preprocess.py

PURPOSE
-------
Create the cleaned, schema-aligned housing dataset after:

    ingest.py
        ↓
    data_profile.py
        ↓
    semantic_schema.py
        ↓
    semantic_schema_validation.py
        ↓
    preprocess.py

This preprocessing stage is schema-driven. It does not hard-code:

- categorical columns;
- numerical columns;
- non-negative columns;
- target columns;
- allowed categories;
- nullable columns.

All such information is imported from semantic_schema.py.

RESPONSIBILITIES
----------------
The script:

1. Confirms that semantic-schema validation has not failed.
2. Loads the raw dataset.
3. Retains columns defined in the approved semantic schema.
4. Reorders columns according to the schema.
5. Coerces numerical columns to numeric storage.
6. Standardises categorical text representation.
7. Preserves allowed missing values for later model-pipeline imputation.
8. Saves a clean, deterministic processed dataset.
9. Saves a preprocessing report.

The script does not:

- fit an imputer;
- impute missing values;
- scale numerical variables;
- one-hot encode categorical variables;
- winsorize observations;
- remove outliers;
- split the dataset;
- fit a statistical or machine-learning model.

INPUT
-----
    data/raw/housing.csv

OUTPUT
------
    data/processed/housing_clean.csv

REPORTS
-------
    results/preprocessing/
        preprocessing_report.csv
        preprocessing_summary.txt

RUN
---
From the HousingPipeline project root:

    python src/preprocess.py
"""

from __future__ import annotations

from pathlib import Path
from sys import path
from typing import Any

import numpy as np
import pandas as pd


# =========================================================
# 1. Project paths and schema import
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
path.insert(0, str(PROJECT_ROOT))

from src.semantic_schema import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    SEMANTIC_SCHEMA,
    TARGET_COLUMN,
    schema_columns,
)


RAW_DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "housing.csv",
    PROJECT_ROOT / "data" / "housing.csv",
]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "housing_clean.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "preprocessing"
)

PREPROCESSING_REPORT_PATH = (
    RESULTS_DIR
    / "preprocessing_report.csv"
)

PREPROCESSING_SUMMARY_PATH = (
    RESULTS_DIR
    / "preprocessing_summary.txt"
)

VALIDATION_SUMMARY_CANDIDATES = [
    (
        PROJECT_ROOT
        / "results"
        / "semantic_schema_validation"
        / "validation_summary.csv"
    ),
    (
        PROJECT_ROOT
        / "results"
        / "data_validation"
        / "validation_summary.csv"
    ),
]


# =========================================================
# 2. File-resolution helpers
# =========================================================

def find_raw_data_file() -> Path:
    """Return the first available raw-data file."""

    for candidate in RAW_DATA_CANDIDATES:
        if candidate.exists():
            return candidate

    checked = "\n".join(
        f"  - {candidate}"
        for candidate in RAW_DATA_CANDIDATES
    )

    raise FileNotFoundError(
        "Raw housing data were not found.\n"
        f"Checked:\n{checked}"
    )


def find_validation_summary() -> Path:
    """Return the first available validation-summary file."""

    for candidate in VALIDATION_SUMMARY_CANDIDATES:
        if candidate.exists():
            return candidate

    checked = "\n".join(
        f"  - {candidate}"
        for candidate in VALIDATION_SUMMARY_CANDIDATES
    )

    raise FileNotFoundError(
        "Semantic-schema validation summary was not found.\n"
        "Run semantic_schema_validation.py before preprocess.py.\n"
        f"Checked:\n{checked}"
    )


# =========================================================
# 3. Validation gate
# =========================================================

def read_validation_status(
    validation_summary_path: Path,
) -> str:
    """Read the overall semantic-schema validation status."""

    summary = pd.read_csv(
        validation_summary_path
    )

    if summary.empty:
        raise ValueError(
            "Validation summary is empty: "
            f"{validation_summary_path}"
        )

    if "status" not in summary.columns:
        raise KeyError(
            "Validation summary does not contain a "
            "'status' column."
        )

    return str(
        summary.iloc[0]["status"]
    ).strip().upper()


def enforce_validation_gate(
    validation_summary_path: Path,
) -> str:
    """
    Stop preprocessing when semantic-schema validation failed.

    PASSED and PASSED WITH WARNINGS are permitted.
    """

    status = read_validation_status(
        validation_summary_path
    )

    accepted_statuses = {
        "PASSED",
        "PASSED WITH WARNINGS",
    }

    if status not in accepted_statuses:
        raise RuntimeError(
            "Preprocessing cannot continue because semantic-schema "
            f"validation status is '{status}'. Review the validation "
            "outputs and correct the raw data or schema first."
        )

    return status


# =========================================================
# 4. Schema-driven preprocessing helpers
# =========================================================

def expected_columns() -> list[str]:
    """Return schema-approved columns in deterministic order."""

    return schema_columns()


def require_schema_columns(
    data: pd.DataFrame,
) -> None:
    """Ensure every schema-defined column is available."""

    missing_columns = [
        column
        for column in expected_columns()
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            "Preprocessing cannot continue. Missing schema columns: "
            + ", ".join(
                missing_columns
            )
        )


def select_and_order_columns(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retain only schema-approved columns and apply schema order.

    Unexpected columns are excluded from the processed dataset but
    reported explicitly.
    """

    approved_columns = expected_columns()

    unexpected_columns = [
        column
        for column in data.columns
        if column not in approved_columns
    ]

    clean = data.loc[
        :,
        approved_columns,
    ].copy()

    return clean, unexpected_columns


def coerce_numeric_columns(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    """
    Coerce schema-defined numerical and target columns to numeric.

    Counts newly introduced missing values so conversion problems
    cannot pass silently.
    """

    clean = data.copy()

    target_columns = [
        column
        for column, rules in SEMANTIC_SCHEMA.items()
        if rules.get(
            "preprocessing_group"
        ) == "target"
    ]

    numeric_columns = list(
        dict.fromkeys(
            NUMERICAL_COLUMNS
            + target_columns
        )
    )

    newly_missing: dict[
        str,
        int
    ] = {}

    for column in numeric_columns:
        before_missing = int(
            clean[column].isna().sum()
        )

        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce",
        )

        after_missing = int(
            clean[column].isna().sum()
        )

        newly_missing[column] = (
            after_missing
            - before_missing
        )

    conversion_failures = {
        column: count
        for column, count in newly_missing.items()
        if count > 0
    }

    if conversion_failures:
        details = "; ".join(
            f"{column}={count}"
            for column, count in conversion_failures.items()
        )

        raise ValueError(
            "Numeric coercion introduced missing values, indicating "
            f"invalid numeric content: {details}"
        )

    return clean, newly_missing


def standardise_categorical_columns(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    """
    Standardise categorical text without changing category meaning.

    Operations:
    - preserve missing values;
    - convert non-missing values to strings;
    - remove leading and trailing whitespace;
    - reject empty strings created by cleaning.
    """

    clean = data.copy()

    changed_counts: dict[
        str,
        int
    ] = {}

    for column in CATEGORICAL_COLUMNS:
        original = clean[column].copy()

        missing_mask = original.isna()

        cleaned = (
            original.astype("string")
            .str.strip()
        )

        cleaned = cleaned.mask(
            missing_mask,
            pd.NA,
        )

        empty_mask = (
            cleaned.notna()
            & cleaned.eq("")
        )

        if int(empty_mask.sum()) > 0:
            raise ValueError(
                f"Categorical column '{column}' contains "
                f"{int(empty_mask.sum())} empty values after trimming."
            )

        changed = (
            original.astype("string")
            .fillna("<MISSING>")
            .ne(
                cleaned.astype("string")
                .fillna("<MISSING>")
            )
        )

        changed_counts[column] = int(
            changed.sum()
        )

        clean[column] = cleaned

    return clean, changed_counts


def enforce_allowed_categories(
    data: pd.DataFrame,
) -> None:
    """
    Confirm that cleaned categorical values remain schema-compliant.

    This is a defensive check. The principal validation should already
    have occurred in semantic_schema_validation.py.
    """

    for column in CATEGORICAL_COLUMNS:
        allowed_values = (
            SEMANTIC_SCHEMA[
                column
            ].get(
                "allowed_values"
            )
        )

        if not allowed_values:
            continue

        observed_values = set(
            data[column]
            .dropna()
            .astype(str)
            .unique()
        )

        unexpected_values = sorted(
            observed_values
            - set(
                map(
                    str,
                    allowed_values,
                )
            )
        )

        if unexpected_values:
            raise ValueError(
                f"Column '{column}' contains categories not permitted "
                "by semantic_schema.py: "
                + ", ".join(
                    unexpected_values
                )
            )


def enforce_nullable_policy(
    data: pd.DataFrame,
) -> None:
    """Confirm that non-nullable columns contain no missing values."""

    violations = {}

    for column, rules in (
        SEMANTIC_SCHEMA.items()
    ):
        nullable = bool(
            rules.get(
                "nullable",
                False,
            )
        )

        missing_count = int(
            data[column].isna().sum()
        )

        if (
            not nullable
            and missing_count > 0
        ):
            violations[column] = (
                missing_count
            )

    if violations:
        details = "; ".join(
            f"{column}={count}"
            for column, count in violations.items()
        )

        raise ValueError(
            "Non-nullable columns contain missing values: "
            + details
        )


def enforce_numeric_bounds(
    data: pd.DataFrame,
) -> None:
    """
    Defensively verify schema-defined numerical bounds.

    Bounds are not hard-coded here; they are read from SEMANTIC_SCHEMA.
    """

    violations = []

    for column, rules in (
        SEMANTIC_SCHEMA.items()
    ):
        if not pd.api.types.is_numeric_dtype(
            data[column].dtype
        ):
            continue

        minimum = rules.get(
            "minimum"
        )

        maximum = rules.get(
            "maximum"
        )

        numeric = data[column]

        if minimum is not None:
            count = int(
                (
                    numeric.notna()
                    & (
                        numeric
                        < minimum
                    )
                ).sum()
            )

            if count:
                violations.append(
                    (
                        column,
                        "minimum",
                        minimum,
                        count,
                    )
                )

        if maximum is not None:
            count = int(
                (
                    numeric.notna()
                    & (
                        numeric
                        > maximum
                    )
                ).sum()
            )

            if count:
                violations.append(
                    (
                        column,
                        "maximum",
                        maximum,
                        count,
                    )
                )

    if violations:
        details = "; ".join(
            (
                f"{column}: {count} values violate "
                f"{bound_name}={bound_value}"
            )
            for (
                column,
                bound_name,
                bound_value,
                count,
            ) in violations
        )

        raise ValueError(
            "Schema-bound violations remain during preprocessing: "
            + details
        )


def enforce_whole_number_policy(
    data: pd.DataFrame,
) -> None:
    """Verify schema-defined whole-number requirements."""

    violations = {}

    for column, rules in (
        SEMANTIC_SCHEMA.items()
    ):
        if rules.get(
            "whole_number"
        ) is not True:
            continue

        numeric = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        mask = (
            numeric.notna()
            & ~np.isclose(
                numeric,
                np.round(
                    numeric
                ),
            )
        )

        count = int(
            mask.sum()
        )

        if count:
            violations[column] = count

    if violations:
        details = "; ".join(
            f"{column}={count}"
            for column, count in violations.items()
        )

        raise ValueError(
            "Whole-number requirements were violated: "
            + details
        )


def apply_schema_dtypes(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply stable pandas storage types after validation.

    Nullable integer storage is used for whole-number variables so
    permitted missing values can be preserved.
    """

    clean = data.copy()

    for column, rules in (
        SEMANTIC_SCHEMA.items()
    ):
        expected_type = rules.get(
            "expected_storage_type"
        )

        whole_number = bool(
            rules.get(
                "whole_number",
                False,
            )
        )

        if expected_type == "numeric":
            if whole_number:
                clean[column] = (
                    pd.to_numeric(
                        clean[column],
                        errors="raise",
                    )
                    .round()
                    .astype("Int64")
                )
            else:
                clean[column] = (
                    pd.to_numeric(
                        clean[column],
                        errors="raise",
                    )
                    .astype("float64")
                )

        elif expected_type == "string":
            clean[column] = (
                clean[column]
                .astype("string")
            )

    return clean


# =========================================================
# 5. Reporting
# =========================================================

def create_preprocessing_report(
    raw_data: pd.DataFrame,
    processed_data: pd.DataFrame,
    unexpected_columns: list[str],
    categorical_changes: dict[str, int],
    validation_status: str,
) -> pd.DataFrame:
    """Create a transparent preprocessing report."""

    rows: list[
        dict[str, Any]
    ] = []

    rows.append(
        {
            "operation": (
                "semantic_validation_gate"
            ),
            "column": "",
            "affected_observations": 0,
            "decision": validation_status,
            "description": (
                "Preprocessing proceeded only after "
                "semantic-schema validation."
            ),
        }
    )

    rows.append(
        {
            "operation": (
                "schema_column_selection"
            ),
            "column": "",
            "affected_observations": len(
                unexpected_columns
            ),
            "decision": (
                "SCHEMA COLUMNS RETAINED"
            ),
            "description": (
                "Unexpected columns excluded: "
                + (
                    ", ".join(
                        unexpected_columns
                    )
                    if unexpected_columns
                    else "none"
                )
            ),
        }
    )

    for column in (
        processed_data.columns
    ):
        rows.append(
            {
                "operation": (
                    "dtype_alignment"
                ),
                "column": column,
                "affected_observations": 0,
                "decision": str(
                    processed_data[
                        column
                    ].dtype
                ),
                "description": (
                    "Storage type aligned with "
                    "semantic_schema.py."
                ),
            }
        )

    for column, count in (
        categorical_changes.items()
    ):
        rows.append(
            {
                "operation": (
                    "categorical_text_standardisation"
                ),
                "column": column,
                "affected_observations": count,
                "decision": (
                    "WHITESPACE TRIMMED"
                ),
                "description": (
                    "Leading and trailing whitespace "
                    "removed from non-missing category values."
                ),
            }
        )

    rows.append(
        {
            "operation": (
                "missing_value_treatment"
            ),
            "column": "",
            "affected_observations": int(
                processed_data.isna()
                .sum()
                .sum()
            ),
            "decision": (
                "PRESERVED"
            ),
            "description": (
                "Allowed missing values were retained. "
                "Model-specific imputation must be fitted "
                "using training data only."
            ),
        }
    )

    rows.append(
        {
            "operation": (
                "row_count_check"
            ),
            "column": "",
            "affected_observations": (
                len(raw_data)
                - len(processed_data)
            ),
            "decision": (
                "NO ROWS REMOVED"
                if len(raw_data)
                == len(processed_data)
                else "ROWS CHANGED"
            ),
            "description": (
                f"Raw rows={len(raw_data)}; "
                f"processed rows={len(processed_data)}."
            ),
        }
    )

    return pd.DataFrame(
        rows
    )


def create_summary_text(
    *,
    raw_path: Path,
    validation_summary_path: Path,
    validation_status: str,
    raw_data: pd.DataFrame,
    processed_data: pd.DataFrame,
    unexpected_columns: list[str],
) -> str:
    """Create the console and text summary."""

    missing_by_column = (
        processed_data.isna()
        .sum()
    )

    missing_text = "\n".join(
        (
            f"  {column}: "
            f"{int(count)}"
        )
        for column, count in (
            missing_by_column.items()
        )
        if count > 0
    )

    if not missing_text:
        missing_text = "  None"

    return f"""SCHEMA-DRIVEN PREPROCESSING SUMMARY
================================================================================

Raw input:
{raw_path}

Validation summary:
{validation_summary_path}

Validation status:
{validation_status}

Schema source:
{PROJECT_ROOT / "src" / "semantic_schema.py"}

Raw dataset:
  Rows: {len(raw_data)}
  Columns: {len(raw_data.columns)}

Processed dataset:
  Rows: {len(processed_data)}
  Columns: {len(processed_data.columns)}

Target column:
{TARGET_COLUMN}

Numerical predictor columns:
{NUMERICAL_COLUMNS}

Categorical predictor columns:
{CATEGORICAL_COLUMNS}

Unexpected raw columns excluded:
{unexpected_columns if unexpected_columns else "None"}

Missing values preserved:
{missing_text}

Output:
{PROCESSED_DATA_PATH}

No rows were removed.
No missing values were imputed.
No variables were scaled.
No categorical variables were encoded.
No outliers were removed or winsorized.
================================================================================
"""


# =========================================================
# 6. Main
# =========================================================

def main() -> None:
    """Run schema-driven deterministic preprocessing."""

    raw_data_path = (
        find_raw_data_file()
    )

    validation_summary_path = (
        find_validation_summary()
    )

    validation_status = (
        enforce_validation_gate(
            validation_summary_path
        )
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_data = pd.read_csv(
        raw_data_path
    )

    require_schema_columns(
        raw_data
    )

    processed_data, unexpected_columns = (
        select_and_order_columns(
            raw_data
        )
    )

    processed_data, _ = (
        coerce_numeric_columns(
            processed_data
        )
    )

    (
        processed_data,
        categorical_changes,
    ) = standardise_categorical_columns(
        processed_data
    )

    enforce_allowed_categories(
        processed_data
    )

    enforce_nullable_policy(
        processed_data
    )

    enforce_numeric_bounds(
        processed_data
    )

    enforce_whole_number_policy(
        processed_data
    )

    processed_data = (
        apply_schema_dtypes(
            processed_data
        )
    )

    if len(processed_data) != len(
        raw_data
    ):
        raise RuntimeError(
            "Preprocessing unexpectedly changed "
            "the number of rows."
        )

    processed_data.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
        encoding="utf-8",
    )

    report = (
        create_preprocessing_report(
            raw_data=raw_data,
            processed_data=processed_data,
            unexpected_columns=(
                unexpected_columns
            ),
            categorical_changes=(
                categorical_changes
            ),
            validation_status=(
                validation_status
            ),
        )
    )

    report.to_csv(
        PREPROCESSING_REPORT_PATH,
        index=False,
        encoding="utf-8",
    )

    summary = create_summary_text(
        raw_path=raw_data_path,
        validation_summary_path=(
            validation_summary_path
        ),
        validation_status=(
            validation_status
        ),
        raw_data=raw_data,
        processed_data=(
            processed_data
        ),
        unexpected_columns=(
            unexpected_columns
        ),
    )

    PREPROCESSING_SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)

    print(
        "\nPROCESSED COLUMN TYPES"
    )

    print(
        processed_data.dtypes
        .rename(
            "processed_dtype"
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
