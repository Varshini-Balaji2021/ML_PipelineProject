"""
Dataset-agnostic structural profiler.

Run:
    python src/data_profile.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    PROJECT_ROOT / "data" / "raw" / "housing.csv",
    PROJECT_ROOT / "data" / "housing.csv",
]
OUTPUT_DIR = PROJECT_ROOT / "results" / "data_profile"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    return parser.parse_args()


def resolve_input(value):
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    for path in DEFAULT_INPUTS:
        if path.exists():
            return path

    raise FileNotFoundError("No default input CSV found.")


def pct(n, d):
    return 0.0 if d == 0 else n / d * 100


def looks_datetime(series):
    if not (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    ):
        return False

    sample = series.dropna().astype(str).head(100)
    if sample.empty:
        return False

    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.90


def infer_type(series):
    if pd.api.types.is_bool_dtype(series.dtype):
        return "Boolean"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "Datetime"
    if looks_datetime(series):
        return "Datetime candidate"
    if pd.api.types.is_integer_dtype(series.dtype):
        return "Integer numeric"
    if pd.api.types.is_float_dtype(series.dtype):
        non_missing = series.dropna()
        if (
            not non_missing.empty
            and np.allclose(non_missing, np.round(non_missing))
        ):
            return "Integer-valued numeric"
        return "Continuous numeric"
    if (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return "String / categorical candidate"
    return "Other"


def profile_column(name, series, row_count, position):
    non_missing = series.dropna()
    unique_count = int(non_missing.nunique())
    unique_ratio = (
        unique_count / len(non_missing)
        if len(non_missing)
        else 0.0
    )

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_non_missing = numeric.dropna()

    is_numeric = pd.api.types.is_numeric_dtype(series.dtype)

    minimum = (
        float(numeric_non_missing.min())
        if is_numeric and not numeric_non_missing.empty
        else np.nan
    )
    maximum = (
        float(numeric_non_missing.max())
        if is_numeric and not numeric_non_missing.empty
        else np.nan
    )
    mean = (
        float(numeric_non_missing.mean())
        if is_numeric and not numeric_non_missing.empty
        else np.nan
    )
    median = (
        float(numeric_non_missing.median())
        if is_numeric and not numeric_non_missing.empty
        else np.nan
    )
    std = (
        float(numeric_non_missing.std(ddof=1))
        if is_numeric and not numeric_non_missing.empty
        else np.nan
    )

    return {
        "column_position": position,
        "column_name": name,
        "observed_pandas_dtype": str(series.dtype),
        "inferred_structural_type": infer_type(series),
        "observations": row_count,
        "non_missing_count": int(series.notna().sum()),
        "missing_count": int(series.isna().sum()),
        "missing_percentage": pct(int(series.isna().sum()), row_count),
        "unique_count": unique_count,
        "unique_percentage": pct(unique_count, row_count),
        "binary_candidate": unique_count == 2,
        "low_cardinality_candidate": (
            unique_count > 0
            and (unique_count <= 20 or unique_ratio <= 0.05)
        ),
        "identifier_candidate": (
            len(non_missing) > 0 and unique_ratio >= 0.98
        ),
        "integer_valued_candidate": (
            bool(np.allclose(numeric_non_missing, np.round(numeric_non_missing)))
            if is_numeric and not numeric_non_missing.empty
            else False
        ),
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "median": median,
        "standard_deviation": std,
        "zero_count": (
            int((numeric_non_missing == 0).sum())
            if is_numeric
            else np.nan
        ),
        "negative_count": (
            int((numeric_non_missing < 0).sum())
            if is_numeric
            else np.nan
        ),
        "example_values": " | ".join(
            non_missing.astype(str).drop_duplicates().head(5).tolist()
        ),
    }


def main():
    args = parse_args()
    input_file = resolve_input(args.input)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_file)

    profile = pd.DataFrame(
        [
            profile_column(
                column,
                data[column],
                len(data),
                position,
            )
            for position, column in enumerate(data.columns, start=1)
        ]
    )

    candidate_rows = []
    for column in profile.loc[
        profile["low_cardinality_candidate"],
        "column_name",
    ]:
        counts = (
            data[column]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts(dropna=False)
        )
        for value, count in counts.items():
            candidate_rows.append(
                {
                    "column_name": column,
                    "observed_value": str(value),
                    "frequency": int(count),
                    "percentage": pct(int(count), len(data)),
                }
            )

    candidates = pd.DataFrame(candidate_rows)

    profile.to_csv(
        OUTPUT_DIR / "data_profile.csv",
        index=False,
    )
    candidates.to_csv(
        OUTPUT_DIR / "categorical_candidates.csv",
        index=False,
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Data Profile</title>
<style>
body{{font-family:Arial;margin:30px;background:#f5f7f9}}
section{{background:white;padding:22px;margin-bottom:20px;border-radius:12px}}
table{{border-collapse:collapse;width:100%;font-size:.86rem}}
th{{background:#17324d;color:white;padding:8px;text-align:left}}
td{{padding:8px;border-bottom:1px solid #ddd}}
.wrap{{overflow-x:auto}}
</style></head><body>
<section><h1>Structural Data Profile</h1>
<p><strong>Input:</strong> {input_file}</p>
<p>This report contains observed structural information only.</p></section>
<section><h2>Column Profile</h2><div class="wrap">
{profile.to_html(index=False, border=0)}
</div></section>
<section><h2>Low-Cardinality Candidates</h2><div class="wrap">
{candidates.to_html(index=False, border=0) if not candidates.empty else "<p>None detected.</p>"}
</div></section>
</body></html>"""

    (OUTPUT_DIR / "data_profile.html").write_text(
        html,
        encoding="utf-8",
    )

    summary = f"""STRUCTURAL DATA PROFILE
Input: {input_file}
Rows: {len(data)}
Columns: {len(data.columns)}
Missing values: {int(data.isna().sum().sum())}
Duplicate rows: {int(data.duplicated().sum())}
No semantic assumptions were applied.
"""
    (OUTPUT_DIR / "profile_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)
    print(
        profile[
            [
                "column_name",
                "observed_pandas_dtype",
                "inferred_structural_type",
                "missing_count",
                "unique_count",
                "minimum",
                "maximum",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
