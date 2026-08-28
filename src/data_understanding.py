"""
data_understanding.py

Documents the structure and quality of the training set.

Run:
    python src/data_understanding.py
"""

from __future__ import annotations

from pathlib import Path
from sys import path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
path.insert(0, str(PROJECT_ROOT))

from src.semantic_schema import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    SEMANTIC_SCHEMA,
    TARGET_COLUMN,
)

TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "data_understanding"


def variable_summary(data):
    rows = []
    for column in data.columns:
        series = data[column]
        row = {
            "column_name": column,
            "role": SEMANTIC_SCHEMA.get(column, {}).get("role", ""),
            "semantic_type": SEMANTIC_SCHEMA.get(column, {}).get("semantic_type", ""),
            "pandas_dtype": str(series.dtype),
            "observations": len(data),
            "non_missing_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": float(series.isna().mean() * 100),
            "unique_count": int(series.nunique(dropna=True)),
            "example_values": " | ".join(
                series.dropna().astype(str).drop_duplicates().head(5).tolist()
            ),
        }

        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric = pd.to_numeric(series, errors="coerce")
            row.update({
                "minimum": float(numeric.min()),
                "maximum": float(numeric.max()),
                "mean": float(numeric.mean()),
                "median": float(numeric.median()),
                "standard_deviation": float(numeric.std(ddof=1)),
                "skewness": float(numeric.skew()),
                "kurtosis": float(numeric.kurt()),
            })
        else:
            row.update({
                "minimum": np.nan,
                "maximum": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "standard_deviation": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
            })

        rows.append(row)

    return pd.DataFrame(rows)


def missing_summary(data):
    rows = []
    for column in data.columns:
        count = int(data[column].isna().sum())
        if count == 0:
            continue
        rules = SEMANTIC_SCHEMA.get(column, {})
        rows.append({
            "column_name": column,
            "missing_count": count,
            "missing_percentage": float(count / len(data) * 100),
            "nullable_in_schema": bool(rules.get("nullable", False)),
            "documented_treatment": rules.get("missing_treatment", ""),
        })
    return pd.DataFrame(rows)


def categorical_frequencies(data):
    rows = []
    for column in CATEGORICAL_COLUMNS:
        counts = (
            data[column]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts(dropna=False)
        )
        for level, count in counts.items():
            rows.append({
                "column_name": column,
                "category_level": str(level),
                "frequency": int(count),
                "percentage": float(count / len(data) * 100),
            })
    return pd.DataFrame(rows)


def logical_checks(data):
    rows = []

    if {"households", "population"}.issubset(data.columns):
        count = int((data["households"] > data["population"]).sum())
        rows.append({
            "check_name": "households_above_population",
            "affected_observations": count,
            "status": "PASS" if count == 0 else "REVIEW",
            "description": "Household count should not normally exceed population.",
        })

    if {"total_bedrooms", "total_rooms"}.issubset(data.columns):
        count = int(
            (
                data["total_bedrooms"].notna()
                & (data["total_bedrooms"] > data["total_rooms"])
            ).sum()
        )
        rows.append({
            "check_name": "bedrooms_above_rooms",
            "affected_observations": count,
            "status": "PASS" if count == 0 else "REVIEW",
            "description": "Bedrooms should not normally exceed total rooms.",
        })

    for column, rules in SEMANTIC_SCHEMA.items():
        if column not in data.columns:
            continue
        if not pd.api.types.is_numeric_dtype(data[column].dtype):
            continue

        numeric = pd.to_numeric(data[column], errors="coerce")

        minimum = rules.get("minimum")
        if minimum is not None:
            count = int((numeric.notna() & (numeric < minimum)).sum())
            rows.append({
                "check_name": f"{column}_below_minimum",
                "affected_observations": count,
                "status": "PASS" if count == 0 else "REVIEW",
                "description": f"Values below schema minimum {minimum}.",
            })

        maximum = rules.get("maximum")
        if maximum is not None:
            count = int((numeric.notna() & (numeric > maximum)).sum())
            rows.append({
                "check_name": f"{column}_above_maximum",
                "affected_observations": count,
                "status": "PASS" if count == 0 else "REVIEW",
                "description": f"Values above schema maximum {maximum}.",
            })

        if rules.get("whole_number") is True:
            count = int(
                (
                    numeric.notna()
                    & ~np.isclose(numeric, np.round(numeric))
                ).sum()
            )
            rows.append({
                "check_name": f"{column}_non_whole_values",
                "affected_observations": count,
                "status": "PASS" if count == 0 else "REVIEW",
                "description": "Checks the whole-number requirement.",
            })

    return pd.DataFrame(rows)


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TRAIN_PATH)

    summary = pd.DataFrame([{
        "dataset": "Training",
        "observations": len(data),
        "variables": len(data.columns),
        "numerical_predictors": len(NUMERICAL_COLUMNS),
        "categorical_predictors": len(CATEGORICAL_COLUMNS),
        "target_column": TARGET_COLUMN,
        "total_missing_values": int(data.isna().sum().sum()),
        "variables_with_missing_values": int((data.isna().sum() > 0).sum()),
        "duplicate_rows": int(data.duplicated().sum()),
    }])

    variables = variable_summary(data)
    missing = missing_summary(data)
    categories = categorical_frequencies(data)
    checks = logical_checks(data)

    summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    variables.to_csv(OUTPUT_DIR / "variable_summary.csv", index=False)
    missing.to_csv(OUTPUT_DIR / "missing_values.csv", index=False)
    categories.to_csv(OUTPUT_DIR / "categorical_frequencies.csv", index=False)
    checks.to_csv(OUTPUT_DIR / "logical_consistency_checks.csv", index=False)

    recommendations = []
    if missing.empty:
        recommendations.append("No missing values were detected.")
    else:
        recommendations.append(
            "Missing values are present in: "
            + ", ".join(missing["column_name"].tolist())
            + ". Treat them only after the training split."
        )

    review_checks = checks.loc[
        checks["status"] == "REVIEW",
        "check_name",
    ].tolist()

    if review_checks:
        recommendations.append(
            "Logical checks requiring review: "
            + ", ".join(review_checks)
            + "."
        )
    else:
        recommendations.append("All implemented logical checks passed.")

    recommendations.append("No data were modified during data understanding.")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Data Understanding</title>
<style>
body{{font-family:Arial;background:#f4f6f8;margin:0;color:#1f2933}}
.page{{max-width:1450px;margin:auto;padding:30px}}
.hero{{background:linear-gradient(135deg,#17324d,#2b7a78);color:white;padding:30px;border-radius:15px}}
section{{background:white;padding:22px;margin-top:20px;border-radius:12px}}
.wrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:.86rem}}
th{{background:#17324d;color:white;padding:8px;text-align:left}}
td{{padding:8px;border-bottom:1px solid #ddd}}
</style></head><body><div class="page">
<div class="hero"><h1>Training Data Understanding Report</h1></div>
<section><h2>Dataset Summary</h2><div class="wrap">{summary.to_html(index=False,border=0)}</div></section>
<section><h2>Variable Summary</h2><div class="wrap">{variables.to_html(index=False,border=0)}</div></section>
<section><h2>Missing Values</h2><div class="wrap">{missing.to_html(index=False,border=0) if not missing.empty else "<p>No missing values.</p>"}</div></section>
<section><h2>Categorical Frequencies</h2><div class="wrap">{categories.to_html(index=False,border=0)}</div></section>
<section><h2>Logical Checks</h2><div class="wrap">{checks.to_html(index=False,border=0)}</div></section>
<section><h2>Recommendations</h2><ul>{''.join(f'<li>{x}</li>' for x in recommendations)}</ul></section>
</div></body></html>"""

    (OUTPUT_DIR / "data_understanding_report.html").write_text(
        html,
        encoding="utf-8",
    )

    text = """DATA UNDERSTANDING SUMMARY
======================================================================
"""
    text += f"Observations: {len(data)}\n"
    text += f"Variables: {len(data.columns)}\n"
    text += f"Missing values: {int(data.isna().sum().sum())}\n"
    text += f"Duplicate rows: {int(data.duplicated().sum())}\n"
    text += "Recommendations:\n- " + "\n- ".join(recommendations)
    text += f"\n\nOutputs: {OUTPUT_DIR}\n"
    text += "No observations or values were modified.\n"
    text += "======================================================================\n"

    (OUTPUT_DIR / "data_understanding_summary.txt").write_text(
        text,
        encoding="utf-8",
    )

    print(text)
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
