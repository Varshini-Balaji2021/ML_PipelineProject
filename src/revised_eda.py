"""
eda.py

Exploratory data analysis for the training set only.

Run:
    python src/eda.py
"""

from __future__ import annotations

from pathlib import Path
from sys import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
path.insert(0, str(PROJECT_ROOT))

from src.semantic_schema import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
)

TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "eda"
FIGURES_DIR = OUTPUT_DIR / "figures"


def validate_input(data):
    expected = [
        *NUMERICAL_COLUMNS,
        *CATEGORICAL_COLUMNS,
        TARGET_COLUMN,
    ]
    missing = [c for c in expected if c not in data.columns]
    if missing:
        raise KeyError("EDA columns missing: " + ", ".join(missing))
    print("EDA input validation passed.")


def distribution_summary(data):
    rows = []
    for column in data.select_dtypes(include="number").columns:
        series = pd.to_numeric(data[column], errors="coerce")
        skew = float(series.skew())
        if abs(skew) < 0.5:
            shape = "Approximately symmetric"
        elif skew > 0:
            shape = "Right-skewed"
        else:
            shape = "Left-skewed"

        rows.append({
            "variable": column,
            "observations": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "standard_deviation": float(series.std(ddof=1)),
            "minimum": float(series.min()),
            "maximum": float(series.max()),
            "skewness": skew,
            "kurtosis": float(series.kurt()),
            "distribution_shape": shape,
        })
    return pd.DataFrame(rows)


def correlation_outputs(data):
    matrix = data.select_dtypes(include="number").corr()
    target = (
        matrix[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .rename("correlation_with_target")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    target["absolute_correlation"] = target[
        "correlation_with_target"
    ].abs()
    return matrix, target


def categorical_target_summary(data):
    rows = []
    for column in CATEGORICAL_COLUMNS:
        grouped = (
            data.groupby(
                column,
                dropna=False,
                observed=True,
            )[TARGET_COLUMN]
            .agg(
                observations="count",
                target_mean="mean",
                target_median="median",
                target_standard_deviation="std",
            )
            .reset_index()
        )
        grouped.insert(0, "categorical_variable", column)
        grouped.rename(
            columns={column: "category_level"},
            inplace=True,
        )
        rows.append(grouped)

    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame()
    )


def modified_z(series):
    numeric = pd.to_numeric(series, errors="coerce")
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(np.nan, index=series.index)
    return 0.6745 * (numeric - median) / mad


def outlier_results(data):
    summaries = []
    observations = []

    for column in data.select_dtypes(include="number").columns:
        series = pd.to_numeric(data[column], errors="coerce")
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        iqr_flag = (
            series.notna()
            & ((series < lower) | (series > upper))
        )
        mz = modified_z(series)
        mz_flag = (mz.abs() > 3.5).fillna(False)
        combined = iqr_flag | mz_flag

        summaries.append({
            "variable": column,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "iqr_lower_bound": lower,
            "iqr_upper_bound": upper,
            "iqr_flag_count": int(iqr_flag.sum()),
            "modified_z_flag_count": int(mz_flag.sum()),
            "combined_flag_count": int(combined.sum()),
            "combined_flag_percentage": float(combined.mean() * 100),
        })

        if int(combined.sum()) > 0:
            flagged = data.loc[combined].copy()
            flagged.insert(0, "outlier_variable", column)
            flagged.insert(1, "iqr_flag", iqr_flag.loc[combined].to_numpy())
            flagged.insert(2, "modified_z_flag", mz_flag.loc[combined].to_numpy())
            flagged.insert(3, "modified_z_score", mz.loc[combined].to_numpy())
            observations.append(flagged)

    summary = (
        pd.DataFrame(summaries)
        .sort_values(
            "combined_flag_percentage",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    detail = (
        pd.concat(observations, ignore_index=False)
        if observations
        else pd.DataFrame()
    )

    return summary, detail


def save_figures(data, correlation_matrix, target_correlations, categorical):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    plt.hist(
        data[TARGET_COLUMN].dropna(),
        bins=50,
        edgecolor="black",
    )
    plt.xlabel(TARGET_COLUMN)
    plt.ylabel("Frequency")
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "target_distribution.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(10, 8))
    image = plt.imshow(correlation_matrix, aspect="auto")
    plt.colorbar(image, label="Correlation")
    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=90,
    )
    plt.yticks(
        range(len(correlation_matrix.index)),
        correlation_matrix.index,
    )
    plt.title("Numerical Correlation Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "correlation_heatmap.png",
        dpi=300,
    )
    plt.close()

    for feature in target_correlations["feature"].head(5):
        plt.figure(figsize=(8, 6))
        plt.scatter(
            data[feature],
            data[TARGET_COLUMN],
            alpha=0.25,
            s=12,
        )
        plt.xlabel(feature)
        plt.ylabel(TARGET_COLUMN)
        plt.title(f"{TARGET_COLUMN} versus {feature}")
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"target_scatter_{feature}.png",
            dpi=300,
        )
        plt.close()

    for column in data.select_dtypes(include="number").columns:
        plt.figure(figsize=(8, 4))
        plt.boxplot(
            data[column].dropna(),
            orientation="horizontal",
        )
        plt.xlabel(column)
        plt.title(f"Boxplot: {column}")
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"boxplot_{column}.png",
            dpi=300,
        )
        plt.close()

    for column in CATEGORICAL_COLUMNS:
        subset = categorical.loc[
            categorical["categorical_variable"] == column
        ].sort_values("target_mean", ascending=False)

        if subset.empty:
            continue

        positions = np.arange(len(subset))
        plt.figure(figsize=(9, 6))
        plt.bar(positions, subset["target_mean"])
        plt.xticks(
            positions,
            subset["category_level"].astype(str),
            rotation=30,
            ha="right",
        )
        plt.ylabel(f"Mean {TARGET_COLUMN}")
        plt.title(f"Mean Target by {column}")
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"categorical_target_{column}.png",
            dpi=300,
        )
        plt.close()


def recommendations(data, distributions, target_corr, outliers):
    notes = []

    target_row = distributions.loc[
        distributions["variable"] == TARGET_COLUMN
    ].iloc[0]

    skewness = float(target_row["skewness"])

    if skewness > 0.5:
        notes.append(
            "The target is right-skewed. Explore transformation only "
            "if it improves validation performance or model adequacy."
        )
    elif skewness < -0.5:
        notes.append("The target is left-skewed.")
    else:
        notes.append("The target is approximately symmetric.")

    strongest = target_corr.iloc[0]
    notes.append(
        f"The strongest observed linear relationship is "
        f"{strongest['feature']} "
        f"(correlation={strongest['correlation_with_target']:.4f})."
    )

    high = outliers.loc[
        outliers["combined_flag_percentage"] >= 5
    ]

    if not high.empty:
        notes.append(
            "Potential extreme values exceed 5% in: "
            + ", ".join(high["variable"].tolist())
            + ". Retain them for predictive modelling unless domain "
            "evidence supports removal; use sensitivity analysis for inference."
        )

    missing = data.columns[data.isna().sum() > 0].tolist()
    if missing:
        notes.append(
            "Missing values remain in: "
            + ", ".join(missing)
            + ". Fit imputation on training data only."
        )

    notes.append(
        "Correlation measures association and does not establish causality."
    )

    return notes


def create_html(distributions, target_corr, categorical, outliers, notes):
    figures = []
    for figure in sorted(FIGURES_DIR.glob("*.png")):
        title = figure.stem.replace("_", " ").title()
        figures.append(
            f'<figure><img src="figures/{figure.name}" alt="{title}">'
            f"<figcaption>{title}</figcaption></figure>"
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>EDA</title>
<style>
body{{font-family:Arial;background:#f4f6f8;margin:0;color:#1f2933}}
.page{{max-width:1450px;margin:auto;padding:30px}}
.hero{{background:linear-gradient(135deg,#17324d,#2b7a78);color:white;padding:30px;border-radius:15px}}
section{{background:white;padding:22px;margin-top:20px;border-radius:12px}}
.wrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:.86rem}}
th{{background:#17324d;color:white;padding:8px;text-align:left}}
td{{padding:8px;border-bottom:1px solid #ddd}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}}
figure{{margin:0;border:1px solid #ddd;border-radius:10px;overflow:hidden}}
figure img{{width:100%;display:block}}
figcaption{{padding:10px;text-align:center;font-weight:bold}}
</style></head><body><div class="page">
<div class="hero"><h1>Exploratory Data Analysis</h1></div>
<section><h2>Numerical Distributions</h2><div class="wrap">{distributions.to_html(index=False,border=0)}</div></section>
<section><h2>Target Correlations</h2><div class="wrap">{target_corr.to_html(index=False,border=0)}</div></section>
<section><h2>Categorical Target Summary</h2><div class="wrap">{categorical.to_html(index=False,border=0)}</div></section>
<section><h2>Potential Outliers</h2><div class="wrap">{outliers.to_html(index=False,border=0)}</div></section>
<section><h2>Recommendations</h2><ul>{''.join(f'<li>{x}</li>' for x in notes)}</ul></section>
<section><h2>Figures</h2><div class="gallery">{''.join(figures)}</div></section>
</div></body></html>"""


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(TRAIN_PATH)
    validate_input(data)

    distributions = distribution_summary(data)
    matrix, target_corr = correlation_outputs(data)
    categorical = categorical_target_summary(data)
    outliers, outlier_rows = outlier_results(data)

    distributions.to_csv(
        OUTPUT_DIR / "numerical_distribution_summary.csv",
        index=False,
    )
    matrix.to_csv(
        OUTPUT_DIR / "correlation_matrix.csv",
        index=True,
    )
    target_corr.to_csv(
        OUTPUT_DIR / "target_correlations.csv",
        index=False,
    )
    categorical.to_csv(
        OUTPUT_DIR / "categorical_target_summary.csv",
        index=False,
    )
    outliers.to_csv(
        OUTPUT_DIR / "outlier_summary.csv",
        index=False,
    )
    outlier_rows.to_csv(
        OUTPUT_DIR / "outlier_observations.csv",
        index=True,
    )

    save_figures(data, matrix, target_corr, categorical)
    notes = recommendations(
        data,
        distributions,
        target_corr,
        outliers,
    )

    (OUTPUT_DIR / "eda_recommendations.txt").write_text(
        "\n".join(f"- {x}" for x in notes),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "eda_report.html").write_text(
        create_html(
            distributions,
            target_corr,
            categorical,
            outliers,
            notes,
        ),
        encoding="utf-8",
    )

    target_row = distributions.loc[
        distributions["variable"] == TARGET_COLUMN
    ].iloc[0]

    print("=" * 90)
    print("EXPLORATORY DATA ANALYSIS SUMMARY")
    print("=" * 90)
    print("Dataset used: Training set only")
    print(f"Observations: {len(data)}")
    print(f"Variables: {len(data.columns)}")
    print(f"Total missing values: {int(data.isna().sum().sum())}")
    print(f"Target mean: {target_row['mean']:,.2f}")
    print(f"Target median: {target_row['median']:,.2f}")
    print(f"Target skewness: {target_row['skewness']:.4f}")
    print(f"Target kurtosis: {target_row['kurtosis']:.4f}")
    print("\nTop correlations with target:")
    print(target_corr.head(5).to_string(index=False))
    print("\nPotential outlier summary:")
    print(
        outliers[
            [
                "variable",
                "iqr_flag_count",
                "modified_z_flag_count",
                "combined_flag_count",
                "combined_flag_percentage",
            ]
        ].to_string(index=False)
    )
    print("\nEDA outputs saved to:")
    print(OUTPUT_DIR)
    print(FIGURES_DIR)
    print("=" * 90)


if __name__ == "__main__":
    main()
