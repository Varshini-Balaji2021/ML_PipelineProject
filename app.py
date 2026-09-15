from pathlib import Path
from datetime import datetime
import hashlib
import json
import subprocess
import sys
import html
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="California House Prices | Regression Workbench",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "housing.csv"
PROCESSED_DIR = DATA_DIR / "processed"

SRC_DIR = PROJECT_ROOT / "src"

INGEST_SCRIPT_PATH = SRC_DIR / "data" / "ingest.py"
FEATURE_ENGINEERING_SCRIPT_PATH = SRC_DIR / "feature_engineering.py"
PREPROCESS_SCRIPT_PATH = SRC_DIR / "preprocessing" / "preprocessing.py"
TRAIN_SCRIPT_PATH = SRC_DIR / "modeling" / "train.py"

EVALUATE_SCRIPT_PATH = (
    SRC_DIR / "evaluation" / "evaluate.py"
)

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"

MODELS_DIR = PROJECT_ROOT / "models" / "trained"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "median_house_value"

# These are the features used by the normal/recommended workflow.
# Users can switch to Customize Features if required.
RECOMMENDED_FEATURES = [
    "median_income",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "ocean_proximity",
    "rooms_per_household",
    "population_per_household",
    "bedrooms_per_household",
]


# ============================================================
# 3. SESSION STATE
# ============================================================

if "selected_features" not in st.session_state:
    st.session_state.selected_features = []

if "pipeline_output" not in st.session_state:
    st.session_state.pipeline_output = ""

if "feature_engineering_output" not in st.session_state:
    st.session_state.feature_engineering_output = ""

if "ai_interpretation" not in st.session_state:
    st.session_state.ai_interpretation = ""

if "test_evaluation_output" not in st.session_state:
    st.session_state.test_evaluation_output = ""


# ============================================================
# 4. DATA HELPERS
# ============================================================

@st.cache_data
def load_housing_data():
    if not RAW_DATA_PATH.exists():
        return None

    try:
        return pd.read_csv(RAW_DATA_PATH)
    except Exception:
        return None


@st.cache_data
def load_engineered_data():
    path = PROCESSED_DIR / "final_engineered_train.csv"

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def clear_data_cache():
    st.cache_data.clear()


def run_python_script(script_path):
    if not script_path.exists():
        return False, f"Script not found:\n{script_path}"

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        output = (
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

        return result.returncode == 0, output

    except Exception as exc:
        return False, str(exc)


def save_feature_store(df, selected_features):
    if df is None:
        raise ValueError("Engineered dataset is not available.")

    if not selected_features:
        raise ValueError("Select at least one feature.")

    missing = [
        feature
        for feature in selected_features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Selected features are missing from the engineered dataset: "
            + ", ".join(missing)
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    columns_to_save = selected_features.copy()

    if TARGET_COL in df.columns:
        columns_to_save.append(TARGET_COL)

    output = df[columns_to_save].copy()

    output.to_csv(
        PROCESSED_DIR / "final_engineered_train.csv",
        index=False
    )

    st.session_state.selected_features = selected_features.copy()

    clear_data_cache()

    return PROCESSED_DIR / "final_engineered_train.csv"


# ============================================================
# 5. MODEL / VALIDATION HELPERS
# ============================================================

def load_best_model():
    if not BEST_MODEL_PATH.exists():
        return None

    try:
        return joblib.load(BEST_MODEL_PATH)
    except Exception:
        return None


def load_validation_data():
    x_path = PROCESSED_DIR / "X_val.npy"
    y_path = PROCESSED_DIR / "y_val.npy"
    feature_path = PROCESSED_DIR / "feature_names.csv"

    if not x_path.exists() or not y_path.exists():
        return None, None, None

    try:
        X_val = np.load(
            x_path,
            allow_pickle=True
        )

        y_val = np.load(
            y_path,
            allow_pickle=True
        )

        feature_names = None

        if feature_path.exists():
            feature_df = pd.read_csv(feature_path)

            if "feature_name" in feature_df.columns:
                feature_names = (
                    feature_df["feature_name"]
                    .astype(str)
                    .tolist()
                )

        return X_val, y_val, feature_names

    except Exception:
        return None, None, None


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    error = y_true - y_pred

    rmse = float(
        np.sqrt(
            np.mean(error ** 2)
        )
    )

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    ss_res = float(
        np.sum(error ** 2)
    )

    ss_tot = float(
        np.sum(
            (y_true - np.mean(y_true)) ** 2
        )
    )

    r2 = (
        1 - ss_res / ss_tot
        if ss_tot != 0
        else np.nan
    )

    return rmse, mae, r2


def get_metrics_dataframe():
    possible_paths = [
        METRICS_DIR / "validation_model_comparison.csv",
        RESULTS_DIR / "validation_model_comparison.csv",
    ]

    for path in possible_paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue

    return None


def get_best_model_row(metrics_df):
    if metrics_df is None or metrics_df.empty:
        return None

    for column in [
        "Val_RMSE",
        "Validation_RMSE",
        "RMSE",
        "Val RMSE",
    ]:
        if column in metrics_df.columns:

            temp = metrics_df.copy()

            temp[column] = pd.to_numeric(
                temp[column],
                errors="coerce"
            )

            temp = temp.dropna(
                subset=[column]
            )

            if not temp.empty:
                return temp.sort_values(
                    column,
                    ascending=True
                ).iloc[0]

    return metrics_df.iloc[0]


def get_model_name(best_row):
    if best_row is None:
        return "Not available"

    for column in [
        "Model",
        "model",
        "Model_Name",
        "model_name",
        "Algorithm",
    ]:
        if column in best_row.index:
            return str(best_row[column])

    return "Best trained model"


# ============================================================
# 6. REPORT HELPERS
# ============================================================

def calculate_file_hash(path):
    if not path.exists():
        return None

    sha = hashlib.sha256()

    try:
        with open(path, "rb") as file:
            for chunk in iter(
                lambda: file.read(8192),
                b"",
            ):
                sha.update(chunk)

        return sha.hexdigest()

    except Exception:
        return None


def save_registry_snapshot(
    df=None,
    selected_features=None,
):
    metrics_df = get_metrics_dataframe()

    features_used = (
        selected_features
        or st.session_state.get(
            "selected_features",
            []
        )
    )

    snapshot = {
        "registry_version": "2.0",
        "generated_at": datetime.now().isoformat(),
        "project": {
            "name": "California House Prices",
            "problem_type": "Regression",
            "target": TARGET_COL,
        },
        "data_registry": {
            "source": str(RAW_DATA_PATH),
            "sha256": calculate_file_hash(
                RAW_DATA_PATH
            ),
        },
        "feature_store": {
            "selected_features": features_used,
            "feature_count": len(features_used),
        },
        "model_registry": {
            "best_model": str(BEST_MODEL_PATH),
            "available": BEST_MODEL_PATH.exists(),
        },
    }

    if df is not None:
        snapshot["data_registry"].update({
            "rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
        })

    if metrics_df is not None:
        snapshot["model_registry"][
            "comparison_results"
        ] = metrics_df.to_dict(
            orient="records"
        )

    path = REPORTS_DIR / "registry.json"

    path.write_text(
        json.dumps(
            snapshot,
            indent=4,
            default=str
        ),
        encoding="utf-8",
    )

    return path


def generate_html_report(
    df=None,
    metrics_df=None,
    best_model_name="Not available",
    ai_text="",
):
    rows = (
        f"{df.shape[0]:,}"
        if df is not None
        else "Not available"
    )

    selected_features = (
        st.session_state.get(
            "selected_features",
            []
        )
    )

    feature_count = len(selected_features)

    if metrics_df is not None:
        metrics_table = metrics_df.to_html(
            index=False,
            border=0,
        )
    else:
        metrics_table = (
            "<p>No model comparison results available.</p>"
        )

    insight = (
        ai_text
        if ai_text
        else (
            f"The model development workflow used "
            f"{feature_count} selected input features. "
            "Models were compared using validation RMSE, "
            "MAE, and R²."
        )
    )

    report = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>California House Prices - ML Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    background: #F8FAFC;
    color: #1E293B;
    margin: 0;
    padding: 40px;
}}

.container {{
    max-width: 1000px;
    margin: auto;
    background: white;
    padding: 40px;
    border-radius: 12px;
}}

h1 {{
    color: #0F172A;
}}

h2 {{
    color: #2563EB;
    margin-top: 30px;
}}

.cards {{
    display: flex;
    gap: 15px;
}}

.card {{
    flex: 1;
    background: #F1F5F9;
    padding: 20px;
    border-radius: 8px;
}}

.card strong {{
    display: block;
    font-size: 20px;
    margin-top: 8px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th {{
    background: #2563EB;
    color: white;
    padding: 10px;
    text-align: left;
}}

td {{
    padding: 10px;
    border-bottom: 1px solid #E2E8F0;
}}

.insight {{
    background: #EFF6FF;
    padding: 20px;
    border-radius: 8px;
}}
</style>
</head>

<body>
<div class="container">

<h1>🏠 California House Prices Prediction Report</h1>

<p>
Generated on:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</p>

<h2>Project Overview</h2>

<div class="cards">

<div class="card">
Dataset Rows
<strong>{rows}</strong>
</div>

<div class="card">
Selected Features
<strong>{feature_count}</strong>
</div>

<div class="card">
Best Model
<strong>{html.escape(str(best_model_name))}</strong>
</div>

</div>

<h2>Selected Features</h2>

<p>
{html.escape(", ".join(selected_features))
if selected_features
else "No selected features recorded."}
</p>

<h2>Model Validation & Comparison</h2>

{metrics_table}

<h2>Executive Insights</h2>

<div class="insight">
{html.escape(insight).replace(chr(10), "<br>")}
</div>

</div>
</body>
</html>
"""

    path = REPORTS_DIR / "complete_ml_report.html"

    path.write_text(
        report,
        encoding="utf-8",
    )

    return path


# ============================================================
# 7. APPLICATION HEADER
# ============================================================

st.title("🏠 California House Prices Workbench")
st.caption(
    "End-to-End Machine Learning Regression Platform"
)
st.divider()


# ============================================================
# 8. SIDEBAR
# ============================================================

stage = st.sidebar.radio(
    "Select Stage",
    [
        "1. Project Dashboard & Ingestion",
        "2. Data Understanding & Profiling",
        "3. Feature Engineering & Model Training",
        "4. Model Evaluation & Results",
        "5. Workflow Orchestration",
        "6. Reports & Export",
        "7. Final Presentation",
    ],
)


# ============================================================
# STAGE 1
# ============================================================

if stage == "1. Project Dashboard & Ingestion":

    st.header(
        "📊 Project Dashboard & Data Ingestion"
    )

    st.info(
        "Load the California Housing dataset before "
        "starting feature engineering."
    )

    st.subheader("1. Data Ingestion Control")

    if st.button("📥 Load / Ingest Dataset"):

        with st.spinner(
            "Running data ingestion..."
        ):

            success, output = run_python_script(
                INGEST_SCRIPT_PATH
            )

        if success:
            st.success(
                "✅ Dataset successfully loaded."
            )
            clear_data_cache()
        else:
            st.error(
                "❌ Ingestion failed."
            )

        with st.expander("Ingestion Log"):
            st.code(output)

    df = load_housing_data()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dataset Status",
        "Available" if df is not None else "Missing",
    )

    col2.metric(
        "Dataset Rows",
        f"{len(df):,}" if df is not None else "0",
    )

    col3.metric(
        "Model Status",
        "Trained"
        if BEST_MODEL_PATH.exists()
        else "Pending",
    )

    if df is not None:

        st.divider()

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True,
        )

# ============================================================
# STAGE 2 — DATA UNDERSTANDING & EXPLORATORY DATA ANALYSIS
# ============================================================

elif stage == "2. Data Understanding & Profiling":

    st.header("Data Understanding & Exploratory Data Analysis")
    st.write(
        "Explore the structure, quality, distributions, relationships, "
        "outliers, and key patterns in the California Housing dataset."
    )

    # --------------------------------------------------------
    # LOAD RAW DATA
    # --------------------------------------------------------

    if not RAW_DATA_PATH.exists():
        st.error(
            f"Raw dataset not found at:\n{RAW_DATA_PATH}\n\n"
            "Please complete Data Ingestion in Stage 1 first."
        )
        st.stop()

    try:
        eda_df = pd.read_csv(RAW_DATA_PATH)
    except Exception as e:
        st.error(f"Unable to load the dataset: {e}")
        st.stop()

    # --------------------------------------------------------
    # 2.1 DATASET OVERVIEW
    # --------------------------------------------------------

    st.subheader("2.1 Dataset Overview")

    total_rows = len(eda_df)
    total_columns = len(eda_df.columns)
    numeric_columns = eda_df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = eda_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    missing_cells = int(eda_df.isnull().sum().sum())
    duplicate_rows = int(eda_df.duplicated().sum())

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{total_rows:,}")

    with col2:
        st.metric("Columns", total_columns)

    with col3:
        st.metric("Numeric Features", len(numeric_columns))

    with col4:
        st.metric("Categorical Features", len(categorical_columns))

    st.markdown("### Dataset Structure")

    structure_df = pd.DataFrame({
        "Column": eda_df.columns,
        "Data Type": eda_df.dtypes.astype(str).values,
        "Missing Values": eda_df.isnull().sum().values,
        "Missing %": (
            eda_df.isnull().mean().mul(100).round(2).values
        ),
        "Unique Values": eda_df.nunique().values,
    })

    st.dataframe(
        structure_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # 2.2 DATA QUALITY ASSESSMENT
    # --------------------------------------------------------

    st.subheader("2.2 Data Quality Assessment")

    quality_col1, quality_col2 = st.columns(2)

    with quality_col1:

        st.markdown("#### Missing-Value Analysis")

        missing_df = pd.DataFrame({
            "Feature": eda_df.columns,
            "Missing Count": eda_df.isnull().sum().values,
            "Missing %": eda_df.isnull().mean().mul(100).round(2).values
        })

        missing_df = missing_df.sort_values(
            "Missing Count",
            ascending=False
        )

        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True
        )

        if missing_cells == 0:
            st.success("No missing values detected in the raw dataset.")
        else:
            st.warning(
                f"{missing_cells:,} missing cells were detected. "
                "This supports the need for an imputation step during preprocessing."
            )

    with quality_col2:

        st.markdown("#### Duplicate Analysis")

        duplicate_summary = pd.DataFrame({
            "Metric": [
                "Total Rows",
                "Duplicate Rows",
                "Duplicate %"
            ],
            "Value": [
                f"{total_rows:,}",
                f"{duplicate_rows:,}",
                f"{(duplicate_rows / total_rows * 100):.2f}%"
                if total_rows > 0 else "0%"
            ]
        })

        st.dataframe(
            duplicate_summary,
            use_container_width=True,
            hide_index=True
        )

        if duplicate_rows == 0:
            st.success("No duplicate rows detected.")
        else:
            st.warning(
                f"{duplicate_rows:,} duplicate rows detected."
            )

    # --------------------------------------------------------
    # 2.3 DESCRIPTIVE STATISTICS
    # --------------------------------------------------------

    st.subheader("2.3 Descriptive Statistics")

    if numeric_columns:
        descriptive_stats = eda_df[numeric_columns].describe().T

        descriptive_stats["median"] = eda_df[numeric_columns].median()
        descriptive_stats["missing"] = eda_df[numeric_columns].isnull().sum()

        descriptive_stats = descriptive_stats[
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
                "missing"
            ]
        ]

        st.dataframe(
            descriptive_stats.round(3),
            use_container_width=True
        )

    # --------------------------------------------------------
    # 2.4 TARGET VARIABLE ANALYSIS
    # --------------------------------------------------------

    st.subheader("2.4 Target Variable Analysis")

    if TARGET_COL in eda_df.columns:

        target = eda_df[TARGET_COL].dropna()

        target_col1, target_col2 = st.columns(2)

        with target_col1:

            st.markdown("#### Target Distribution")

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.hist(target, bins=40)
            ax.set_title("Distribution of Median House Value")
            ax.set_xlabel("Median House Value")
            ax.set_ylabel("Frequency")

            st.pyplot(fig)
            plt.close(fig)

        with target_col2:

            st.markdown("#### Target Statistics")

            target_stats = pd.DataFrame({
                "Statistic": [
                    "Mean",
                    "Median",
                    "Standard Deviation",
                    "Minimum",
                    "Maximum",
                    "Skewness"
                ],
                "Value": [
                    target.mean(),
                    target.median(),
                    target.std(),
                    target.min(),
                    target.max(),
                    target.skew()
                ]
            })

            st.dataframe(
                target_stats.round(3),
                use_container_width=True,
                hide_index=True
            )

    else:
        st.warning(
            f"Target column '{TARGET_COL}' was not found in the dataset."
        )

    # --------------------------------------------------------
    # 2.5 UNIVARIATE ANALYSIS
    # --------------------------------------------------------

    st.subheader("2.5 Numeric Feature Distributions")

    if numeric_columns:

        selected_distribution_feature = st.selectbox(
            "Select a numeric feature",
            numeric_columns,
            key="eda_distribution_feature"
        )

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.hist(
            eda_df[selected_distribution_feature].dropna(),
            bins=40
        )

        ax.set_title(
            f"Distribution of {selected_distribution_feature}"
        )
        ax.set_xlabel(selected_distribution_feature)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------------
    # 2.6 CATEGORICAL ANALYSIS
    # --------------------------------------------------------

    st.subheader("2.6 Categorical Feature Analysis")

    if categorical_columns:

        selected_category = st.selectbox(
            "Select a categorical feature",
            categorical_columns,
            key="eda_category_feature"
        )

        category_counts = (
            eda_df[selected_category]
            .value_counts(dropna=False)
            .reset_index()
        )

        category_counts.columns = [
            selected_category,
            "Count"
        ]

        category_counts["Percentage"] = (
            category_counts["Count"]
            / category_counts["Count"].sum()
            * 100
        ).round(2)

        st.dataframe(
            category_counts,
            use_container_width=True,
            hide_index=True
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            category_counts[selected_category].astype(str),
            category_counts["Count"]
        )

        ax.set_title(
            f"Distribution of {selected_category}"
        )
        ax.set_xlabel(selected_category)
        ax.set_ylabel("Count")

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------------
    # 2.7 CORRELATION ANALYSIS
    # --------------------------------------------------------

    st.subheader("2.7 Correlation & Feature Relationships")

    if len(numeric_columns) >= 2:

        correlation_matrix = eda_df[numeric_columns].corr()

        st.markdown("#### Correlation Matrix")

        st.dataframe(
            correlation_matrix.round(3),
            use_container_width=True
        )

        st.markdown("#### Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(11, 8))

        image = ax.imshow(
            correlation_matrix,
            aspect="auto"
        )

        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))

        ax.set_xticklabels(
            correlation_matrix.columns,
            rotation=90
        )

        ax.set_yticklabels(
            correlation_matrix.columns
        )

        ax.set_title("Feature Correlation Heatmap")

        fig.colorbar(image, ax=ax)

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------------
    # 2.8 INCOME VS HOUSE VALUE
    # --------------------------------------------------------

    if (
        "median_income" in eda_df.columns
        and TARGET_COL in eda_df.columns
    ):

        st.markdown("#### Median Income vs Median House Value")

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.scatter(
            eda_df["median_income"],
            eda_df[TARGET_COL],
            alpha=0.35,
            s=12
        )

        ax.set_title(
            "Median Income vs Median House Value"
        )
        ax.set_xlabel("Median Income")
        ax.set_ylabel("Median House Value")

        st.pyplot(fig)
        plt.close(fig)

        income_correlation = eda_df[
            ["median_income", TARGET_COL]
        ].corr().iloc[0, 1]

        st.info(
            f"Correlation between median income and median house value: "
            f"{income_correlation:.3f}"
        )

    # --------------------------------------------------------
    # 2.9 GEOGRAPHIC ANALYSIS
    # --------------------------------------------------------

    if all(
        col in eda_df.columns
        for col in ["longitude", "latitude", TARGET_COL]
    ):

        st.subheader("2.9 Geographic Analysis")

        st.write(
            "The geographic variables help examine how housing values "
            "vary across different locations in California."
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(
            eda_df["longitude"],
            eda_df["latitude"],
            c=eda_df[TARGET_COL],
            alpha=0.45,
            s=10
        )

        ax.set_title(
            "Geographical Distribution of Median House Value"
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        fig.colorbar(
            scatter,
            ax=ax,
            label="Median House Value"
        )

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------------
    # 2.10 OUTLIER ANALYSIS
    # --------------------------------------------------------

    st.subheader("2.10 Outlier Analysis")

    outlier_candidates = [
        col
        for col in [
            "median_income",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            TARGET_COL
        ]
        if col in eda_df.columns
    ]

    if outlier_candidates:

        selected_outlier_feature = st.selectbox(
            "Select a feature for outlier analysis",
            outlier_candidates,
            key="eda_outlier_feature"
        )

        series = eda_df[selected_outlier_feature].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())

        outlier_col1, outlier_col2, outlier_col3 = st.columns(3)

        with outlier_col1:
            st.metric(
                "Q1",
                f"{q1:,.3f}"
            )

        with outlier_col2:
            st.metric(
                "Q3",
                f"{q3:,.3f}"
            )

        with outlier_col3:
            st.metric(
                "Potential Outliers",
                f"{outlier_count:,}"
            )

        fig, ax = plt.subplots(figsize=(10, 3))

        ax.boxplot(
            series,
            vert=False
        )

        ax.set_title(
            f"Boxplot — {selected_outlier_feature}"
        )
        ax.set_xlabel(selected_outlier_feature)

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Potential outliers are identified using the 1.5 × IQR rule. "
            "This is an exploratory diagnostic and does not automatically "
            "mean that observations should be removed."
        )

    # --------------------------------------------------------
    # 2.11 KEY EDA INSIGHTS
    # --------------------------------------------------------

    st.subheader("2.11 Key EDA Insights")

    insights = []

    # Target insight
    if TARGET_COL in eda_df.columns:

        target_series = eda_df[TARGET_COL].dropna()

        insights.append(
            f"The target variable '{TARGET_COL}' contains "
            f"{len(target_series):,} non-missing observations and has "
            f"a mean of {target_series.mean():,.2f}."
        )

        if abs(target_series.skew()) > 1:
            insights.append(
                f"The target distribution is substantially skewed "
                f"(skewness = {target_series.skew():.2f}), which should "
                "be considered during model development."
            )

    # Missing-value insight
    if missing_cells > 0:

        highest_missing = missing_df.iloc[0]

        insights.append(
            f"Missing values are present in the dataset. "
            f"The feature with the highest missing count is "
            f"'{highest_missing['Feature']}' with "
            f"{int(highest_missing['Missing Count']):,} missing values."
        )

    else:

        insights.append(
            "No missing values were detected in the raw dataset."
        )

    # Income relationship
    if (
        "median_income" in eda_df.columns
        and TARGET_COL in eda_df.columns
    ):

        income_corr = eda_df[
            ["median_income", TARGET_COL]
        ].corr().iloc[0, 1]

        direction = (
            "positive"
            if income_corr > 0
            else "negative"
        )

        insights.append(
            f"Median income has a {direction} relationship with "
            f"median house value (correlation = {income_corr:.3f})."
        )

    # Categorical insight
    if categorical_columns:

        category = categorical_columns[0]

        most_common = (
            eda_df[category]
            .value_counts(dropna=False)
            .index[0]
        )

        insights.append(
            f"'{category}' is a categorical variable and "
            f"'{most_common}' is its most frequent category."
        )

    # Geographic insight
    if all(
        col in eda_df.columns
        for col in ["longitude", "latitude", TARGET_COL]
    ):

        insights.append(
            "Longitude and latitude provide geographic information, "
            "allowing spatial differences in housing values to be "
            "examined during exploratory analysis."
        )

    # Outlier insight
    if outlier_candidates:

        insights.append(
            "Several continuous variables contain potential extreme "
            "observations based on the IQR rule. These observations "
            "should be evaluated rather than automatically removed."
        )

    for i, insight in enumerate(insights, start=1):
        st.markdown(f"**{i}.** {insight}")

    # --------------------------------------------------------
    # 2.12 EDA → PIPELINE DECISION LINK
    # --------------------------------------------------------

    st.subheader("2.12 EDA Findings → ML Pipeline Decisions")

    decision_df = pd.DataFrame({
        "EDA Finding": [
            "Dataset contains numerical and categorical variables",
            "Variables have different scales and distributions",
            "Missing values may be present",
            "Categorical variable: ocean_proximity",
            "Derived relationships exist between housing variables",
            "Potential extreme observations exist"
        ],
        "Pipeline Implication": [
            "Separate numerical and categorical preprocessing is required",
            "Scaling is considered during preprocessing",
            "Imputation is required where applicable",
            "Categorical encoding is required",
            "Feature engineering can create additional predictive variables",
            "Outliers should be investigated before modeling"
        ]
    })

    st.dataframe(
        decision_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # 2.13 GENERATE EDA REPORT
    # --------------------------------------------------------

    st.subheader("2.13 EDA Report")

    st.write(
        "Generate a standalone HTML report containing the main "
        "EDA findings, statistics, data-quality information, "
        "and analytical observations."
    )

    if st.button(
        "Generate EDA Report",
        key="generate_eda_report"
    ):

        REPORTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        report_path = REPORTS_DIR / "eda_report.html"

        # Build HTML tables
        structure_html = structure_df.to_html(
            index=False,
            classes="data-table",
            border=0
        )

        stats_html = (
            descriptive_stats.round(3).to_html(
                classes="data-table",
                border=0
            )
            if numeric_columns
            else "<p>No numerical features available.</p>"
        )

        missing_html = missing_df.to_html(
            index=False,
            classes="data-table",
            border=0
        )

        correlation_html = (
            correlation_matrix.round(3).to_html(
                classes="data-table",
                border=0
            )
            if len(numeric_columns) >= 2
            else "<p>Correlation analysis unavailable.</p>"
        )

        insights_html = "".join(
            f"<li>{html.escape(str(insight))}</li>"
            for insight in insights
        )

        report_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">

<title>California Housing — EDA Report</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    line-height: 1.6;
    color: #222;
}}

h1 {{
    margin-bottom: 5px;
}}

h2 {{
    margin-top: 35px;
}}

.summary {{
    display: flex;
    gap: 20px;
    margin: 25px 0;
}}

.card {{
    border: 1px solid #ddd;
    padding: 15px;
    border-radius: 8px;
    min-width: 150px;
}}

.card strong {{
    display: block;
    font-size: 22px;
}}

.data-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}}

.data-table th,
.data-table td {{
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}}

.data-table th {{
    font-weight: bold;
}}

li {{
    margin-bottom: 10px;
}}

.footer {{
    margin-top: 40px;
    padding-top: 15px;
    border-top: 1px solid #ddd;
    font-size: 13px;
}}

</style>
</head>

<body>

<h1>California Housing — Exploratory Data Analysis Report</h1>

<p>
Generated from the raw California Housing dataset.
</p>

<div class="summary">

<div class="card">
<strong>{total_rows:,}</strong>
Rows
</div>

<div class="card">
<strong>{total_columns}</strong>
Columns
</div>

<div class="card">
<strong>{missing_cells:,}</strong>
Missing Cells
</div>

<div class="card">
<strong>{duplicate_rows:,}</strong>
Duplicate Rows
</div>

</div>

<h2>1. Dataset Structure</h2>

{structure_html}

<h2>2. Data Quality</h2>

<h3>Missing Values</h3>

{missing_html}

<h3>Duplicate Rows</h3>

<p>
The dataset contains {duplicate_rows:,} duplicate rows.
</p>

<h2>3. Descriptive Statistics</h2>

{stats_html}

<h2>4. Target Variable</h2>

<p>
The target variable is <strong>{TARGET_COL}</strong>.
</p>

<h2>5. Correlation Analysis</h2>

{correlation_html}

<h2>6. Key EDA Insights</h2>

<ul>
{insights_html}
</ul>

<h2>7. EDA → ML Pipeline Decisions</h2>

{decision_df.to_html(
    index=False,
    classes="data-table",
    border=0
)}

<div class="footer">
California Housing ML Pipeline — EDA Stage
</div>

</body>
</html>
"""

        try:

            report_path.write_text(
                report_html,
                encoding="utf-8"
            )

            st.success(
                f"EDA report generated successfully: {report_path}"
            )

            with open(
                report_path,
                "rb"
            ) as report_file:

                st.download_button(
                    label="Download EDA Report",
                    data=report_file,
                    file_name="california_housing_eda_report.html",
                    mime="text/html",
                    key="download_eda_report"
                )

        except Exception as e:

            st.error(
                f"Failed to generate EDA report: {e}"
            )

# ============================================================
# STAGE 3
# ============================================================

elif stage == "3. Feature Engineering & Model Training":

    st.header(
        "⚙️ Feature Engineering & Model Training"
    )

    df = load_housing_data()

    if df is None:

        st.warning(
            "Please ingest the dataset first from Stage 1."
        )

    else:

        # ----------------------------------------------------
        # Feature Engineering
        # ----------------------------------------------------

        st.subheader(
            "1. Feature Engineering"
        )

        st.write(
            "Create the approved deterministic housing "
            "features before selecting model inputs."
        )

        if st.button(
            "⚙️ Run Feature Engineering"
        ):

            with st.spinner(
                "Creating engineered features..."
            ):

                success, output = run_python_script(
                    FEATURE_ENGINEERING_SCRIPT_PATH
                )

            st.session_state.feature_engineering_output = output

            if success:

                st.success(
                    "✅ Feature engineering completed successfully."
                )

                clear_data_cache()

            else:

                st.error(
                    "❌ Feature engineering failed."
                )

            with st.expander(
                "Feature Engineering Log"
            ):
                st.code(output)

        engineered_df = load_engineered_data()

        if engineered_df is None:

            st.info(
                "Run Feature Engineering to create the "
                "engineered feature store."
            )

        else:

            st.success(
                "✅ Engineered dataset is available."
            )

            # ------------------------------------------------
            # Feature Selection
            # ------------------------------------------------

            st.subheader(
                "2. Targeted Feature Selection"
            )

            available_features = [
                column
                for column in engineered_df.columns
                if column != TARGET_COL
            ]

            recommended = [
                feature
                for feature in RECOMMENDED_FEATURES
                if feature in available_features
            ]

            feature_mode = st.radio(
                "Feature Selection Mode",
                [
                    "Recommended Features",
                    "Customize Features",
                ],
                horizontal=True,
            )

            if feature_mode == "Recommended Features":

                selected_features = recommended

                st.info(
                    f"Using **{len(selected_features)}** "
                    "recommended features."
                )

                st.caption(
                    "Recommended features: "
                    + ", ".join(selected_features)
                )

            else:

                default_custom = [
                    feature
                    for feature in st.session_state.selected_features
                    if feature in available_features
                ]

                if not default_custom:
                    default_custom = recommended

                selected_features = st.multiselect(
                    "Select features for modelling:",
                    available_features,
                    default=default_custom,
                    key="custom_feature_selector",
                )

                st.info(
                    f"Active Features Selected: "
                    f"**{len(selected_features)}**"
                )

            # ------------------------------------------------
            # Save feature store
            # ------------------------------------------------

            if st.button(
                "💾 Save Processed Feature Store"
            ):

                try:

                    path = save_feature_store(
                        engineered_df,
                        selected_features,
                    )

                    st.success(
                        f"✅ Feature store saved successfully "
                        f"with {len(selected_features)} features."
                    )

                    st.caption(
                        f"Saved to: {path}"
                    )

                except Exception as exc:

                    st.error(
                        f"❌ Could not save feature store: {exc}"
                    )

            st.divider()

            # ------------------------------------------------
            # Preprocessing
            # ------------------------------------------------

            st.subheader(
                "3. Execute Preprocessing Pipeline"
            )

            if st.button(
                "▶ Run Preprocessing Script"
            ):

                feature_store_path = (
                    PROCESSED_DIR
                    / "final_engineered_train.csv"
                )

                if not feature_store_path.exists():

                    st.error(
                        "❌ Feature store not found. "
                        "Save the selected features first."
                    )

                else:

                    with st.spinner(
                        "Running preprocessing..."
                    ):

                        success, output = (
                            run_python_script(
                                PREPROCESS_SCRIPT_PATH
                            )
                        )

                    if success:

                        st.success(
                            "✅ Preprocessing completed successfully."
                        )

                        clear_data_cache()

                    else:

                        st.error(
                            "❌ Preprocessing failed."
                        )

                    with st.expander(
                        "Preprocessing Log"
                    ):
                        st.code(output)

            st.divider()

            # ------------------------------------------------
            # Training
            # ------------------------------------------------

            st.subheader(
                "4. Execute Model Training"
            )

            if st.button(
                "🚀 Run Model Training"
            ):

                train_path = (
                    PROCESSED_DIR / "train.csv"
                )

                if not train_path.exists():

                    st.error(
                        "❌ Processed training data not found. "
                        "Run preprocessing first."
                    )

                else:

                    with st.spinner(
                        "Training and comparing models..."
                    ):

                        success, output = (
                            run_python_script(
                                TRAIN_SCRIPT_PATH
                            )
                        )

                    if success:

                        st.success(
                            "✅ Model training completed successfully."
                        )

                        clear_data_cache()

                    else:

                        st.error(
                            "❌ Model training failed."
                        )

                    with st.expander(
                        "Training Execution Log"
                    ):
                        st.code(output)


# ============================================================
# STAGE 4

elif stage == "4. Model Evaluation & Results":

    st.header("📈 Model Evaluation & Results")

    st.subheader("4.1 Model Performance Comparison")
    metrics_df = get_metrics_dataframe()

    if metrics_df is not None and not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        best_row = get_best_model_row(metrics_df)
        best_name = get_model_name(best_row)
        st.success(f"🏆 Best Model Selected: **{best_name}**")
        st.caption("The model with the lowest validation RMSE is selected as the best model.")
    else:
        st.info("No model comparison results found. Run preprocessing and model training first.")

    st.divider()
    st.subheader("4.2 Validation Diagnostics")
    X_val, y_val, feature_names = load_validation_data()
    model = load_best_model()

    if model is not None and X_val is not None and y_val is not None:
        try:
            y_val_arr = np.asarray(y_val, dtype=float).flatten()
            if hasattr(model, "n_features_in_") and X_val.shape[1] != model.n_features_in_:
                raise ValueError(
                    f"Feature mismatch: validation data has {X_val.shape[1]} features, "
                    f"while the trained model expects {model.n_features_in_}."
                )
            predictions = np.asarray(model.predict(X_val), dtype=float).flatten()
            if len(y_val_arr) != len(predictions):
                raise ValueError("Validation target and prediction lengths do not match.")

            rmse, mae, r2 = calculate_metrics(y_val_arr, predictions)
            c1, c2, c3 = st.columns(3)
            c1.metric("Validation RMSE", f"{rmse:,.2f}")
            c2.metric("Validation MAE", f"{mae:,.2f}")
            c3.metric("Validation R²", f"{r2:.4f}")

            st.divider()
            st.subheader("📊 Comprehensive Model Diagnostics")
            residuals = y_val_arr - predictions

            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.scatter(y_val_arr, predictions, alpha=0.35, edgecolors="none")
            minimum = float(min(np.min(y_val_arr), np.min(predictions)))
            maximum = float(max(np.max(y_val_arr), np.max(predictions)))
            ax1.plot([minimum, maximum], [minimum, maximum], linestyle="--", linewidth=1.5, label="Ideal Fit")
            ax1.set_title("Actual vs. Predicted")
            ax1.set_xlabel("Actual Value")
            ax1.set_ylabel("Predicted Value")
            ax1.legend(loc="upper left")
            st.pyplot(fig1)
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.scatter(predictions, residuals, alpha=0.35, edgecolors="none")
            ax2.axhline(0, linestyle="--", linewidth=1.5)
            ax2.set_title("Residuals vs. Predicted")
            ax2.set_xlabel("Predicted Value")
            ax2.set_ylabel("Residual")
            st.pyplot(fig2)
            plt.close(fig2)

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            ax3.hist(residuals, bins=40, edgecolor="black")
            ax3.axvline(0, linestyle="--", linewidth=1.5)
            ax3.set_title("Residual Distribution")
            ax3.set_xlabel("Prediction Error")
            ax3.set_ylabel("Frequency")
            st.pyplot(fig3)
            plt.close(fig3)

        except Exception as exc:
            st.error(f"❌ Could not generate validation diagnostics: {exc}")
    else:
        st.info("Validation artifacts or trained model are missing. Run preprocessing and training first.")

    st.divider()
    st.subheader("4.3 Final Test Evaluation")
    st.write(
        "Evaluate the selected best model on the unseen test dataset. "
        "This step is performed after model selection using the existing evaluate.py script."
    )

    test_data_path = PROCESSED_DIR / "test.csv"
    test_metrics_path = METRICS_DIR / "test_evaluation_metrics.csv"
    submission_path = RESULTS_DIR / "submissions" / "submission.csv"

    if not BEST_MODEL_PATH.exists():
        st.warning("Best model artifact not found. Run model training first.")
    elif not test_data_path.exists():
        st.warning("Processed test dataset not found. Run preprocessing first.")
    elif not EVALUATE_SCRIPT_PATH.exists():
        st.error(f"Evaluation script not found at:\n{EVALUATE_SCRIPT_PATH}")
    else:
        if st.button("🧪 Run Final Test Evaluation", key="run_final_test_evaluation"):
            with st.spinner("Evaluating the best model on unseen test data..."):
                success, output = run_python_script(EVALUATE_SCRIPT_PATH)
            st.session_state.test_evaluation_output = output
            if success:
                st.success("✅ Final test evaluation completed successfully.")
            else:
                st.error("❌ Final test evaluation failed.")

        if st.session_state.test_evaluation_output:
            with st.expander("Final Test Evaluation Log", expanded=False):
                st.code(st.session_state.test_evaluation_output)

    st.subheader("4.4 Final Test Metrics")
    if test_metrics_path.exists():
        try:
            test_metrics_df = pd.read_csv(test_metrics_path)
            st.dataframe(test_metrics_df, use_container_width=True, hide_index=True)
            if not test_metrics_df.empty:
                test_row = test_metrics_df.iloc[0]
                c1, c2, c3 = st.columns(3)
                rmse_value = pd.to_numeric(test_row.get("Test_RMSE"), errors="coerce")
                mae_value = pd.to_numeric(test_row.get("Test_MAE"), errors="coerce")
                r2_value = pd.to_numeric(test_row.get("Test_R2"), errors="coerce")
                c1.metric("Test RMSE", f"{rmse_value:,.2f}" if pd.notna(rmse_value) else "N/A")
                c2.metric("Test MAE", f"{mae_value:,.2f}" if pd.notna(mae_value) else "N/A")
                c3.metric("Test R²", f"{r2_value:.4f}" if pd.notna(r2_value) else "N/A")
        except Exception as exc:
            st.error(f"❌ Could not read test evaluation metrics: {exc}")
    else:
        st.info("Final test metrics are not available yet. Click **Run Final Test Evaluation** above.")

    st.subheader("4.5 Test Prediction Output")
    if submission_path.exists():
        try:
            submission_df = pd.read_csv(submission_path)
            st.dataframe(submission_df.head(20), use_container_width=True, hide_index=True)
            st.download_button(
                label="📥 Download Test Predictions",
                data=submission_df.to_csv(index=False),
                file_name="submission.csv",
                mime="text/csv",
                key="download_test_predictions",
            )
            st.caption(f"Prediction file contains {len(submission_df):,} rows.")
        except Exception as exc:
            st.error(f"❌ Could not load submission file: {exc}")
    else:
        st.info("Test prediction output is not available yet. Run Final Test Evaluation first.")


# STAGE 5
# ============================================================

elif stage == "5. Workflow Orchestration":

    st.header(
        "⚙️ Workflow Orchestration Control Center"
    )

    st.write(
        "Run the complete pipeline or execute individual "
        "stages in the correct dependency order."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "⚡ Complete End-to-End Pipeline"
        )

        st.caption(
            "Ingestion → Feature Engineering → "
            "Recommended Feature Store → Preprocessing → Training"
        )

        if st.button(
            "Run Full Pipeline"
        ):

            logs = []

            with st.spinner(
                "Executing complete workflow..."
            ):

                # ------------------------------------------
                # 1. INGESTION
                # ------------------------------------------

                success, output = (
                    run_python_script(
                        INGEST_SCRIPT_PATH
                    )
                )

                logs.append(
                    "===== 1. INGESTION =====\n"
                    + output
                )

                if not success:

                    st.error(
                        "❌ Ingestion failed. "
                        "Pipeline stopped."
                    )

                else:

                    clear_data_cache()

                    # --------------------------------------
                    # 2. FEATURE ENGINEERING
                    # --------------------------------------

                    success, output = (
                        run_python_script(
                            FEATURE_ENGINEERING_SCRIPT_PATH
                        )
                    )

                    logs.append(
                        "===== 2. FEATURE ENGINEERING =====\n"
                        + output
                    )

                    if not success:

                        st.error(
                            "❌ Feature engineering failed. "
                            "Pipeline stopped."
                        )

                    else:

                        clear_data_cache()

                        engineered_df = (
                            load_engineered_data()
                        )

                        try:

                            recommended = [
                                feature
                                for feature in RECOMMENDED_FEATURES
                                if feature in engineered_df.columns
                            ]

                            save_feature_store(
                                engineered_df,
                                recommended,
                            )

                            logs.append(
                                "===== 3. FEATURE STORE =====\n"
                                f"Saved {len(recommended)} "
                                "recommended features."
                            )

                            # --------------------------------
                            # 3. PREPROCESSING
                            # --------------------------------

                            success, output = (
                                run_python_script(
                                    PREPROCESS_SCRIPT_PATH
                                )
                            )

                            logs.append(
                                "===== 4. PREPROCESSING =====\n"
                                + output
                            )

                            if not success:

                                st.error(
                                    "❌ Preprocessing failed. "
                                    "Pipeline stopped."
                                )

                            else:

                                # ----------------------------
                                # 4. TRAINING
                                # ----------------------------

                                success, output = (
                                    run_python_script(
                                        TRAIN_SCRIPT_PATH
                                    )
                                )

                                logs.append(
                                    "===== 5. MODEL TRAINING =====\n"
                                    + output
                                )

                                if success:

                                    st.success(
                                        "✅ Complete pipeline "
                                        "executed successfully!"
                                    )

                                    clear_data_cache()

                                else:

                                    st.error(
                                        "❌ Model training failed."
                                    )

                        except Exception as exc:

                            st.error(
                                f"❌ Feature-store creation failed: {exc}"
                            )

            st.session_state.pipeline_output = (
                "\n\n".join(logs)
            )

    with col2:

        st.subheader(
            "🎯 Stage-Wise Orchestration"
        )

        selected_stage_op = st.selectbox(
            "Select stage:",
            [
                "Ingestion",
                "Feature Engineering",
                "Preprocessing",
                "Training",
            ],
        )

        if st.button(
            f"Execute {selected_stage_op}"
        ):

            script_map = {
                "Ingestion": INGEST_SCRIPT_PATH,
                "Feature Engineering":
                    FEATURE_ENGINEERING_SCRIPT_PATH,
                "Preprocessing":
                    PREPROCESS_SCRIPT_PATH,
                "Training":
                    TRAIN_SCRIPT_PATH,
            }

            script_path = script_map[
                selected_stage_op
            ]

            with st.spinner(
                f"Running {selected_stage_op}..."
            ):

                success, output = (
                    run_python_script(
                        script_path
                    )
                )

            st.session_state.pipeline_output = output

            if success:

                st.success(
                    f"✅ {selected_stage_op} completed successfully."
                )

                clear_data_cache()

            else:

                st.error(
                    f"❌ {selected_stage_op} failed."
                )

    if st.session_state.pipeline_output:

        st.divider()

        st.subheader(
            "Execution Log"
        )

        st.code(
            st.session_state.pipeline_output
        )


# ============================================================
# STAGE 6
# ============================================================

elif stage == "6. Reports & Export":

    st.header(
        "📁 Reports, Registry & Artifact Export"
    )

    df = load_housing_data()
    metrics_df = get_metrics_dataframe()

    best_row = get_best_model_row(
        metrics_df
    )

    best_model_name = get_model_name(
        best_row
    )

    selected_features = (
        st.session_state.get(
            "selected_features",
            []
        )
    )

    st.subheader(
        "Project Registry"
    )

    registry_path = save_registry_snapshot(
        df=df,
        selected_features=selected_features,
    )

    st.success(
        "✅ Registry snapshot generated."
    )

    with open(
        registry_path,
        "rb"
    ) as file:

        st.download_button(
            label="📥 Download Registry JSON",
            data=file,
            file_name="registry.json",
            mime="application/json",
        )

    st.divider()

    st.subheader(
        "Complete HTML Report"
    )

    report_path = generate_html_report(
        df=df,
        metrics_df=metrics_df,
        best_model_name=best_model_name,
        ai_text=st.session_state.ai_interpretation,
    )

    st.success(
        "✅ Complete report generated."
    )

    with open(
        report_path,
        "rb"
    ) as file:

        st.download_button(
            label="📥 Download Complete HTML Report",
            data=file,
            file_name="complete_ml_report.html",
            mime="text/html",
        )


# ============================================================
# STAGE 7
# ============================================================

elif stage == "7. Final Presentation":

    st.header(
        "🎓 Final Academic Presentation & Architecture"
    )

    st.markdown(
        """
### 1. Problem Statement & Context

* **Objective:** Predict California median housing values using demographic and spatial variables.
* **Problem Type:** **Supervised Regression**
* **Target:** `median_house_value`
* **Evaluation Metrics:** RMSE, MAE, and R².

### 2. End-to-End Pipeline

```text
Raw Housing Dataset
        ↓
Data Ingestion
        ↓
Feature Engineering
        ↓
Targeted Feature Selection
        ↓
Train / Validation Split
        ↓
Missing-Value Imputation
        ↓
Model Training
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Validation Diagnostics
        ↓
Reports & Export
```

### 3. Models Evaluated

* Ridge Regression — baseline
* Random Forest Regressor
* Gradient Boosting Regressor

### 4. Validation Diagnostics

* Actual vs Predicted
* Residuals vs Predicted
* Residual Distribution
* Validation RMSE
* Validation MAE
* Validation R²

### 5. Workflow Orchestration

The full workflow executes in dependency order:

**Ingestion → Feature Engineering → Feature Store → Preprocessing → Training**
"""
    )

    st.success(
        "✅ Regression workbench is ready for academic presentation."
    )
