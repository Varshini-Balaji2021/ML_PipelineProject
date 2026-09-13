from pathlib import Path
from datetime import datetime
import hashlib
import json
import subprocess
import sys
import html

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
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
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
# STAGE 2
# ============================================================

elif stage == "2. Data Understanding & Profiling":

    st.header(
        "🔍 Data Understanding & Exploratory Data Analysis"
    )

    df = load_housing_data()

    if df is None:

        st.warning(
            "Please ingest the dataset first from Stage 1."
        )

    else:

        st.subheader(
            "1. Summary Statistics & Distribution Analysis"
        )

        numeric_df = df.select_dtypes(
            include=np.number
        )

        if not numeric_df.empty:
            summary = numeric_df.describe().T
            st.dataframe(
                summary,
                use_container_width=True,
            )

        st.divider()

        st.subheader(
            "2. Target Distribution"
        )

        if TARGET_COL in df.columns:

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            ax.hist(
                df[TARGET_COL].dropna(),
                bins=40,
                edgecolor="black",
            )

            ax.set_title(
                "Distribution of Median House Value"
            )

            ax.set_xlabel(
                "Median House Value"
            )

            ax.set_ylabel(
                "Frequency"
            )

            st.pyplot(fig)

            plt.close(fig)

        st.divider()

        st.subheader(
            "3. Correlation Matrix"
        )

        if not numeric_df.empty:

            corr = numeric_df.corr()

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            cax = ax.matshow(
                corr,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
            )

            fig.colorbar(cax)

            ax.set_xticks(
                range(len(corr.columns))
            )

            ax.set_yticks(
                range(len(corr.columns))
            )

            ax.set_xticklabels(
                corr.columns,
                rotation=90,
            )

            ax.set_yticklabels(
                corr.columns
            )

            ax.set_title(
                "Feature Correlation Heatmap",
                pad=20,
            )

            st.pyplot(fig)

            plt.close(fig)


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
# ============================================================

elif stage == "4. Model Evaluation & Results":

    st.header(
        "📈 Model Evaluation & Results"
    )

    metrics_df = get_metrics_dataframe()

    if metrics_df is not None:

        st.subheader(
            "Model Performance Comparison"
        )

        st.dataframe(
            metrics_df,
            use_container_width=True,
        )

        best_row = get_best_model_row(
            metrics_df
        )

        best_name = get_model_name(
            best_row
        )

        st.success(
            f"🏆 Best Model Selected: **{best_name}**"
        )

    else:

        st.info(
            "No evaluation metrics found. "
            "Run preprocessing and model training first."
        )

    st.divider()

    st.subheader(
        "Validation Diagnostics"
    )

    X_val, y_val, feature_names = (
        load_validation_data()
    )

    model = load_best_model()

    if (
        model is not None
        and X_val is not None
        and y_val is not None
    ):

        try:

            y_val_arr = np.asarray(
                y_val,
                dtype=float
            ).flatten()

            predictions = model.predict(
                X_val
            )

            predictions = np.asarray(
                predictions,
                dtype=float
            ).flatten()

            if len(y_val_arr) != len(predictions):
                raise ValueError(
                    "Validation target and prediction lengths do not match."
                )

            rmse, mae, r2 = calculate_metrics(
                y_val_arr,
                predictions
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Validation RMSE",
                f"{rmse:,.2f}"
            )

            c2.metric(
                "Validation MAE",
                f"{mae:,.2f}"
            )

            c3.metric(
                "Validation R²",
                f"{r2:.4f}"
            )

            st.divider()

            st.subheader(
                "📊 Comprehensive Model Diagnostics"
            )

            residuals = (
                y_val_arr - predictions
            )

            # Actual vs predicted
            fig1, ax1 = plt.subplots(
                figsize=(8, 5)
            )

            ax1.scatter(
                y_val_arr,
                predictions,
                alpha=0.35,
                edgecolors="none",
            )

            minimum = float(
                min(
                    np.min(y_val_arr),
                    np.min(predictions),
                )
            )

            maximum = float(
                max(
                    np.max(y_val_arr),
                    np.max(predictions),
                )
            )

            ax1.plot(
                [minimum, maximum],
                [minimum, maximum],
                linestyle="--",
                linewidth=1.5,
                label="Ideal Fit",
            )

            ax1.set_title(
                "Actual vs. Predicted"
            )

            ax1.set_xlabel(
                "Actual Value"
            )

            ax1.set_ylabel(
                "Predicted Value"
            )

            ax1.legend(
                loc="upper left"
            )

            st.pyplot(fig1)

            plt.close(fig1)

            # Residuals vs predicted
            fig2, ax2 = plt.subplots(
                figsize=(8, 5)
            )

            ax2.scatter(
                predictions,
                residuals,
                alpha=0.35,
                edgecolors="none",
            )

            ax2.axhline(
                0,
                linestyle="--",
                linewidth=1.5,
            )

            ax2.set_title(
                "Residuals vs. Predicted"
            )

            ax2.set_xlabel(
                "Predicted Value"
            )

            ax2.set_ylabel(
                "Residual"
            )

            st.pyplot(fig2)

            plt.close(fig2)

            # Residual distribution
            fig3, ax3 = plt.subplots(
                figsize=(8, 5)
            )

            ax3.hist(
                residuals,
                bins=40,
                edgecolor="black",
            )

            ax3.axvline(
                0,
                linestyle="--",
                linewidth=1.5,
            )

            ax3.set_title(
                "Residual Distribution"
            )

            ax3.set_xlabel(
                "Prediction Error"
            )

            ax3.set_ylabel(
                "Frequency"
            )

            st.pyplot(fig3)

            plt.close(fig3)

        except Exception as exc:

            st.error(
                f"❌ Could not generate diagnostics: {exc}"
            )

    else:

        st.info(
            "Validation artifacts or trained model are missing. "
            "Run preprocessing and training first."
        )


# ============================================================
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
