from pathlib import Path
import subprocess
import pandas as pd
import streamlit as st

# Optional workflow imports (wrapped in try-except in case files don't exist yet)
try:
    from workflow.pipeline_builder import run_full_pipeline, run_step
except ImportError:
    run_full_pipeline = None
    run_step = None

# ==========================================
# 1. Project Paths Definition (Defined First!)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
EDA_SUMMARY_DIR = RESULTS_DIR / "eda_analysis_summary"
EDA_FIGURES_DIR = RESULTS_DIR / "eda" / "figures"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "housing.csv"

# Correct script path pointing to your preprocessing script
PREPROCESS_SCRIPT_PATH = PROJECT_ROOT / "src" / "preprocessing" / "preprocessing.py"
TRAIN_SCRIPT_PATH = PROJECT_ROOT / "src" / "modeling" / "train.py"

st.set_page_config(page_title="ML Workbench Platform", layout="wide")
st.title("🛠️ AI-Assisted ML Project Platform")

# ==========================================
# 2. Sidebar Navigation
# ==========================================
stage = st.sidebar.radio(
    "Select Stage",
    [
        "1. Project Dashboard",
        "2. Data Understanding & Profiling",
        "3. Feature Engineering Plan & Approval",
        "4. Preprocessing & Modeling",
        "5. Model Evaluation & Results",
        "6. Workflow Orchestration",
        "7. Reports & Export",
        "8. Final Presentation"
    ]
)

# ==========================================
# 3. Stage Views
# ==========================================

if stage == "1. Project Dashboard":
    st.header("Project Overview & Status")
    st.info("House Prices Prediction Pipeline - Regression Workbench")

    col1, col2, col3 = st.columns(3)
    train_exists = (SPLITS_DIR / "train.csv").exists() or RAW_DATA_PATH.exists()
    col1.metric("Data Splits / Raw", "Available" if train_exists else "Missing")
    col2.metric("EDA Status", "Completed" if EDA_SUMMARY_DIR.exists() else "Pending")

    processed_exists = (PROCESSED_DIR / "train.npz").exists() or (PROCESSED_DIR / "train.csv").exists()
    col3.metric("Preprocessing Status", "Completed" if processed_exists else "Pending")

elif stage == "2. Data Understanding & Profiling":
    st.header("Exploratory Data Analysis & Visual Figures")

    cand_path = EDA_SUMMARY_DIR / "candidate_explanatory_variables.csv"
    if cand_path.exists():
        st.subheader("Candidate Explanatory Variables")
        df_cand = pd.read_csv(cand_path)
        st.dataframe(df_cand, use_container_width=True)
    else:
        st.warning("Candidate variables file not found.")

    if EDA_FIGURES_DIR.exists():
        fig_files = sorted(list(EDA_FIGURES_DIR.glob("*.png")) + list(EDA_FIGURES_DIR.glob("*.jpg")))
        if fig_files:
            for i in range(0, len(fig_files), 2):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(str(fig_files[i]), caption=fig_files[i].stem.replace('_', ' ').title(), use_container_width=True)
                if i + 1 < len(fig_files):
                    with col2:
                        st.image(str(fig_files[i + 1]), caption=fig_files[i + 1].stem.replace('_', ' ').title(), use_container_width=True)

    html_path = EDA_SUMMARY_DIR / "analytical_summary.html"
    if html_path.exists():
        st.subheader("Interactive Analytical Summary Report")
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)

elif stage == "3. Feature Engineering Plan & Approval":
    st.header("Feature Engineering Governance & Approval")
    fe_plan_path = EDA_SUMMARY_DIR / "feature_engineering_recommendations.csv"
    if fe_plan_path.exists():
        fe_df = pd.read_csv(fe_plan_path)
        st.dataframe(fe_df, use_container_width=True)
        approved = st.checkbox("Approve Deterministic Ratios & Features for Preprocessing")
        if approved:
            st.success("Feature Engineering Plan Approved!")
    else:
        st.warning("Feature engineering recommendations file not found.")

elif stage == "4. Preprocessing & Modeling":
    st.header("Execution Pipeline: Preprocessing & Training")
    st.markdown("Trigger your background Python scripts directly from the platform interface.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Run Preprocessing")
        if st.button("Execute Preprocessing Script"):
            with st.spinner("Running preprocessing..."):
                if not PREPROCESS_SCRIPT_PATH.exists():
                    st.error(f"Preprocessing script not found at {PREPROCESS_SCRIPT_PATH}")
                else:
                    res = subprocess.run(["python", str(PREPROCESS_SCRIPT_PATH)], capture_output=True, text=True)
                    if res.returncode == 0:
                        st.success("Preprocessing completed successfully!")
                        st.text(res.stdout)
                    else:
                        st.error("Preprocessing failed.")
                        st.text(res.stderr)

    with col2:
        st.subheader("2. Run Model Training")
        if st.button("Execute Model Training Script"):
            with st.spinner("Training models..."):
                script_to_run = TRAIN_SCRIPT_PATH if TRAIN_SCRIPT_PATH.exists() else PROJECT_ROOT / "train.py"
                if not script_to_run.exists():
                    st.error(f"Training script not found at {script_to_run}")
                else:
                    res = subprocess.run(["python", str(script_to_run)], capture_output=True, text=True)
                    if res.returncode == 0:
                        st.success("Training completed successfully!")
                        st.text(res.stdout)
                    else:
                        st.error("Training failed.")
                        st.text(res.stderr)

elif stage == "5. Model Evaluation & Results":
    st.header("Model Evaluation & Final Results")

    metrics_path = RESULTS_DIR / "metrics" / "validation_model_comparison.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        st.subheader("Validation Model Comparison")
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("No validation comparison metrics found. Run training first.")

    submission_path = RESULTS_DIR / "submissions" / "submission.csv"
    if submission_path.exists():
        st.subheader("Submission Predictions Preview")
        sub_df = pd.read_csv(submission_path)
        st.dataframe(sub_df.head(10), use_container_width=True)
        st.success(f"Submission file ready at: {submission_path}")

elif stage == "6. Workflow Orchestration":
    st.title("⚙️ Workflow Orchestration & Pipeline Engine")
    st.write("Trigger and monitor automated end-to-end execution of your ML pipeline stages.")
    
    if st.button("🚀 Run Entire Pipeline"):
        with st.spinner("Executing end-to-end ML pipeline..."):
            try:
                if run_full_pipeline:
                    run_full_pipeline()
                    st.success("Pipeline executed successfully!")
                else:
                    st.error("run_full_pipeline module not found.")
            except Exception as e:
                st.error(f"Execution failed: {e}")

elif stage == "7. Reports & Export":
    st.header("📁 Reports & Export Center")
    st.markdown("Download stage-wise reports generated by the platform.")
    
    if REPORTS_DIR.exists():
        report_files = sorted(list(REPORTS_DIR.glob("*.html")) + list(REPORTS_DIR.glob("*.pdf")))
        if report_files:
            for file in report_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 **{file.name}**")
                with col2:
                    with open(file, "rb") as f:
                        st.download_button("Download", data=f, file_name=file.name, key=file.name)
        else:
            st.info("No reports found in the reports folder yet.")
    else:
        st.warning("Reports directory does not exist yet.")

elif stage == "8. Final Presentation":
    st.header("📢 Project Final Presentation Summary")
    st.markdown("Here are key highlights ready for your project review.")
    st.write("- **Dataset:** Regression / House Prices Dataset")
    st.write("- **Objective:** Build an end-to-end governed machine learning pipeline.")
    st.success("Your project platform pipeline is fully complete and presentation-ready!")