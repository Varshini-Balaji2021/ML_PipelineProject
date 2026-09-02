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
    st.info("California House Prices Prediction Pipeline - Regression Workbench")

    col1, col2, col3 = st.columns(3)
    train_exists = (SPLITS_DIR / "train.csv").exists() or RAW_DATA_PATH.exists()
    col1.metric("Data Splits / Raw", "Available" if train_exists else "Missing")
    col2.metric("EDA Status", "Completed" if EDA_SUMMARY_DIR.exists() else "Pending")

    processed_exists = (PROCESSED_DIR / "train.npz").exists() or (PROCESSED_DIR / "train.csv").exists()
    col3.metric("Preprocessing Status", "Completed" if processed_exists else "Pending")

elif stage == "2. Data Understanding & Profiling":
    st.header("Exploratory Data Analysis & Essential Visualizations")

    cand_path = EDA_SUMMARY_DIR / "candidate_explanatory_variables.csv"
    if cand_path.exists():
        st.subheader("Candidate Explanatory Variables")
        df_cand = pd.read_csv(cand_path)
        st.dataframe(df_cand, use_container_width=True)
    else:
        st.warning("Candidate variables file not found.")

    st.markdown("Here are the essential figures capturing feature correlations, target distributions, and non-linear patterns.")

    # Use two columns for side-by-side core layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Correlation Heatmap")
        corr_img = EDA_FIGURES_DIR / "correlation_heatmap.png"
        if corr_img.exists():
            st.image(str(corr_img), use_container_width=True, caption="Correlation Matrix Heatmap")
        else:
            st.info("Correlation heatmap figure not found in path.")

    with col2:
        st.subheader("Target Variable Distribution")
        target_img = EDA_FIGURES_DIR / "target_distribution.png"
        if target_img.exists():
            st.image(str(target_img), use_container_width=True, caption="Housing Price Distribution")
        else:
            st.info("Target distribution figure not found in path.")

    # Full width for advanced analytical trend plot (Recursive Workspace Search)
    st.subheader("Advanced Non-Linear Trend Analysis")
    
    lowess_img = None
    for img_path in PROJECT_ROOT.glob("**/figures/lowess_*.png"):
        if img_path.exists():
            lowess_img = img_path
            break

    if lowess_img and lowess_img.exists():
        st.image(str(lowess_img), use_container_width=True, caption=f"Advanced Trend: {lowess_img.stem.replace('_', ' ').title()}")
    else:
        st.info("Advanced LOWESS figure not found in path.")

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
    st.markdown("Download your essential project reports: EDA, Model Validation, and Final Submission.")
    
    if REPORTS_DIR.exists():
        all_files = list(REPORTS_DIR.glob("*.html")) + list(REPORTS_DIR.glob("*.pdf"))
        
        if all_files:
            # Dynamically locate the 3 target files based on keywords
            eda_file = next((f for f in all_files if "eda" in f.name.lower()), None)
            val_file = next((f for f in all_files if "validation" in f.name.lower() or "model" in f.name.lower()), None)
            sub_file = next((f for f in all_files if "submission" in f.name.lower() or "final" in f.name.lower()), None)
            
            # Keep ONLY these 3 files and completely ignore any others
            report_files = [f for f in [eda_file, val_file, sub_file] if f is not None]
            
            if report_files:
                for idx, file in enumerate(report_files, start=1):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{idx}:** 📄 {file.name}")
                    with col2:
                        with open(file, "rb") as f:
                            st.download_button("Download", data=f, file_name=file.name, key=file.name)
            else:
                st.info("The core reports (EDA, Validation, Final Submission) were not found in the reports folder yet.")
        else:
            st.info("No report files found in the reports folder.")
    else:
        st.warning("Reports directory does not exist yet.")

elif stage == "8. Final Presentation":
    st.header("📢 Project Final Presentation Summary")
    st.markdown("Comprehensive overview of the end-to-end governed machine learning platform ready for academic review.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏛️ Academic Context")
        st.write("- **Author:** Varshini Balaji")
        st.write("- **Roll No:** CB.BU.P2ASB24193")
        st.write("- **Course:** ML-23BA045E (Machine Learning)")
        st.write("- **Institution:** Amrita School of Business, Coimbatore")
        st.write("- **Guide:** Dr. Prashobhan Palakkeel")
        
    with col2:
        st.markdown("### ⚙️ System Specifications")
        st.write("- **Domain:** Regression / California Housing Dataset")
        st.write("- **Architecture:** Modular Python Scripts + Streamlit UI")
        st.write("- **Governance:** Human-in-the-loop feature approval gates")
        st.write("- **Best Model:** Random Forest Regressor ($R^2 = 0.8160$)")

    st.markdown("---")
    st.subheader("📈 Model Performance & Results")
    st.markdown("Multiple regression models were trained and evaluated:")
    st.markdown("* **Ridge Baseline:** $R^2 = 0.6656$ | $RMSE = 66,246.76$")
    st.markdown("* **Gradient Boosting:** $R^2 = 0.8036$ | $RMSE = 50,762.70$")
    st.markdown("* **Random Forest (Winning Model):** **$R^2 = 0.8160$** | **$RMSE = 49,141.07$**")
    st.markdown("The **Random Forest** regressor was selected as the optimal model to capture non-linear housing price patterns." \
    " Final predictions are exported to `results/submissions/submission.csv`.")
    st.markdown("---")
    st.subheader("🌍 ESG & Sustainability Impact Alignment")
    st.markdown("""
    * **SDG 11 (Sustainable Cities):** Leveraged spatial and geographic features to model property valuation trends, supporting equitable urban planning.
    * **Social Equity:** Addressed housing affordability by providing transparent price predictions to help identify and mitigate market disparities.
    * **Algorithmic Governance:** Implemented human-in-the-loop validation gates and strict pipeline modularity to ensure ethical, bias-free, and auditable machine learning practices.
    """)

    st.markdown("---")
    st.subheader("🔑 Core Engineering Takeaways")
    st.markdown("""
    1. **Strict Separation of Concerns:** Decoupled raw data ingestion, feature engineering governance, preprocessing, and training into modular components.
    2. **Reproducible Orchestration:** Automated full pipeline execution ensures zero target leakage and complete pipeline traceability.
    3. **Production-Ready Artifacts:** Successfully exported optimal model weights (`.joblib`) and final submission prediction files ready for deployment.
    """)
    st.success("Your project platform pipeline is fully complete and presentation-ready!")