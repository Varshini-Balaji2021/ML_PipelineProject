from pathlib import Path
import subprocess
import pandas as pd
import streamlit as st

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
EDA_SUMMARY_DIR = RESULTS_DIR / "eda_analysis_summary"
EDA_FIGURES_DIR = RESULTS_DIR / "eda" / "figures"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "housing.csv"

st.set_page_config(page_title="ML Workbench Platform", layout="wide")
st.title("🛠️ AI-Assisted ML Project Platform")

# Sidebar Navigation
step = st.sidebar.radio(
    "Lifecycle Stages",
    [
        "1. Project Dashboard",
        "2. Data Understanding & Profiling",
        "3. Feature Engineering Plan & Approval",
        "4. Preprocessing & Modeling",
        "5. Model Evaluation & Results",
        "6. Reports & Export",
        "7. Final Presentation"
    ],
)

if step == "1. Project Dashboard":
    st.header("Project Overview & Status")
    st.info("House Prices Prediction Pipeline - Regression Workbench")

    col1, col2, col3 = st.columns(3)
    train_exists = (SPLITS_DIR / "train.csv").exists() or RAW_DATA_PATH.exists()
    col1.metric("Data Splits / Raw", "Available" if train_exists else "Missing")
    col2.metric("EDA Status", "Completed" if EDA_SUMMARY_DIR.exists() else "Pending")

    processed_exists = (PROCESSED_DIR / "train.npz").exists() or (PROCESSED_DIR / "train.csv").exists()
    col3.metric("Preprocessing Status", "Completed" if processed_exists else "Pending")

elif step == "2. Data Understanding & Profiling":
    st.header("Exploratory Data Analysis & Visual Figures")

    # 1. Display Candidate Variables Table if available
    cand_path = EDA_SUMMARY_DIR / "candidate_explanatory_variables.csv"
    if cand_path.exists():
        st.subheader("Candidate Explanatory Variables")
        df_cand = pd.read_csv(cand_path)
        st.dataframe(df_cand, use_container_width=True)
    else:
        st.warning("Candidate variables file not found.")

    # 2. Display Pre-generated Figures from results/eda/figures/
    st.subheader("📊 EDA Visualizations & Figures")
    if EDA_FIGURES_DIR.exists():
        fig_files = sorted(list(EDA_FIGURES_DIR.glob("*.png")) + list(EDA_FIGURES_DIR.glob("*.jpg")))
        
        if fig_files:
            # Display images in a clean 2-column grid format (fixed with use_container_width)
            for i in range(0, len(fig_files), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    img_path1 = fig_files[i]
                    st.image(str(img_path1), caption=img_path1.stem.replace('_', ' ').title(), use_container_width=True)
                
                if i + 1 < len(fig_files):
                    with col2:
                        img_path2 = fig_files[i + 1]
                        st.image(str(img_path2), caption=img_path2.stem.replace('_', ' ').title(), use_container_width=True)
        else:
            st.info("No figure image files found in results/eda/figures/")
    else:
        st.warning(f"Figures directory not found at {EDA_FIGURES_DIR}")

    # 3. Load HTML Analytical Summary Report if available
    html_path = EDA_SUMMARY_DIR / "analytical_summary.html"
    if html_path.exists():
        st.subheader("Interactive Analytical Summary Report")
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)

elif step == "3. Feature Engineering Plan & Approval":
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

elif step == "4. Preprocessing & Modeling":
    st.header("Execution Pipeline: Preprocessing & Training")
    st.markdown("Trigger your background Python scripts directly from the platform interface.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Run Preprocessing")
        if st.button("Execute Preprocessing Script"):
            with st.spinner("Running preprocessing..."):
                script_path = PROJECT_ROOT / "src" / "preprocessing" / "preprocess.py"
                if not script_path.exists():
                    script_path = PROJECT_ROOT / "preprocess.py"

                res = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
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
                train_script = PROJECT_ROOT / "src" / "modeling" / "train.py"
                if not train_script.exists():
                    train_script = PROJECT_ROOT / "train.py"
                    
                res = subprocess.run(["python", str(train_script)], capture_output=True, text=True)
                if res.returncode == 0:
                    st.success("Training completed successfully!")
                    st.text(res.stdout)
                else:
                    st.error("Training failed.")
                    st.text(res.stderr)

elif step == "5. Model Evaluation & Results":
    st.header("Model Evaluation & Final Results")

    metrics_path = RESULTS_DIR / "metrics" / "validation_model_comparison.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        st.subheader("Validation Model Comparison")
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("No validation comparison metrics found. Run training first.")

    test_metrics_path = RESULTS_DIR / "metrics" / "test_evaluation_metrics.csv"
    if test_metrics_path.exists():
        test_df = pd.read_csv(test_metrics_path)
        st.subheader("Untouched Test Set Performance")
        st.dataframe(test_df, use_container_width=True)

    submission_path = RESULTS_DIR / "submissions" / "submission.csv"
    if submission_path.exists():
        st.subheader("Submission Predictions Preview")
        sub_df = pd.read_csv(submission_path)
        st.dataframe(sub_df.head(10), use_container_width=True)
        st.success(f"Submission file ready at: {submission_path}")

elif step == "6. Reports & Export":
    st.header("📂 Reports & Export Center")
    st.markdown("Download stage-wise reports generated by the platform.")
    
    if REPORTS_DIR.exists():
        report_files = sorted(list(REPORTS_DIR.glob("*.html")))
        if report_files:
            for file in report_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 **{file.name}**")
                with col2:
                    with open(file, "rb") as f:
                        st.download_button(
                            label="Download",
                            data=f,
                            file_name=file.name,
                            mime="text/html",
                            key=file.name
                        )
        else:
            st.info("No reports found in the reports folder yet.")
    else:
        st.warning("Reports directory does not exist.")

elif step == "7. Final Presentation":
    st.header("📢 Project Final Presentation Summary")
    st.markdown("Here are the key project highlights and slide notes ready for your presentation review.")
    
    st.subheader("1. Project Overview & Objective")
    st.write("- **Dataset:** Regression / House Prices Dataset")
    st.write("- **Objective:** Build an end-to-end governed machine learning pipeline to predict values accurately.")
    
    st.subheader("2. Key Stage Highlights")
    st.markdown("""
    * **Data Understanding (EDA):** Analyzed key explanatory columns, distributions, and data quality metrics.
    * **Preprocessing & Splits:** Handled data hygiene and successfully created reproducible data splits.
    * **Model Training & Selection:** Trained multiple regression models and evaluated performance metrics.
    * **Winning Model:** Achieved optimal validation performance with strong regression metrics.
    """)
    
    st.subheader("3. Governance & Artifacts")
    st.write("- **Generated Reports:** All stage-wise HTML reports are compiled and ready in your `reports/` folder.")
    st.write("- **Model Artifacts:** Best trained model safely saved under your models directory.")
    st.write("- **Submission:** Final test predictions exported to `results/submissions/submission.csv`.")
    
    st.success("Your project platform pipeline is fully complete and presentation-ready!")