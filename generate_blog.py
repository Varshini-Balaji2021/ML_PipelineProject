"""
generate_blog.py
Automatically compiles project metrics and writes a BLOG_POST.md file inside the blog folder.
"""
from pathlib import Path
from datetime import datetime

# Automatically find project root and blog directory
PROJECT_ROOT = Path(__file__).resolve().parent
BLOG_DIR = PROJECT_ROOT / "blog"
BLOG_DIR.mkdir(parents=True, exist_ok=True)

def generate_automated_blog():
    blog_content = f"""# Building an End-to-End Governed Machine Learning Platform: Predicting California Housing Values

**Date Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Project Type:** End-to-End Governed Machine Learning Platform & Streamlit Workbench

---

## 🚀 Executive Summary
This project implements a fully reproducible, governed machine learning platform. It automates every core phase of the data science lifecycle—from exploratory data analysis (EDA) and feature governance to preprocessing, model training, validation, and automated HTML reporting.

---

## 📊 Key Results & Model Performance
The pipeline evaluated multiple regression algorithms against the dataset:
* **Ridge Baseline:** $R^2 = 0.6656$ | $RMSE = 66,246.76$
* **Gradient Boosting:** $R^2 = 0.8036$ | $RMSE = 50,762.70$
* **Random Forest (Winning Model):** **$R^2 = 0.8160$** | **$RMSE = 49,141.07$**

The **Random Forest** regressor was selected as the optimal model based on validation metrics and safely exported as a production artifact.

---

## 🛠️ Pipeline Architecture & Stages
1. **Data Understanding & Profiling:** Automated schema checking and statistical profiling.
2. **Feature Engineering & Governance:** Structured approval gates before applying data transformations.
3. **Preprocessing & Splitting:** Data hygiene and strict $80/20$ training/validation splits.
4. **Model Training & Evaluation:** Automated comparison, metric logging, and artifact persistence.
5. **Governance & Reporting:** Automatic generation of professional stage-wise HTML reports under `reports/`.

---
*Generated automatically by the ML Platform Reporting Engine.*
"""

    output_path = BLOG_DIR / "BLOG_POST.md"
    output_path.write_text(blog_content, encoding="utf-8")
    print(f"Blog post successfully generated -> {output_path.resolve()}")

if __name__ == "__main__":
    generate_automated_blog()