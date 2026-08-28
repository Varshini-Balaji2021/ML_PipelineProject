---
> **Project Author Note:** 
> This project was completed by **Varshini Balaji** (Roll No: **CB.BU.P2ASB24193**) as part of the Machine Learning course **ML-23BA045E**, under the guidance of **Prashoban Sir**, Amrita School of Business, Coimbatore.
---
# End-to-End Governed Machine Learning Platform & Pipeline

## 📋 Project Overview
This project implements an enterprise-grade, reproducible, and governed machine learning platform. Designed to bridge the gap between exploratory data science and production software engineering, the system automates and audits every phase of the ML lifecycle—from raw data ingestion and exploratory data analysis (EDA) to feature engineering governance, model training, evaluation, and automated HTML reporting.

---

## 🏗️ System Architecture & Workflow Stages
The platform follows a strict sequential pipeline managed via an interactive **Streamlit** dashboard (`app.py`):

1. **Project Dashboard (`1. Project Dashboard`):** Tracks overall pipeline health, dataset availability, and stage statuses in real-time.
2. **Data Understanding & Profiling (`2. Data Understanding & Profiling`):** Inspects raw data, validates semantic schemas, and executes statistical profiling to identify anomalies and missing data.
3. **Feature Engineering Plan & Governance (`3. Feature Engineering Plan & Approval`):** Formalizes candidate features, business rationales, and requires explicit user/analyst approval checkboxes before any transformation is applied.
4. **Preprocessing & Modeling (`4. Preprocessing & Modeling`):** Executes data hygiene protocols, handles missing values/outliers, and enforces a strict 80/20 train/validation split.
5. **Model Evaluation & Results (`5. Model Evaluation & Results`):** Compares multiple regression algorithms using robust validation metrics (RMSE, $R^2$) and selects the optimal model.
6. **Reports & Export Center (`6. Reports & Export`):** Automatically compiles and generates professional HTML stage-wise governance reports.
7. **Final Presentation View (`7. Final Presentation`):** Summarizes key project highlights, slide notes, and architectural metrics directly inside the UI.

---

## 📊 Data Understanding & Methodology
* **Data Source & Hygiene:** The platform ingests tabular housing data, validates types against strict semantic schemas, and isolates the test set early to prevent data leakage.
* **Exploratory Data Analysis (EDA):** Automated scripts analyze distributions, identify skewness, map spatial coordinates, and compile summary metrics into `results/eda/`.
* **Governance Gates:** No feature engineering or model training can proceed without meeting explicit governance checks and approval logs.

---

## 📈 Model Performance & Results
Multiple regression models were trained and evaluated:
* **Ridge Baseline:** $R^2 = 0.6656$ | $RMSE = 66,246.76$
* **Gradient Boosting:** $R^2 = 0.8036$ | $RMSE = 50,762.70$
* **Random Forest (Winning Model):** **$R^2 = 0.8160$** | **$RMSE = 49,141.07$**

The **Random Forest** regressor was selected as the optimal model to capture non-linear housing price patterns. Final predictions are exported to `results/submissions/submission.csv`.

---


   
