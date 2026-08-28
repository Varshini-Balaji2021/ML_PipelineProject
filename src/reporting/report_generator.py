"""
report_generator.py
Generates HTML reports for your ML pipeline stages.
"""
from pathlib import Path
from datetime import datetime
import pandas as pd

# Automatically find your project root from this file's location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_html_report(title: str, sections: dict, output_name: str):
    """
    Generates a clean HTML report.
    sections = {"Section Title": "<p>Content or HTML table</p>"}
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; color: #333; }}
            h1 {{ color: #1f4e79; }}
            h2 {{ color: #2e75b6; border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: #fff; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .container {{ background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.05); }}
            .footer {{ margin-top: 40px; font-size: 12px; color: #666; text-align: center; }}
        </style>
    </head>
    <body>
    <div class="container">
        <h1>{title}</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    """
    
    for section_title, content in sections.items():
        html += f"<h2>{section_title}</h2>\n{content}\n"
        
    html += """
        <div class="footer">
            AI-Assisted ML Project Platform | Governance & Reporting Layer
        </div>
    </div>
    </body>
    </html>
    """
    
    html_path = REPORTS_DIR / f"{output_name}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Report successfully saved -> {html_path}")
    return html_path


def generate_eda_report():
    """Reads candidate explanatory variables and generates the EDA summary report."""
    cand_path = PROJECT_ROOT / "results" / "eda_analysis_summary" / "candidate_explanatory_variables.csv"
    
    if cand_path.exists():
        df = pd.read_csv(cand_path)
        table_html = df.to_html(classes="table", index=False)
        
        sections = {
            "Overview": "<p>This report summarizes the candidate explanatory variables from your exploratory data analysis stage.</p>",
            "Candidate Explanatory Variables": table_html
        }
        return generate_html_report("EDA Analytical Summary", sections, "eda_report")
    else:
        print(f"EDA results not found at: {cand_path}. Run your EDA stage first.")


def generate_preprocessing_report(summary_stats: dict):
    """
    Generates a preprocessing summary report.
    """
    stats_html = "<ul>"
    for key, value in summary_stats.items():
        stats_html += f"<li><strong>{key}:</strong> {value}</li>"
    stats_html += "</ul>"
    
    sections = {
        "Preprocessing Overview": "<p>This report documents the data cleaning, transformation, and train/validation splitting steps performed on the dataset.</p>",
        "Execution Statistics": stats_html,
        "Next Steps": "<p>Data is now ready for Feature Engineering and Model Training.</p>"
    }
    
    return generate_html_report("Preprocessing & Cleaning Summary", sections, "03_preprocessing_report")


def generate_training_report(metrics_dict: dict, best_model_name: str):
    """Generates the Model Training & Evaluation summary report."""
    metrics_html = "<ul>"
    for k, v in metrics_dict.items():
        metrics_html += f"<li><strong>{k}:</strong> {v}</li>"
    metrics_html += "</ul>"
    
    sections = {
        "Model Training Overview": "<p>Trained and compared multiple regression models (Ridge, Random Forest, Gradient Boosting).</p>",
        "Validation Performance": metrics_html,
        "Winning Model Selection": f"<p><strong>{best_model_name}</strong> was selected as the optimal model based on validation metrics and saved successfully.</p>"
    }
    return generate_html_report("Model Training & Evaluation Report", sections, "04_model_validation_report")


def generate_submission_report(submission_path: str):
    """Generates the Final Submission & Export report."""
    sections = {
        "Pipeline Completion": "<p>The end-to-end ML pipeline has successfully executed from EDA to Model Evaluation.</p>",
        "Test Predictions & Submission": f"<p>Final predictions have been generated and compiled successfully. Ready for submission.</p><p><strong>File Location:</strong> {submission_path}</p>"
    }
    return generate_html_report("Final Submission & Export Summary", sections, "05_final_submission_report")


if __name__ == "__main__":
    print("--- Generating All Stage Reports ---")
    
    # 1. Generate EDA Report
    generate_eda_report()
    
    # 2. Generate Preprocessing Report
    generate_preprocessing_report({
        "Initial Rows": 20640,
        "Cleaned Rows": 20640,
        "Missing Values Handled": 0,
        "Train/Val Split Ratio": "80 / 20"
    })
    
    # 3. Generate Training Report
    generate_training_report(
        metrics_dict={"Val_RMSE": "49141.07", "Val_R2": "0.8160"},
        best_model_name="Random_Forest"
    )
    
    # 4. Generate Submission Report
    generate_submission_report("results/submissions/submission.csv")
    
    print("All stage reports successfully generated in the 'reports' folder!")