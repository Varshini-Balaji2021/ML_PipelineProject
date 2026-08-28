from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Resolve project root and paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
SUBMISSION_DIR = PROJECT_ROOT / "results" / "submissions"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def evaluate_model():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")

    # Load frozen best model and test data
    model = joblib.load(model_path)
    test_data = np.load(PROCESSED_DIR / "test.npz")
    X_test, y_test = test_data["X"], test_data["y"]

    print("Running inference on test dataset...")
    predictions = model.predict(X_test)

    # If test labels are available, compute test metrics
    if y_test is not None and len(y_test) > 0 and not np.isnan(y_test).all():
        test_rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        test_mae = float(mean_absolute_error(y_test, predictions))
        test_r2 = float(r2_score(y_test, predictions))

        metrics_summary = pd.DataFrame([{
            "Test_RMSE": round(test_rmse, 2),
            "Test_MAE": round(test_mae, 2),
            "Test_R2": round(test_r2, 4)
        }])
        metrics_summary.to_csv(RESULTS_DIR / "test_evaluation_metrics.csv", index=False)

        print("\n" + "="*40)
        print("TEST SET EVALUATION METRICS")
        print("="*40)
        print(metrics_summary.to_string(index=False))

    # Generate submission file matching original test structure
    test_csv_path = SPLITS_DIR / "test.csv"
    if test_csv_path.exists():
        test_df = pd.read_csv(test_csv_path)
        submission = pd.DataFrame({
            "id": test_df.index,
            "median_house_value": predictions
        })
    else:
        submission = pd.DataFrame({
            "prediction": predictions
        })

    submission_file = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(submission_file, index=False)
    print(f"\nSubmission file successfully exported to: {submission_file}")


if __name__ == "__main__":
    evaluate_model()