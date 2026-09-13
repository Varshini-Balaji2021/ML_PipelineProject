from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

MODELS_DIR = PROJECT_ROOT / "models" / "trained"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
SUBMISSION_DIR = PROJECT_ROOT / "results" / "submissions"

TARGET_COL = "median_house_value"


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load trained best model
    # --------------------------------------------------------

    model_path = MODELS_DIR / "best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run train.py first."
        )

    model = joblib.load(model_path)

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # 2. Load processed test dataset
    # --------------------------------------------------------

    test_path = PROCESSED_DIR / "test.csv"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Processed test dataset not found at {test_path}. "
            "Run preprocessing.py first."
        )

    test_df = pd.read_csv(test_path)

    print(f"\nProcessed test dataset shape: {test_df.shape}")

    # --------------------------------------------------------
    # 3. Separate features and target
    # --------------------------------------------------------

    if TARGET_COL not in test_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in processed test data."
        )

    y_test = test_df[TARGET_COL].to_numpy()

    X_test = test_df.drop(columns=[TARGET_COL])

    # --------------------------------------------------------
    # 4. Align test features with training features
    # --------------------------------------------------------

    feature_names_path = MODELS_DIR / "feature_names.joblib"

    if feature_names_path.exists():

        feature_names = joblib.load(feature_names_path)

        # Convert to list in case it is stored as another format
        feature_names = list(feature_names)

        missing_features = [
            col for col in feature_names
            if col not in X_test.columns
        ]

        extra_features = [
            col for col in X_test.columns
            if col not in feature_names
        ]

        if missing_features:
            raise ValueError(
                "Test dataset is missing features required by the model:\n"
                f"{missing_features}"
            )

        # Keep exactly the same feature order used during training
        X_test = X_test[feature_names]

        print(f"Model features: {len(feature_names)}")

        if extra_features:
            print(
                f"Ignoring {len(extra_features)} extra test columns."
            )

    else:

        print(
            "Warning: feature_names.joblib not found. "
            "Using current test column order."
        )

    # --------------------------------------------------------
    # 5. Convert to numeric
    # --------------------------------------------------------

    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Safety check
    if X_test.isna().any().any():

        print(
            "\nWarning: Missing/invalid values detected in test data."
        )

        X_test = X_test.fillna(X_test.median())

    # --------------------------------------------------------
    # 6. Convert to NumPy
    # --------------------------------------------------------

    X_test = X_test.to_numpy(dtype=float)

    print(f"Final test feature shape: {X_test.shape}")
    print(f"Expected model features: {model.n_features_in_}")

    # --------------------------------------------------------
    # 7. Final feature-count validation
    # --------------------------------------------------------

    if X_test.shape[1] != model.n_features_in_:

        raise ValueError(
            f"Feature mismatch: test data has "
            f"{X_test.shape[1]} features, but the model expects "
            f"{model.n_features_in_} features."
        )

    # --------------------------------------------------------
    # 8. Generate predictions
    # --------------------------------------------------------

    print("\nRunning inference on test dataset...")

    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # 9. Calculate test metrics
    # --------------------------------------------------------

    test_rmse = float(
        np.sqrt(mean_squared_error(y_test, predictions))
    )

    test_mae = float(
        mean_absolute_error(y_test, predictions)
    )

    test_r2 = float(
        r2_score(y_test, predictions)
    )

    metrics_summary = pd.DataFrame([
        {
            "Test_RMSE": round(test_rmse, 2),
            "Test_MAE": round(test_mae, 2),
            "Test_R2": round(test_r2, 4),
        }
    ])

    metrics_path = RESULTS_DIR / "test_evaluation_metrics.csv"

    metrics_summary.to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # 10. Display metrics
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION METRICS")
    print("=" * 60)

    print(metrics_summary.to_string(index=False))

    # --------------------------------------------------------
    # 11. Generate submission
    # --------------------------------------------------------

    original_test_path = SPLITS_DIR / "test.csv"

    if original_test_path.exists():

        original_test_df = pd.read_csv(
            original_test_path
        )

        submission = pd.DataFrame({
            "id": original_test_df.index,
            TARGET_COL: predictions,
        })

    else:

        submission = pd.DataFrame({
            "id": np.arange(len(predictions)),
            TARGET_COL: predictions,
        })

    submission_path = (
        SUBMISSION_DIR / "submission.csv"
    )

    submission.to_csv(
        submission_path,
        index=False
    )

    # --------------------------------------------------------
    # 12. Completion message
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nTest rows       : {len(y_test)}")
    print(f"Test features   : {X_test.shape[1]}")
    print(f"Test RMSE       : {test_rmse:.2f}")
    print(f"Test MAE        : {test_mae:.2f}")
    print(f"Test R²         : {test_r2:.4f}")

    print("\nGenerated files:")

    print(f"✓ {metrics_path}")
    print(f"✓ {submission_path}")

    print("\n[SUCCESS] Model evaluation stage completed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_model()