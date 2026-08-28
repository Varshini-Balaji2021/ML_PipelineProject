from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Correct path resolution going up 3 levels from src/modeling/train.py to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"


def train_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load processed data arrays
    train_data = np.load(PROCESSED_DIR / "train.npz")
    val_data = np.load(PROCESSED_DIR / "val.npz")

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]

    if y_train is None or y_val is None:
        raise ValueError("Target values (`y`) are missing from the processed dataset arrays.")

    # Define models to evaluate
    models = {
        "Ridge_Baseline": Ridge(alpha=1.0),
        "Random_Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient_Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42),
    }

    results = []
    best_rmse = float("inf")
    best_model_name = None
    best_model_obj = None

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        r2 = float(r2_score(y_val, preds))

        results.append({
            "Model": name,
            "Val_RMSE": round(rmse, 2),
            "Val_R2": round(r2, 4)
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model_obj = model

    # Save comparison metrics
    results_df = pd.DataFrame(results).sort_values("Val_RMSE")
    results_df.to_csv(RESULTS_DIR / "validation_model_comparison.csv", index=False)

    # Save the best frozen model artifact
    joblib.dump(best_model_obj, MODELS_DIR / "best_model.joblib")

    print("\n" + "="*40)
    print("MODEL VALIDATION RESULTS")
    print("="*40)
    print(results_df.to_string(index=False))
    print(f"\nBest Model Selected: {best_model_name} (RMSE: {best_rmse:.2f})")
    print(f"Saved to: {MODELS_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    train_models()