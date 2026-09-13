from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "final_engineered_val.csv"

FEATURE_NAMES_PATH = PROCESSED_DIR / "feature_names.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
FEATURE_NAMES_JOBLIB_PATH = MODELS_DIR / "feature_names.joblib"

X_VAL_PATH = PROCESSED_DIR / "X_val.npy"
Y_VAL_PATH = PROCESSED_DIR / "y_val.npy"

COMPARISON_PATH = RESULTS_DIR / "validation_model_comparison.csv"

TARGET_COL = "median_house_value"


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_processed_data():

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:\n{TRAIN_PATH}\n\n"
            "Run preprocessing.py first."
        )

    if not VAL_PATH.exists():
        raise FileNotFoundError(
            f"Validation data not found:\n{VAL_PATH}\n\n"
            "Run preprocessing.py first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    if TARGET_COL not in train_df.columns:
        raise ValueError(
            f"Target column `{TARGET_COL}` is missing from training data."
        )

    if TARGET_COL not in val_df.columns:
        raise ValueError(
            f"Target column `{TARGET_COL}` is missing from validation data."
        )

    return train_df, val_df


# ============================================================
# 3. PREPARE X AND Y
# ============================================================

def prepare_features(train_df, val_df):

    X_train = train_df.drop(columns=[TARGET_COL]).copy()
    y_train = train_df[TARGET_COL].copy()

    X_val = val_df.drop(columns=[TARGET_COL]).copy()
    y_val = val_df[TARGET_COL].copy()

    # --------------------------------------------------------
    # Make sure validation columns exactly match training
    # --------------------------------------------------------

    X_val = X_val.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # --------------------------------------------------------
    # Convert everything to numeric
    # --------------------------------------------------------

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_val = X_val.apply(pd.to_numeric, errors="coerce")

    # --------------------------------------------------------
    # Handle infinite values
    # --------------------------------------------------------

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_val = X_val.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Fill missing values using TRAINING medians only
    # --------------------------------------------------------

    train_medians = X_train.median()

    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)

    # Safety fallback
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    feature_names = X_train.columns.tolist()

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        feature_names
    )


# ============================================================
# 4. TRAIN MODELS
# ============================================================

def train_models():

    try:

        print("=" * 70)
        print("MODEL TRAINING PIPELINE")
        print("=" * 70)

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        PROCESSED_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Load processed data
        # ----------------------------------------------------

        print("\n[1/6] Loading processed datasets...")

        train_df, val_df = load_processed_data()

        print(
            f"[INFO] Training dataset shape: "
            f"{train_df.shape}"
        )

        print(
            f"[INFO] Validation dataset shape: "
            f"{val_df.shape}"
        )

        # ----------------------------------------------------
        # Prepare features
        # ----------------------------------------------------

        print("\n[2/6] Preparing features...")

        (
            X_train,
            X_val,
            y_train,
            y_val,
            feature_names
        ) = prepare_features(
            train_df,
            val_df
        )

        print(
            f"[INFO] Number of model features: "
            f"{len(feature_names)}"
        )

        print("\n[INFO] Features used by model:")

        for i, feature in enumerate(
            feature_names,
            start=1
        ):
            print(f"  {i}. {feature}")

        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        X_train_np = X_train.to_numpy(
            dtype=float
        )

        X_val_np = X_val.to_numpy(
            dtype=float
        )

        y_train_np = y_train.to_numpy(
            dtype=float
        )

        y_val_np = y_val.to_numpy(
            dtype=float
        )

        # ----------------------------------------------------
        # Save validation data
        # ----------------------------------------------------

        print("\n[3/6] Saving validation data...")

        np.save(
            X_VAL_PATH,
            X_val_np
        )

        np.save(
            Y_VAL_PATH,
            y_val_np
        )

        print(
            f"[SAVED] {X_VAL_PATH}"
        )

        print(
            f"[SAVED] {Y_VAL_PATH}"
        )

        # ----------------------------------------------------
        # Save feature names
        # ----------------------------------------------------

        feature_names_df = pd.DataFrame({
            "feature_name": feature_names
        })

        feature_names_df.to_csv(
            FEATURE_NAMES_PATH,
            index=False
        )

        joblib.dump(
            feature_names,
            FEATURE_NAMES_JOBLIB_PATH
        )

        print(
            f"[SAVED] {FEATURE_NAMES_JOBLIB_PATH}"
        )

        # ----------------------------------------------------
        # Define models
        # ----------------------------------------------------

        print("\n[4/6] Training models...")

        models = {

            "Ridge_Baseline":
                Ridge(
                    alpha=1.0
                ),

            "Random_Forest":
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=1
                ),

            "Gradient_Boosting":
                GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=42
                )
        }

        results = []

        best_rmse = float("inf")
        best_model = None
        best_model_name = None

        # ----------------------------------------------------
        # Train + evaluate each model
        # ----------------------------------------------------

        for name, model in models.items():

            print(
                f"\n[INFO] Training: {name}"
            )

            model.fit(
                X_train_np,
                y_train_np
            )

            predictions = model.predict(
                X_val_np
            )

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_val_np,
                        predictions
                    )
                )
            )

            mae = float(
                mean_absolute_error(
                    y_val_np,
                    predictions
                )
            )

            r2 = float(
                r2_score(
                    y_val_np,
                    predictions
                )
            )

            results.append({
                "Model": name,
                "Val_RMSE": round(
                    rmse,
                    2
                ),
                "Val_MAE": round(
                    mae,
                    2
                ),
                "Val_R2": round(
                    r2,
                    4
                ),
                "Features": len(
                    feature_names
                )
            })

            print(
                f"  RMSE: {rmse:.2f}"
            )

            print(
                f"  MAE : {mae:.2f}"
            )

            print(
                f"  R2   : {r2:.4f}"
            )

            # Lower RMSE = better model
            if rmse < best_rmse:

                best_rmse = rmse
                best_model = model
                best_model_name = name

        # ----------------------------------------------------
        # Save comparison results
        # ----------------------------------------------------

        print("\n[5/6] Saving model comparison...")

        results_df = pd.DataFrame(
            results
        ).sort_values(
            "Val_RMSE"
        )

        results_df.to_csv(
            COMPARISON_PATH,
            index=False
        )

        print(
            f"[SAVED] {COMPARISON_PATH}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        print("\n[6/6] Saving best model...")

        if best_model is None:
            raise RuntimeError(
                "No model was successfully trained."
            )

        joblib.dump(
            best_model,
            BEST_MODEL_PATH
        )

        print(
            f"[SAVED] {BEST_MODEL_PATH}"
        )

        # ----------------------------------------------------
        # Final summary
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(
            f"\nBest Model       : {best_model_name}"
        )

        print(
            f"Best Validation RMSE: "
            f"{best_rmse:.2f}"
        )

        print(
            f"Training Rows    : "
            f"{len(X_train_np)}"
        )

        print(
            f"Validation Rows  : "
            f"{len(X_val_np)}"
        )

        print(
            f"Model Features   : "
            f"{X_train_np.shape[1]}"
        )

        print("\nGenerated files:")

        print(
            f"[SAVED] {BEST_MODEL_PATH}"
        )

        print(
            f"[SAVED] {FEATURE_NAMES_JOBLIB_PATH}"
        )

        print(
            f"[SAVED] {X_VAL_PATH}"
        )

        print(
            f"[SAVED] {Y_VAL_PATH}"
        )

        print(
            f"[SAVED] {COMPARISON_PATH}"
        )

        print(
            "\n[SUCCESS] Pipeline training stage completed."
        )

        sys.stdout.flush()

    except Exception as e:

        print(
            f"\n[ERROR] Training failed: {str(e)}"
        )

        sys.stdout.flush()

        raise


# ============================================================
# 5. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_models()