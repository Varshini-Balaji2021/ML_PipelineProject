from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Resolve project root and paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered ratio features based on the approved feature plan."""
    data = df.copy()
    data["rooms_per_household"] = data["total_rooms"] / (data["households"] + 1e-6)
    data["bedrooms_per_room"] = data["total_bedrooms"] / (data["total_rooms"] + 1e-6)
    data["population_per_household"] = data["population"] / (data["households"] + 1e-6)
    data["bedrooms_per_household"] = data["total_bedrooms"] / (data["households"] + 1e-6)
    return data


def run_preprocessing():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "validation.csv"
    test_path = SPLITS_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split: {train_path}. Please ensure splits are created.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if val_path.exists() else None
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    # Compute engineered features
    train_eng = compute_ratios(train_df)
    if val_df is not None:
        val_eng = compute_ratios(val_df)
    if test_df is not None:
        test_eng = compute_ratios(test_df)

    num_cols = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "rooms_per_household",
        "bedrooms_per_room",
        "population_per_household",
        "bedrooms_per_household",
    ]
    cat_cols = ["ocean_proximity"]
    target_col = "median_house_value"

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])

    # Fit transformer strictly on training data
    X_train = preprocessor.fit_transform(train_eng[num_cols + cat_cols])
    y_train = train_eng[target_col].to_numpy() if target_col in train_eng.columns else None

    np.savez(PROCESSED_DIR / "train.npz", X=X_train, y=y_train)

    if val_df is not None:
        X_val = preprocessor.transform(val_eng[num_cols + cat_cols])
        y_val = val_eng[target_col].to_numpy() if target_col in val_eng.columns else None
        np.savez(PROCESSED_DIR / "val.npz", X=X_val, y=y_val)

    if test_df is not None:
        X_test = preprocessor.transform(test_eng[num_cols + cat_cols])
        y_test = test_eng[target_col].to_numpy() if target_col in test_eng.columns else None
        np.savez(PROCESSED_DIR / "test.npz", X=X_test, y=y_test)

    # Save fitted preprocessor artifact
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    print("Preprocessing completed successfully.")
    print(f"Train feature matrix: {X_train.shape}")
    if val_df is not None:
        print(f"Validation feature matrix: {X_val.shape}")
    if test_df is not None:
        print(f"Test feature matrix: {X_test.shape}")


if __name__ == "__main__":
    run_preprocessing()