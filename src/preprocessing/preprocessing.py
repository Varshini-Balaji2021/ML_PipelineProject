from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------
# INPUT: CREATED BY src/split.py
# ------------------------------------------------------------

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_INPUT_PATH = SPLITS_DIR / "train.csv"
VALIDATION_INPUT_PATH = SPLITS_DIR / "validation.csv"
TEST_INPUT_PATH = SPLITS_DIR / "test.csv"

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_OUTPUT_PATH = PROCESSED_DIR / "train.csv"

VALIDATION_OUTPUT_PATH = (
    PROCESSED_DIR / "final_engineered_val.csv"
)

TEST_OUTPUT_PATH = (
    PROCESSED_DIR / "test.csv"
)

X_VAL_PATH = PROCESSED_DIR / "X_val.npy"
Y_VAL_PATH = PROCESSED_DIR / "y_val.npy"

FEATURE_NAMES_PATH = (
    PROCESSED_DIR / "feature_names.csv"
)

# ------------------------------------------------------------
# FEATURE STORE CREATED BY STREAMLIT STAGE 3
# ------------------------------------------------------------

FEATURE_STORE_PATH = (
    PROCESSED_DIR / "final_engineered_train.csv"
)

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

TARGET_COL = "median_house_value"


# ============================================================
# 1. LOAD SPLIT DATA
# ============================================================

def load_split_data():

    required_files = {
        "training": TRAIN_INPUT_PATH,
        "validation": VALIDATION_INPUT_PATH,
        "test": TEST_INPUT_PATH,
    }

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    for name, path in required_files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{name.capitalize()} split not found:\n"
                f"{path}\n\n"
                "Run src/split.py first."
            )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_INPUT_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_INPUT_PATH
    )

    test_df = pd.read_csv(
        TEST_INPUT_PATH
    )

    # --------------------------------------------------------
    # Validate datasets
    # --------------------------------------------------------

    for name, df in {
        "training": train_df,
        "validation": validation_df,
        "test": test_df,
    }.items():

        if df.empty:

            raise ValueError(
                f"{name.capitalize()} dataset is empty."
            )

        if TARGET_COL not in df.columns:

            raise ValueError(
                f"Target column '{TARGET_COL}' "
                f"is missing from {name} data."
            )

    # --------------------------------------------------------
    # Print input shapes
    # --------------------------------------------------------

    print(
        f"[INFO] Training input shape: "
        f"{train_df.shape}"
    )

    print(
        f"[INFO] Validation input shape: "
        f"{validation_df.shape}"
    )

    print(
        f"[INFO] Test input shape: "
        f"{test_df.shape}"
    )

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def create_engineered_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    result = df.copy()

    # --------------------------------------------------------
    # Rooms per household
    # --------------------------------------------------------

    if {
        "total_rooms",
        "households"
    }.issubset(result.columns):

        denominator = (
            result["households"]
            .replace(0, np.nan)
        )

        result["rooms_per_household"] = (
            result["total_rooms"]
            / denominator
        )

    # --------------------------------------------------------
    # Population per household
    # --------------------------------------------------------

    if {
        "population",
        "households"
    }.issubset(result.columns):

        denominator = (
            result["households"]
            .replace(0, np.nan)
        )

        result["population_per_household"] = (
            result["population"]
            / denominator
        )

    # --------------------------------------------------------
    # Bedrooms per household
    # --------------------------------------------------------

    if {
        "total_bedrooms",
        "households"
    }.issubset(result.columns):

        denominator = (
            result["households"]
            .replace(0, np.nan)
        )

        result["bedrooms_per_household"] = (
            result["total_bedrooms"]
            / denominator
        )

    return result


# ============================================================
# 2B. APPLY SELECTED FEATURES
# ============================================================

def apply_selected_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame
):
    """
    Apply the feature selection made in Streamlit Stage 3.

    The feature store is used only to determine which columns
    were selected.

    The actual train, validation, and test values still come
    from their respective split datasets.

    This prevents data leakage between the datasets.
    """

    # --------------------------------------------------------
    # Check feature store
    # --------------------------------------------------------

    if not FEATURE_STORE_PATH.exists():

        raise FileNotFoundError(
            "Selected feature store not found:\n"
            f"{FEATURE_STORE_PATH}\n\n"
            "Run Feature Engineering and save the selected "
            "features from Streamlit Stage 3 first."
        )

    # --------------------------------------------------------
    # Load selected feature store
    # --------------------------------------------------------

    selected_df = pd.read_csv(
        FEATURE_STORE_PATH
    )

    if selected_df.empty:

        raise ValueError(
            "Selected feature store is empty."
        )

    # --------------------------------------------------------
    # Identify selected features
    # --------------------------------------------------------

    selected_features = [
        column
        for column in selected_df.columns
        if column != TARGET_COL
    ]

    if not selected_features:

        raise ValueError(
            "No selected features found in the feature store."
        )

    # --------------------------------------------------------
    # Check selected features exist in every split
    # --------------------------------------------------------

    for name, df in {
        "training": train_df,
        "validation": validation_df,
        "test": test_df,
    }.items():

        missing = [
            feature
            for feature in selected_features
            if feature not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Selected features missing from "
                f"{name} dataset:\n"
                + ", ".join(missing)
            )

    # --------------------------------------------------------
    # Keep selected features + target
    # --------------------------------------------------------

    columns_to_keep = (
        selected_features
        + [TARGET_COL]
    )

    train_df = train_df[
        columns_to_keep
    ].copy()

    validation_df = validation_df[
        columns_to_keep
    ].copy()

    test_df = test_df[
        columns_to_keep
    ].copy()

    # --------------------------------------------------------
    # Print selected features
    # --------------------------------------------------------

    print(
        f"[INFO] Selected features applied: "
        f"{len(selected_features)}"
    )

    for feature in selected_features:

        print(
            f"[OK] {feature}"
        )

    return (
        train_df,
        validation_df,
        test_df,
        selected_features
    )


# ============================================================
# 3. SEPARATE TARGET
# ============================================================

def separate_target(df):

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = pd.to_numeric(
        df[TARGET_COL],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = df.drop(
        columns=[TARGET_COL]
    ).copy()

    return X, y


# ============================================================
# 4. ENCODE DATA
# ============================================================

def encode_data(
    X_train,
    X_validation,
    X_test
):

    # --------------------------------------------------------
    # Identify categorical columns using TRAINING data
    # --------------------------------------------------------

    categorical_columns = (
        X_train
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    if categorical_columns:

        X_train = pd.get_dummies(
            X_train,
            columns=categorical_columns,
            drop_first=True,
            dtype=float
        )

        X_validation = pd.get_dummies(
            X_validation,
            columns=categorical_columns,
            drop_first=True,
            dtype=float
        )

        X_test = pd.get_dummies(
            X_test,
            columns=categorical_columns,
            drop_first=True,
            dtype=float
        )

    # --------------------------------------------------------
    # Make validation/test columns exactly match training
    # --------------------------------------------------------

    X_validation = X_validation.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_validation = X_validation.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Replace infinity
    # --------------------------------------------------------

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_validation = X_validation.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return (
        X_train,
        X_validation,
        X_test
    )


# ============================================================
# 5. IMPUTATION
# ============================================================

def impute_data(
    X_train,
    X_validation,
    X_test
):

    # --------------------------------------------------------
    # CRITICAL:
    # Fit imputer ONLY on training data
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    X_train_processed = (
        imputer.fit_transform(
            X_train
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    X_validation_processed = (
        imputer.transform(
            X_validation
        )
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    X_test_processed = (
        imputer.transform(
            X_test
        )
    )

    return (
        X_train_processed,
        X_validation_processed,
        X_test_processed
    )


# ============================================================
# 6. MAIN PREPROCESSING PIPELINE
# ============================================================

def run_preprocessing():

    print("=" * 70)
    print("PREPROCESSING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Create processed directory
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 1. LOAD EXISTING SPLITS
    # ========================================================

    print(
        "\n[1/6] Loading train / validation / test splits..."
    )

    (
        train_df,
        validation_df,
        test_df
    ) = load_split_data()

    # ========================================================
    # 2. FEATURE ENGINEERING
    # ========================================================

    print(
        "\n[2/6] Creating engineered features..."
    )

    train_df = create_engineered_features(
        train_df
    )

    validation_df = create_engineered_features(
        validation_df
    )

    test_df = create_engineered_features(
        test_df
    )

    # --------------------------------------------------------
    # Display engineered features
    # --------------------------------------------------------

    engineered_features = [
        "rooms_per_household",
        "population_per_household",
        "bedrooms_per_household",
    ]

    print(
        "[INFO] Engineered features:"
    )

    for feature in engineered_features:

        if feature in train_df.columns:

            print(
                f"[OK] {feature}"
            )

    # ========================================================
    # 2B. APPLY SELECTED FEATURES
    # ========================================================

    print(
        "\n[2B] Applying selected features..."
    )

    (
        train_df,
        validation_df,
        test_df,
        selected_features
    ) = apply_selected_features(
        train_df,
        validation_df,
        test_df
    )

    # --------------------------------------------------------
    # Display selected feature count
    # --------------------------------------------------------

    print(
        f"[INFO] Total selected raw features: "
        f"{len(selected_features)}"
    )

    # ========================================================
    # 3. SEPARATE TARGET
    # ========================================================

    print(
        "\n[3/6] Separating target variable..."
    )

    (
        X_train,
        y_train
    ) = separate_target(
        train_df
    )

    (
        X_validation,
        y_validation
    ) = separate_target(
        validation_df
    )

    (
        X_test,
        y_test
    ) = separate_target(
        test_df
    )

    # --------------------------------------------------------
    # Remove rows with missing target
    # --------------------------------------------------------

    train_valid = y_train.notna()
    validation_valid = y_validation.notna()
    test_valid = y_test.notna()

    X_train = X_train.loc[
        train_valid
    ].reset_index(drop=True)

    y_train = y_train.loc[
        train_valid
    ].reset_index(drop=True)

    X_validation = X_validation.loc[
        validation_valid
    ].reset_index(drop=True)

    y_validation = y_validation.loc[
        validation_valid
    ].reset_index(drop=True)

    X_test = X_test.loc[
        test_valid
    ].reset_index(drop=True)

    y_test = y_test.loc[
        test_valid
    ].reset_index(drop=True)

    # ========================================================
    # 4. ENCODING
    # ========================================================

    print(
        "\n[4/6] Encoding categorical variables..."
    )

    (
        X_train,
        X_validation,
        X_test
    ) = encode_data(
        X_train,
        X_validation,
        X_test
    )

    # --------------------------------------------------------
    # Store final model feature names
    # --------------------------------------------------------

    feature_names = (
        X_train.columns.tolist()
    )

    print(
        f"[INFO] Model features: "
        f"{len(feature_names)}"
    )

    # ========================================================
    # 5. IMPUTATION
    # ========================================================

    print(
        "\n[5/6] Applying training-fitted imputation..."
    )

    (
        X_train_processed,
        X_validation_processed,
        X_test_processed
    ) = impute_data(
        X_train,
        X_validation,
        X_test
    )

    # ========================================================
    # 6. SAVE
    # ========================================================

    print(
        "\n[6/6] Saving processed datasets..."
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names
    )

    train_processed[TARGET_COL] = (
        y_train.to_numpy()
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_processed = pd.DataFrame(
        X_validation_processed,
        columns=feature_names
    )

    validation_processed[TARGET_COL] = (
        y_validation.to_numpy()
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names
    )

    test_processed[TARGET_COL] = (
        y_test.to_numpy()
    )

    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    train_processed.to_csv(
        TRAIN_OUTPUT_PATH,
        index=False
    )

    validation_processed.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False
    )

    test_processed.to_csv(
        TEST_OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # SAVE VALIDATION ARRAYS
    # Used by Streamlit diagnostics
    # ========================================================

    np.save(
        X_VAL_PATH,
        X_validation_processed
    )

    np.save(
        Y_VAL_PATH,
        y_validation.to_numpy(
            dtype=float
        )
    )

    # ========================================================
    # SAVE FEATURE NAMES
    # ========================================================

    pd.DataFrame({
        "feature_name": feature_names
    }).to_csv(
        FEATURE_NAMES_PATH,
        index=False
    )

    # ========================================================
    # OUTPUT INFORMATION
    # ========================================================

    print(
        f"\n[SAVED] {TRAIN_OUTPUT_PATH}"
    )

    print(
        f"[SAVED] {VALIDATION_OUTPUT_PATH}"
    )

    print(
        f"[SAVED] {TEST_OUTPUT_PATH}"
    )

    print(
        f"[SAVED] {X_VAL_PATH}"
    )

    print(
        f"[SAVED] {Y_VAL_PATH}"
    )

    print(
        f"[SAVED] {FEATURE_NAMES_PATH}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PREPROCESSING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        f"Training rows     : "
        f"{len(train_processed)}"
    )

    print(
        f"Validation rows   : "
        f"{len(validation_processed)}"
    )

    print(
        f"Test rows         : "
        f"{len(test_processed)}"
    )

    print(
        f"Selected features : "
        f"{len(selected_features)}"
    )

    print(
        f"Model features    : "
        f"{len(feature_names)}"
    )

    print(
        "\n[SUCCESS] Preprocessing stage completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_preprocessing()