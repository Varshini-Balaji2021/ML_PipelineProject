from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "housing.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "final_engineered_train.csv"
)

TARGET_COL = "median_house_value"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_engineered_features(df):

    result = df.copy()

    # --------------------------------------------------------
    # 1. Rooms per household
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
    # 2. Population per household
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
    # 3. Bedrooms per household
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
# MAIN
# ============================================================

def run_feature_engineering():

    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    if not RAW_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_DATA_PATH}"
        )

    print(
        f"\n[INFO] Loading dataset:"
    )

    print(
        f"       {RAW_DATA_PATH}"
    )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    if df.empty:

        raise ValueError(
            "Input dataset is empty."
        )

    if TARGET_COL not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COL}' "
            "not found."
        )

    print(
        f"[INFO] Original shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Create features
    # --------------------------------------------------------

    print(
        "\n[1/2] Creating engineered features..."
    )

    engineered_df = create_engineered_features(
        df
    )

    engineered_features = [
        feature
        for feature in [
            "rooms_per_household",
            "population_per_household",
            "bedrooms_per_household",
        ]
        if feature in engineered_df.columns
    ]

    print(
        f"[INFO] Engineered features created: "
        f"{len(engineered_features)}"
    )

    for feature in engineered_features:

        print(
    f"[OK] {feature}"
)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\n[2/2] Saving engineered dataset..."
    )

    engineered_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"[SAVED] {OUTPUT_PATH}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Original features : {df.shape[1] - 1}"
    )

    print(
        f"Engineered        : {len(engineered_features)}"
    )

    print(
        f"Final columns     : {engineered_df.shape[1]}"
    )

    print(
        "\n[SUCCESS] Feature engineering stage completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_feature_engineering()