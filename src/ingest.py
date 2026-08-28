from pathlib import Path
import kagglehub
import pandas as pd

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "housing.csv"


def load_housing_data() -> pd.DataFrame:
    """Download the Kaggle dataset and save a local copy to data/raw/."""
    # Ensure the target directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download latest version from Kaggle
    dataset_dir = Path(kagglehub.dataset_download("harrywang/housing"))
    source_file = dataset_dir / "housing.csv"

    # Load and save a persistent copy to data/raw/
    df = pd.read_csv(source_file)
    df.to_csv(RAW_DATA_PATH, index=False)

    return df


if __name__ == "__main__":
    housing_df = load_housing_data()

    print("Dataset loaded successfully.")
    print("Shape:", housing_df.shape)
    print("\nColumns:")
    print(housing_df.columns.tolist())
    print("\nFirst five records:")
    print(housing_df.head())
    print("\nRaw copy saved to:")
    print(RAW_DATA_PATH)