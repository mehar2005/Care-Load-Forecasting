from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "HHS_Unaccompanied_Alien_Children_Program.csv"
)


def load_dataset() -> pd.DataFrame:
    """Load the original HHS care-load CSV without modifying its contents."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}\n"
            "Place the original CSV in the project's data/raw directory."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    if df.empty:
        raise ValueError("The dataset was loaded, but it contains no rows.")

    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Display a first-pass overview of the raw dataset."""
    print("\nDATASET OVERVIEW")
    print("=" * 60)
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nCOLUMN NAMES")
    print("=" * 60)
    for column in df.columns:
        print(column)

    print("\nDATA TYPES")
    print("=" * 60)
    print(df.dtypes.to_string())

    print("\nMISSING CELL VALUES")
    print("=" * 60)
    print(df.isna().sum().to_string())

    print("\nFIRST FIVE ROWS")
    print("=" * 60)
    print(df.head().to_string(index=False))

    print("\nLAST FIVE ROWS")
    print("=" * 60)
    print(df.tail().to_string(index=False))


if __name__ == "__main__":
    dataset = load_dataset()
    inspect_dataset(dataset)
