from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw"
    / "HHS_Unaccompanied_Alien_Children_Program.csv"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_care_load.csv"
DAILY_CALENDAR_PATH = PROCESSED_DIR / "daily_care_load_calendar.csv"
QUALITY_REPORT_PATH = PROCESSED_DIR / "data_quality_report.json"

COLUMN_RENAME_MAP = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "cbp_daily_intake",
    "Children in CBP custody": "cbp_custody",
    "Children transferred out of CBP custody": "transfers_from_cbp_to_hhs",
    "Children in HHS Care": "hhs_care_load",
    "Children discharged from HHS Care": "hhs_discharges",
}
NUMERIC_COLUMNS = [
    "cbp_daily_intake",
    "cbp_custody",
    "transfers_from_cbp_to_hhs",
    "hhs_care_load",
    "hhs_discharges",
]
EXPECTED_COLUMNS = list(COLUMN_RENAME_MAP)


def load_raw_dataset() -> pd.DataFrame:
    """Load the original CSV without changing the source file."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}. "
            "Place the original CSV in data/raw/."
        )
    df = pd.read_csv(RAW_DATA_PATH)
    if df.empty:
        raise ValueError("The dataset contains no rows.")
    return df


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate, standardize, and chronologically sort observed records.

    Missing calendar dates are reported, not automatically imputed. The source
    appears to have a recurring reporting pattern, so filling dates with
    interpolated values would require a separate, justified modelling decision.
    """
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    data = df[EXPECTED_COLUMNS].copy()
    original_rows = len(data)

    data["Date"] = pd.to_datetime(
        data["Date"], format="%d-%b-%y", errors="coerce"
    )
    invalid_dates = int(data["Date"].isna().sum())
    data = data.dropna(subset=["Date"])

    for col in NUMERIC_COLUMNS:
        source_col = next(
            original for original, standardized in COLUMN_RENAME_MAP.items()
            if standardized == col
        )
        values = (
            data[source_col].astype("string").str.replace(",", "", regex=False)
        )
        data[source_col] = pd.to_numeric(values, errors="coerce")

    data = data.rename(columns=COLUMN_RENAME_MAP)
    numeric_invalid_before_drop = int(data[NUMERIC_COLUMNS].isna().sum().sum())
    rows_with_invalid_numeric = int(data[NUMERIC_COLUMNS].isna().any(axis=1).sum())
    data = data.dropna(subset=NUMERIC_COLUMNS)

    duplicate_date_rows = int(data.duplicated(subset=["date"], keep="first").sum())
    data = data.sort_values("date").drop_duplicates(subset=["date"], keep="first")
    data = data.reset_index(drop=True)

    negative_counts = {
        col: int((data[col] < 0).sum()) for col in NUMERIC_COLUMNS
    }
    observed_dates = pd.DatetimeIndex(data["date"])
    if data.empty:
        raise ValueError("No valid rows remain after parsing and validation.")

    calendar = pd.date_range(data["date"].min(), data["date"].max(), freq="D")
    missing_dates = calendar.difference(observed_dates)
    date_gaps = data["date"].diff().dt.days.dropna().astype(int)
    gap_counts = {str(int(k)): int(v) for k, v in date_gaps.value_counts().sort_index().items()}
    weekday_counts = {
        str(k): int(v)
        for k, v in data["date"].dt.day_name().value_counts().items()
    }

    quality_report = {
        "source_file": RAW_DATA_PATH.name,
        "original_rows": int(original_rows),
        "rows_after_cleaning": int(len(data)),
        "columns_after_standardization": list(data.columns),
        "invalid_dates_removed": invalid_dates,
        "rows_with_invalid_numeric_values_removed": rows_with_invalid_numeric,
        "invalid_numeric_cells_before_row_removal": numeric_invalid_before_drop,
        "duplicate_date_rows_removed": duplicate_date_rows,
        "remaining_missing_cells": int(data.isna().sum().sum()),
        "negative_value_counts_by_column": negative_counts,
        "start_date": data["date"].min().strftime("%Y-%m-%d"),
        "end_date": data["date"].max().strftime("%Y-%m-%d"),
        "observed_unique_dates": int(data["date"].nunique()),
        "calendar_days_in_range": int(len(calendar)),
        "calendar_dates_without_observations": int(len(missing_dates)),
        "observation_interval_days_counts": gap_counts,
        "observations_by_weekday": weekday_counts,
        "missing_dates_handling": "Not imputed; retained as explicit gaps in the calendar file.",
    }
    return data, quality_report


def save_prepared_data(data: pd.DataFrame, quality_report: dict) -> None:
    """Save cleaned observations, a full daily calendar, and a quality report."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(CLEANED_DATA_PATH, index=False, date_format="%Y-%m-%d")

    daily_calendar = data.set_index("date").asfreq("D")
    daily_calendar.index.name = "date"
    daily_calendar["is_observed_record"] = daily_calendar[NUMERIC_COLUMNS].notna().all(axis=1)
    daily_calendar.to_csv(DAILY_CALENDAR_PATH, date_format="%Y-%m-%d")

    with QUALITY_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(quality_report, file, indent=4)


def main() -> None:
    raw_data = load_raw_dataset()
    cleaned_data, report = prepare_dataset(raw_data)
    save_prepared_data(cleaned_data, report)

    print("DATA PREPARATION COMPLETED")
    print("=" * 60)
    print(f"Original rows: {report['original_rows']:,}")
    print(f"Cleaned observed rows: {report['rows_after_cleaning']:,}")
    print(f"Date range: {report['start_date']} to {report['end_date']}")
    print(f"Calendar days in range: {report['calendar_days_in_range']:,}")
    print(f"Dates without observations: {report['calendar_dates_without_observations']:,}")
    print(f"Remaining missing cells in cleaned observations: {report['remaining_missing_cells']}")
    print("\nObservation intervals (days):")
    for interval, count in report["observation_interval_days_counts"].items():
        print(f"  {interval} day(s): {count}")
    print("\nObservations by weekday:")
    for weekday, count in report["observations_by_weekday"].items():
        print(f"  {weekday}: {count}")
    print("\nFiles saved:")
    print(f"  {CLEANED_DATA_PATH}")
    print(f"  {DAILY_CALENDAR_PATH}")
    print(f"  {QUALITY_REPORT_PATH}")
    print("\nNote: absent dates were not interpolated or filled with invented values.")


if __name__ == "__main__":
    main()
