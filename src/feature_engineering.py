from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_care_load.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_PATH = OUTPUT_DIR / "forecasting_features.csv"
MODEL_PATH = OUTPUT_DIR / "model_ready_features.csv"
FEATURE_REPORT_PATH = OUTPUT_DIR / "feature_engineering_report.json"

TARGET_COLUMNS = ["hhs_care_load", "hhs_discharges"]
SOURCE_NUMERIC_COLUMNS = [
    "cbp_daily_intake",
    "cbp_custody",
    "transfers_from_cbp_to_hhs",
    "hhs_care_load",
    "hhs_discharges",
]
LAG_PERIODS = [1, 2, 3, 7]
ROLLING_WINDOWS = [3, 7]

def load_cleaned_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_PATH}. Run `python -m src.data_preparation` first."
        )
    data = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    required = ["date", *SOURCE_NUMERIC_COLUMNS]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    if data.empty:
        raise ValueError("The cleaned dataset is empty.")
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for column in SOURCE_NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[SOURCE_NUMERIC_COLUMNS].isna().any().any():
        raise ValueError("Unexpected missing or non-numeric values in cleaned data.")
    return data

def build_features(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build leakage-aware features for forecasting the next observed record.

    Lag and rolling features use only earlier observed records. Because the source
    has missing calendar dates, lag_1_observation means the previous available
    report, not necessarily the previous calendar day.
    """
    data = data.copy().sort_values("date").reset_index(drop=True)
    features = data.copy()
    date = features["date"]
    features["year"] = date.dt.year
    features["month"] = date.dt.month
    features["day_of_month"] = date.dt.day
    features["day_of_week"] = date.dt.dayofweek
    features["day_of_year"] = date.dt.dayofyear
    features["week_of_year"] = date.dt.isocalendar().week.astype(int)
    features["is_weekend"] = (date.dt.dayofweek >= 5).astype(int)
    features["days_since_previous_observation"] = date.diff().dt.days

    for column in SOURCE_NUMERIC_COLUMNS:
        for lag in LAG_PERIODS:
            features[f"{column}_lag_{lag}_observation"] = features[column].shift(lag)
        for window in ROLLING_WINDOWS:
            prior = features[column].shift(1)
            features[f"{column}_rolling_mean_{window}_observations"] = prior.rolling(
                window=window, min_periods=window
            ).mean()
            features[f"{column}_rolling_std_{window}_observations"] = prior.rolling(
                window=window, min_periods=window
            ).std()
        features[f"{column}_change_from_previous_observation"] = features[column].shift(1).diff()

    feature_columns = [column for column in features.columns if column not in ["date", *TARGET_COLUMNS]]
    model_columns = ["date", *feature_columns, *TARGET_COLUMNS]
    model_ready = features[model_columns].dropna().reset_index(drop=True)
    report = {
        "source_rows": int(len(data)),
        "feature_rows_before_warmup_removal": int(len(features)),
        "model_ready_rows": int(len(model_ready)),
        "rows_removed_due_to_lag_and_rolling_warmup": int(len(features) - len(model_ready)),
        "target_columns": TARGET_COLUMNS,
        "lag_periods_are_observations_not_calendar_days": True,
        "lag_periods": LAG_PERIODS,
        "rolling_windows_are_observations_not_calendar_days": True,
        "rolling_windows": ROLLING_WINDOWS,
        "rolling_features_shifted_to_use_prior_observations_only": True,
        "calendar_gaps_imputed": False,
        "feature_count_excluding_date_and_targets": int(len(feature_columns)),
        "feature_columns": feature_columns,
    }
    return features, model_ready, report

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_cleaned_data()
    features, model_ready, report = build_features(data)
    features.to_csv(FEATURE_PATH, index=False, date_format="%Y-%m-%d")
    model_ready.to_csv(MODEL_PATH, index=False, date_format="%Y-%m-%d")
    with FEATURE_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 60)
    print(f"Source observations: {report['source_rows']:,}")
    print(f"Generated predictors: {report['feature_count_excluding_date_and_targets']:,}")
    print(f"Model-ready observations: {report['model_ready_rows']:,}")
    print(f"Rows removed for lag/rolling warm-up: {report['rows_removed_due_to_lag_and_rolling_warmup']:,}")
    print("Lag/rolling windows refer to previous observed records, not calendar days.")
    print("All lag and rolling features use prior observations only.")
    print("Missing dates were not imputed.")
    print("\nFiles saved:")
    print(f"  {FEATURE_PATH}")
    print(f"  {MODEL_PATH}")
    print(f"  {FEATURE_REPORT_PATH}")

if __name__ == "__main__":
    main()
