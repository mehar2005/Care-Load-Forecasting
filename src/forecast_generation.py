from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_care_load.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "rolling_validation_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
FORECAST_PATH = OUTPUT_DIR / "future_forecasts.csv"
REPORT_PATH = OUTPUT_DIR / "forecast_generation_report.json"
TARGETS = ["hhs_care_load", "hhs_discharges"]


def load_history():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python -m src.data_preparation` first.")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    if df.empty or not set(["date", *TARGETS]).issubset(df.columns):
        raise ValueError("Cleaned dataset is empty or missing required target columns.")
    return df.reset_index(drop=True)


def error_radii():
    if not VALIDATION_PATH.exists():
        return {"hhs_care_load": None, "hhs_discharges": None}, "Unavailable; run validation first."
    predictions = pd.read_csv(VALIDATION_PATH)
    radii = {}
    for target in TARGETS:
        errors = predictions.loc[predictions["target"] == target, "absolute_error"].dropna()
        radii[target] = float(errors.quantile(0.90)) if len(errors) else None
    return radii, "90th percentile of rolling out-of-fold absolute errors; observed-report horizon, not calendar-day calibrated."


def generate_forecasts(history, horizons=(7, 14, 30)):
    last_date = history["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=max(horizons), freq="D")
    radii, interval_method = error_radii()
    rows = []

    care_last = float(history["hhs_care_load"].iloc[-1])
    discharge_recent = history["hhs_discharges"].tail(28).astype(float)
    discharge_forecast = float(discharge_recent.median())

    for horizon in horizons:
        for day_number, date in enumerate(future_dates[:horizon], start=1):
            for target, point in (("hhs_care_load", care_last), ("hhs_discharges", discharge_forecast)):
                radius = radii[target]
                lower = max(0.0, point - radius) if radius is not None else np.nan
                upper = point + radius if radius is not None else np.nan
                rows.append({
                    "horizon_days": horizon,
                    "forecast_step_day": day_number,
                    "date": date,
                    "target": target,
                    "forecast": point,
                    "lower_90_empirical": lower,
                    "upper_90_empirical": upper,
                    "forecast_method": "Persistence for care load; trailing 28-observation median for discharges",
                    "uncertainty_method": interval_method,
                })
    return pd.DataFrame(rows), radii, interval_method


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    forecasts, radii, interval_method = generate_forecasts(history)
    forecasts.to_csv(FORECAST_PATH, index=False, date_format="%Y-%m-%d")
    report = {
        "history_start_date": history["date"].min().strftime("%Y-%m-%d"),
        "history_end_date": history["date"].max().strftime("%Y-%m-%d"),
        "last_observed_care_load": float(history["hhs_care_load"].iloc[-1]),
        "discharge_point_forecast": float(history["hhs_discharges"].tail(28).median()),
        "forecast_horizons_calendar_days": [7, 14, 30],
        "forecast_dates_begin": forecasts["date"].min().strftime("%Y-%m-%d"),
        "forecast_dates_end": forecasts["date"].max().strftime("%Y-%m-%d"),
        "uncertainty_radii": radii,
        "uncertainty_method": interval_method,
        "limitations": [
            "Calendar gaps in the source were not imputed.",
            "Care-load forecast is a persistence baseline; it is not a fitted multi-step dynamic model.",
            "Discharge forecast uses the median of the latest 28 observed reports.",
            "Empirical error bands are calibrated on next-observed-report errors and are not validated for each calendar-day horizon.",
            "Use as a transparent baseline only; do not interpret as operational advice or a guaranteed prediction interval."
        ]
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FORECAST GENERATION COMPLETED")
    print(f"History: {report['history_start_date']} to {report['history_end_date']} ({len(history)} observed records)")
    print(f"Forecast dates: {report['forecast_dates_begin']} to {report['forecast_dates_end']}")
    print("Horizons: 7, 14, and 30 calendar days")
    print(f"Last observed HHS care load: {report['last_observed_care_load']:.0f}")
    print(f"Discharge baseline (recent 28-report median): {report['discharge_point_forecast']:.1f}")
    print(f"Saved: {FORECAST_PATH}\nSaved: {REPORT_PATH}")
    print("Caution: empirical bands are not calendar-horizon calibrated prediction intervals.")


if __name__ == "__main__":
    main()
