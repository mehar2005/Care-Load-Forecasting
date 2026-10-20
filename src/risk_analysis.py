from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
HISTORY_PATH = PROCESSED / "cleaned_care_load.csv"
FORECAST_PATH = PROCESSED / "future_forecasts.csv"
OUTPUT_PATH = PROCESSED / "forecast_risk_indicators.csv"
REPORT_PATH = PROCESSED / "risk_analysis_report.json"

TARGETS = {
    "hhs_care_load": "Children in HHS Care",
    "hhs_discharges": "Children discharged from HHS Care",
}


def load_inputs():
    if not HISTORY_PATH.exists() or not FORECAST_PATH.exists():
        raise FileNotFoundError(
            "Required inputs are missing. Run data preparation and forecast generation first."
        )
    history = pd.read_csv(HISTORY_PATH, parse_dates=["date"])
    forecasts = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    required_history = {"date", *TARGETS.keys()}
    if not required_history.issubset(history.columns):
        raise ValueError(f"History must contain columns: {sorted(required_history)}")
    required_forecast = {"date", "target", "forecast", "lower_90_empirical", "upper_90_empirical", "horizon_days"}
    if not required_forecast.issubset(forecasts.columns):
        raise ValueError(f"Forecasts must contain columns: {sorted(required_forecast)}")
    return history.sort_values("date"), forecasts


def run_risk_analysis():
    FIGURES.mkdir(parents=True, exist_ok=True)
    history, forecasts = load_inputs()
    rows = []
    thresholds = {}

    for target, label in TARGETS.items():
        observed = pd.to_numeric(history[target], errors="coerce").dropna()
        if observed.empty:
            continue
        q50 = float(observed.quantile(0.50))
        q75 = float(observed.quantile(0.75))
        q90 = float(observed.quantile(0.90))
        thresholds[target] = {
            "label": label,
            "observed_count": int(observed.count()),
            "historical_median": q50,
            "historical_75th_percentile": q75,
            "historical_90th_percentile": q90,
            "threshold_interpretation": "Descriptive historical percentile, not an operational capacity limit.",
        }
        target_forecasts = forecasts.loc[forecasts["target"] == target].copy()
        for _, row in target_forecasts.iterrows():
            value = float(row["forecast"])
            upper = float(row["upper_90_empirical"])
            if value >= q90:
                status = "At or above historical 90th percentile"
            elif value >= q75:
                status = "Between historical 75th and 90th percentiles"
            else:
                status = "Below historical 75th percentile"
            rows.append({
                "date": row["date"].date().isoformat(),
                "target": target,
                "target_label": label,
                "horizon_days": int(row["horizon_days"]),
                "forecast": value,
                "empirical_lower_band": float(row["lower_90_empirical"]),
                "empirical_upper_band": upper,
                "historical_median": q50,
                "historical_75th_percentile": q75,
                "historical_90th_percentile": q90,
                "point_forecast_percentile_status": status,
                "upper_band_reaches_historical_90th_percentile": bool(upper >= q90),
                "risk_interpretation": "Historical comparison only; not a capacity or safety alert.",
            })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    report = {
        "history_start_date": history["date"].min().date().isoformat(),
        "history_end_date": history["date"].max().date().isoformat(),
        "forecast_start_date": forecasts["date"].min().date().isoformat(),
        "forecast_end_date": forecasts["date"].max().date().isoformat(),
        "forecast_rows_assessed": int(len(result)),
        "historical_thresholds": thresholds,
        "interpretation_notes": [
            "Percentile thresholds are derived from observed historical values and are descriptive only.",
            "No facility capacity or staffing thresholds are supplied by this dataset, so no capacity breach is inferred.",
            "Empirical forecast bands use next-observed-report errors and are not calibrated separately for calendar-day horizons.",
            "The supplied data ends on 2025-12-21; forecast dates are historical projections from that origin, not current operational forecasts.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for target, label in TARGETS.items():
        hist = history[["date", target]].dropna().tail(120)
        pred = forecasts.loc[forecasts["target"] == target].copy()
        if hist.empty or pred.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(hist["date"], hist[target], label="Observed history", linewidth=2)
        ax.plot(pred["date"], pred["forecast"], label="Baseline forecast", linestyle="--")
        ax.fill_between(pred["date"], pred["lower_90_empirical"], pred["upper_90_empirical"], alpha=0.2, label="Empirical error band")
        threshold = thresholds[target]["historical_90th_percentile"]
        ax.axhline(threshold, linestyle=":", linewidth=1.5, label="Historical 90th percentile (descriptive)")
        ax.set_title(f"{label}: historical context and forecast indicators")
        ax.set_xlabel("Date")
        ax.set_ylabel("Children")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / f"11_risk_context_{target}.png", dpi=150)
        plt.close(fig)

    print("Risk analysis completed successfully.")
    print(f"Forecast rows assessed: {len(result)}")
    print(f"Indicators saved to: {OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Figures saved to: {FIGURES}")
    return result, report


if __name__ == "__main__":
    run_risk_analysis()
