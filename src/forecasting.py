from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = OUTPUT_DIR / "forecast_model_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "forecast_test_predictions.csv"
REPORT_PATH = OUTPUT_DIR / "forecasting_report.json"
MODELS_DIR = PROJECT_ROOT / "models"
TARGETS = ["hhs_care_load", "hhs_discharges"]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}. Run `python -m src.feature_engineering` first."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    if df.empty or len(df) < 30:
        raise ValueError("At least 30 model-ready observations are required.")
    return df.sort_values("date").reset_index(drop=True)


def metric_row(target, model_name, y_true, y_pred, test_dates):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else None
    return {
        "target": target,
        "model": model_name,
        "test_observations": int(len(y_true)),
        "test_start_date": pd.Timestamp(test_dates.iloc[0]).strftime("%Y-%m-%d"),
        "test_end_date": pd.Timestamp(test_dates.iloc[-1]).strftime("%Y-%m-%d"),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_percent": mape,
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_forecasting(df):
    feature_cols = [c for c in df.columns if c not in ["date", *TARGETS]]
    X_all = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X_all.isna().any().any():
        raise ValueError("Predictor columns contain missing or non-numeric values.")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.8,
            random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        ),
    }
    metric_rows, prediction_frames, report_models = [], [], {}
    split_at = int(len(df) * 0.8)
    if split_at < 20 or len(df) - split_at < 10:
        raise ValueError("Not enough rows for the chronological train/test split.")

    for target in TARGETS:
        X = X_all.iloc[:-1].reset_index(drop=True)
        y = df[target].shift(-1).iloc[:-1].reset_index(drop=True)
        target_dates = df["date"].shift(-1).iloc[:-1].reset_index(drop=True)
        boundary = min(split_at, len(X) - 1)
        X_train, X_test = X.iloc[:boundary], X.iloc[boundary:]
        y_train, y_test = y.iloc[:boundary], y.iloc[boundary:]
        test_dates = target_dates.iloc[boundary:]

        baseline_pred = df[target].iloc[boundary:boundary + len(y_test)].to_numpy(dtype=float)
        metric_rows.append(metric_row(target, "Naive Persistence", y_test, baseline_pred, test_dates))
        prediction_frames.append(pd.DataFrame({
            "target": target, "model": "Naive Persistence", "date": test_dates,
            "actual": y_test, "predicted": baseline_pred
        }))

        target_models = {}
        for name, estimator in models.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_train, y_train)
                predictions = np.maximum(0, estimator.predict(X_test))
            metric_rows.append(metric_row(target, name, y_test, predictions, test_dates))
            prediction_frames.append(pd.DataFrame({
                "target": target, "model": name, "date": test_dates,
                "actual": y_test.to_numpy(), "predicted": predictions
            }))
            target_models[name] = estimator

        target_metrics = [r for r in metric_rows if r["target"] == target]
        best = min(target_metrics, key=lambda r: r["mae"])
        best_name = best["model"]
        report_models[target] = {
            "best_model_by_mae": best_name,
            "best_model_test_mae": best["mae"],
            "best_model_test_rmse": best["rmse"],
            "train_observations": int(len(X_train)),
            "test_observations": int(len(X_test)),
            "prediction_target": "next observed report, not necessarily next calendar day",
            "features_used": feature_cols,
        }
        if best_name in target_models:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(target_models[best_name], MODELS_DIR / f"best_model_{target}.joblib")

    metrics = pd.DataFrame(metric_rows).sort_values(["target", "mae"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metrics, predictions, {
        "source_rows": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "chronological_train_fraction": 0.8,
        "random_shuffling_used": False,
        "target_alignment": "Predict target at the next observed report from features available at the current report.",
        "calendar_gaps_imputed": False,
        "metrics": report_models,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    metrics, predictions, report = run_forecasting(df)
    metrics.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d")
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("FORECASTING MODEL DEVELOPMENT COMPLETED")
    print("=" * 60)
    print(f"Source observations: {len(df):,}")
    print(f"Predictor features: {report['feature_count']}")
    print("Validation: chronological 80/20 split; no random shuffle")
    print("Forecast target: next observed report, not necessarily next calendar day")
    print("\nModel comparison (sorted by target and MAE):")
    print(metrics[["target", "model", "mae", "rmse", "r2"]].to_string(index=False))
    print("\nFiles saved:")
    print(f"  {METRICS_PATH}")
    print(f"  {PREDICTIONS_PATH}")
    print(f"  {REPORT_PATH}")
    print(f"  {MODELS_DIR}")

if __name__ == "__main__":
    main()
