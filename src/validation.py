from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_PATH = OUTPUT_DIR / "rolling_validation_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "rolling_validation_predictions.csv"
REPORT_PATH = OUTPUT_DIR / "validation_report.json"
TARGETS = ["hhs_care_load", "hhs_discharges"]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python -m src.feature_engineering` before validation.")
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    data = data.sort_values("date").reset_index(drop=True)
    if len(data) < 100:
        raise ValueError("At least 100 observations are required for rolling validation.")
    return data


def build_folds(n_rows, n_splits=5, min_train_fraction=0.5):
    first_test = int(n_rows * min_train_fraction)
    remaining = n_rows - first_test
    block_size = max(1, remaining // n_splits)
    folds = []
    start = first_test
    for fold in range(n_splits):
        end = n_rows if fold == n_splits - 1 else min(n_rows, start + block_size)
        if start >= n_rows or end <= start:
            break
        folds.append((start, end))
        start = end
    return folds


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    feature_cols = [c for c in data.columns if c not in ["date", *TARGETS]]
    X_all = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X_all.isna().any().any():
        raise ValueError("Feature matrix contains missing or non-numeric values.")

    X = X_all.iloc[:-1].reset_index(drop=True)
    folds = build_folds(len(X), n_splits=5, min_train_fraction=0.5)
    model_factories = {
        "Naive Persistence": None,
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        ),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, max_features=0.8,
            random_state=42, n_jobs=-1
        ),
    }
    prediction_rows = []
    metric_rows = []

    for target in TARGETS:
        y = data[target].shift(-1).iloc[:-1].reset_index(drop=True)
        target_dates = data["date"].shift(-1).iloc[:-1].reset_index(drop=True)
        for fold_number, (start, end) in enumerate(folds, start=1):
            X_train, X_test = X.iloc[:start], X.iloc[start:end]
            y_train, y_test = y.iloc[:start], y.iloc[start:end]
            dates = target_dates.iloc[start:end]
            for model_name, factory in model_factories.items():
                if factory is None:
                    preds = data[target].iloc[start:end].to_numpy(dtype=float)
                else:
                    model = factory()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_train, y_train)
                        preds = np.maximum(0, model.predict(X_test))
                for date, actual, predicted in zip(dates, y_test, preds):
                    prediction_rows.append({
                        "target": target, "model": model_name, "fold": fold_number,
                        "date": date, "actual": float(actual), "predicted": float(predicted),
                        "absolute_error": float(abs(actual - predicted)),
                    })

    predictions = pd.DataFrame(prediction_rows)
    interval_summary = {}
    for (target, model_name), group in predictions.groupby(["target", "model"]):
        radius = float(group["absolute_error"].quantile(0.90))
        group_idx = group.index
        predictions.loc[group_idx, "lower_90_empirical"] = np.maximum(0, predictions.loc[group_idx, "predicted"] - radius)
        predictions.loc[group_idx, "upper_90_empirical"] = predictions.loc[group_idx, "predicted"] + radius
        y_true = group["actual"].to_numpy()
        y_pred = group["predicted"].to_numpy()
        metric_rows.append({
            "target": target, "model": model_name,
            "validation_folds": int(group["fold"].nunique()),
            "validation_observations": int(len(group)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "empirical_90_band_radius": radius,
            "empirical_90_band_coverage": float(((group["actual"] >= predictions.loc[group_idx, "lower_90_empirical"]) & (group["actual"] <= predictions.loc[group_idx, "upper_90_empirical"])).mean()),
        })
        interval_summary[f"{target}|{model_name}"] = {
            "absolute_error_90th_percentile": radius,
            "interpretation": "Empirical symmetric error band calibrated on rolling validation errors; not a guaranteed prediction interval."
        }

    metrics = pd.DataFrame(metric_rows).sort_values(["target", "mae"]).reset_index(drop=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d")
    metrics.to_csv(METRICS_PATH, index=False)
    report = {
        "validation_method": "expanding-window rolling-origin validation",
        "fold_count": len(folds),
        "initial_training_fraction": 0.5,
        "random_shuffling": False,
        "target_alignment": "Features at one observed report predict the target at the next observed report.",
        "calendar_gaps_imputed": False,
        "interval_method": "90th percentile of rolling out-of-fold absolute errors, used as a symmetric empirical error band.",
        "interval_caveat": "Bands are empirical and not guaranteed calibrated prediction intervals; future work should test horizon-specific coverage.",
        "results": metrics.to_dict(orient="records"),
        "intervals": interval_summary,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ROLLING-ORIGIN VALIDATION COMPLETED")
    print(f"Rows: {len(data)} | folds: {len(folds)} | features: {len(feature_cols)}")
    print(metrics[["target", "model", "validation_observations", "mae", "rmse", "r2", "empirical_90_band_coverage"]].to_string(index=False))
    print(f"\nSaved: {METRICS_PATH}\nSaved: {PREDICTIONS_PATH}\nSaved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
