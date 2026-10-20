from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
FEATURES_PATH = DATA / "model_ready_features.csv"
METRICS_PATH = DATA / "forecast_model_metrics.csv"
VALIDATION_PATH = DATA / "rolling_validation_metrics.csv"
COMPARISON_PATH = DATA / "model_comparison_summary.csv"
IMPORTANCE_PATH = DATA / "permutation_feature_importance.csv"
REPORT_PATH = DATA / "model_explainability_report.json"
TARGETS = ["hhs_care_load", "hhs_discharges"]


def run():
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not FEATURES_PATH.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError("Run feature engineering and forecasting before model explainability.")

    metrics = pd.read_csv(METRICS_PATH)
    metrics = metrics.sort_values(["target", "mae"]).reset_index(drop=True)
    metrics["mae_rank"] = metrics.groupby("target")["mae"].rank(method="min").astype(int)
    metrics["rmse_rank"] = metrics.groupby("target")["rmse"].rank(method="min").astype(int)
    metrics.to_csv(COMPARISON_PATH, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, target, title in zip(axes, TARGETS, ["HHS care load", "HHS discharges"]):
        part = metrics[metrics["target"] == target].sort_values("mae")
        ax.barh(part["model"], part["mae"])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Mean Absolute Error (lower is better)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Chronological holdout model comparison")
    fig.tight_layout()
    fig.savefig(FIGURES / "12_model_comparison_mae.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ["date", *TARGETS]]
    X_all = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X_all.isna().any().any():
        raise ValueError("Feature matrix contains missing or non-numeric values.")

    importance_rows = []
    for target in TARGETS:
        X = X_all.iloc[:-1].reset_index(drop=True)
        y = df[target].shift(-1).iloc[:-1].reset_index(drop=True)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.8, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        perm = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42, scoring="neg_mean_absolute_error", n_jobs=-1)
        for feature, mean, std in zip(feature_cols, perm.importances_mean, perm.importances_std):
            importance_rows.append({"target": target, "feature": feature, "importance_mean": float(mean), "importance_std": float(std)})
        pred = model.predict(X_test)
        importance_rows.append({"target": target, "feature": "__DIAGNOSTIC__", "importance_mean": float(mean_absolute_error(y_test, pred)), "importance_std": float(np.sqrt(mean_squared_error(y_test, pred)))} )

    imp = pd.DataFrame(importance_rows)
    diagnostic = imp[imp["feature"] == "__DIAGNOSTIC__"].copy()
    imp = imp[imp["feature"] != "__DIAGNOSTIC__"].copy()
    imp.to_csv(IMPORTANCE_PATH, index=False)

    for target, title, filename in [("hhs_care_load", "HHS care-load feature importance", "13_feature_importance_care_load.png"), ("hhs_discharges", "HHS discharge feature importance", "14_feature_importance_discharges.png")]:
        part = imp[imp["target"] == target].sort_values("importance_mean", ascending=False).head(15).sort_values("importance_mean")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(part["feature"], part["importance_mean"], xerr=part["importance_std"])
        ax.set_title(title + " (permutation importance)")
        ax.set_xlabel("Increase in MAE when feature is shuffled")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    rolling = pd.read_csv(VALIDATION_PATH).to_dict(orient="records") if VALIDATION_PATH.exists() else []
    best = {}
    for target in TARGETS:
        part = metrics[metrics["target"] == target].sort_values("mae")
        best[target] = {"model": str(part.iloc[0]["model"]), "mae": float(part.iloc[0]["mae"]), "rmse": float(part.iloc[0]["rmse"]), "r2": float(part.iloc[0]["r2"])}
    report = {
        "comparison_source": str(METRICS_PATH.relative_to(ROOT)),
        "holdout_best_by_mae": best,
        "diagnostic_random_forest": {row["target"]: {"mae": float(row["importance_mean"]), "rmse": float(row["importance_std"])} for _, row in diagnostic.iterrows()},
        "feature_importance_method": "Permutation importance on chronological holdout; importance is the increase in MAE after shuffling a feature.",
        "limitations": [
            "Permutation importance measures predictive usefulness for this split, not causation.",
            "Correlated features can share or dilute importance.",
            "The target is the next observed report, not necessarily the next calendar day.",
            "The supplied data ends on 2025-12-21; results are historical experiments, not current operational forecasts.",
            "The existing benchmark results and diagnostic Random Forest results are reported separately because their evaluation purpose differs."
        ],
        "rolling_validation_rows": len(rolling)
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Model comparison and explainability completed.")
    print(f"Saved: {COMPARISON_PATH}")
    print(f"Saved: {IMPORTANCE_PATH}")
    print(f"Saved: {REPORT_PATH}")
    print("Best holdout models by MAE:")
    for target, result in best.items():
        print(f"  {target}: {result['model']} (MAE={result['mae']:.3f})")

if __name__ == "__main__":
    run()
