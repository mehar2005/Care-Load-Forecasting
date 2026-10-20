from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_care_load.csv"
QUALITY_REPORT_PATH = PROCESSED_DIR / "data_quality_report.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
EDA_REPORT_PATH = PROCESSED_DIR / "eda_summary.json"

DATE_COLUMN = "date"
NUMERIC_COLUMNS = [
    "cbp_daily_intake",
    "cbp_custody",
    "transfers_from_cbp_to_hhs",
    "hhs_care_load",
    "hhs_discharges",
]
COLUMN_LABELS = {
    "cbp_daily_intake": "Daily CBP intake",
    "cbp_custody": "Children in CBP custody",
    "transfers_from_cbp_to_hhs": "Transfers from CBP to HHS",
    "hhs_care_load": "Children in HHS care",
    "hhs_discharges": "HHS discharges",
}

sns.set_theme(style="whitegrid", context="notebook")


def load_cleaned_data() -> pd.DataFrame:
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {CLEANED_DATA_PATH}. "
            "Run `python -m src.data_preparation` first."
        )
    data = pd.read_csv(CLEANED_DATA_PATH, parse_dates=[DATE_COLUMN])
    missing = [column for column in [DATE_COLUMN, *NUMERIC_COLUMNS] if column not in data.columns]
    if missing:
        raise ValueError(f"Required columns missing from cleaned dataset: {missing}")
    if data.empty:
        raise ValueError("The cleaned dataset contains no rows.")
    return data.sort_values(DATE_COLUMN).reset_index(drop=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_time_series_plot(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in ["hhs_care_load", "cbp_custody"]:
        ax.plot(
            data[DATE_COLUMN], data[column], marker=".", markersize=3,
            linewidth=1.2, label=COLUMN_LABELS[column]
        )
    ax.set_title("Observed Care Load Over Time")
    ax.set_xlabel("Reporting date")
    ax.set_ylabel("Number of children")
    ax.legend()
    save_figure(fig, "01_care_load_over_time.png")


def create_flow_plot(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in ["cbp_daily_intake", "transfers_from_cbp_to_hhs", "hhs_discharges"]:
        ax.plot(
            data[DATE_COLUMN], data[column], marker=".", markersize=3,
            linewidth=1.0, label=COLUMN_LABELS[column]
        )
    ax.set_title("Observed Intake, Transfers, and Discharges")
    ax.set_xlabel("Reporting date")
    ax.set_ylabel("Reported count")
    ax.legend()
    save_figure(fig, "02_intake_transfers_discharges.png")


def create_distributions(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, column in zip(axes.flat, NUMERIC_COLUMNS):
        ax.hist(data[column].dropna(), bins="auto", edgecolor="black", alpha=0.8)
        ax.set_title(COLUMN_LABELS[column])
        ax.set_xlabel("Reported count")
    axes.flat[-1].axis("off")
    fig.suptitle("Distributions of Numeric Variables", y=1.02)
    save_figure(fig, "03_numeric_distributions.png")


def create_boxplots(data: pd.DataFrame) -> None:
    long_data = data[NUMERIC_COLUMNS].melt(var_name="variable", value_name="value")
    long_data["variable"] = long_data["variable"].map(COLUMN_LABELS)
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [COLUMN_LABELS[column] for column in NUMERIC_COLUMNS]
    values = [data[column].dropna().to_numpy() for column in NUMERIC_COLUMNS]
    ax.boxplot(values, tick_labels=labels, showfliers=True)
    ax.set_title("Numeric Variables: Spread and Potential Outliers")
    ax.set_xlabel("")
    ax.set_ylabel("Reported count")
    ax.tick_params(axis="x", rotation=18)
    save_figure(fig, "04_numeric_boxplots.png")


def create_correlation_heatmap(data: pd.DataFrame) -> pd.DataFrame:
    correlation = data[NUMERIC_COLUMNS].corr(method="pearson")
    display_correlation = correlation.rename(index=COLUMN_LABELS, columns=COLUMN_LABELS)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(display_correlation, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pearson Correlation Between Numeric Variables")
    save_figure(fig, "05_correlation_heatmap.png")
    correlation.to_csv(PROCESSED_DIR / "correlation_matrix.csv", index=True)
    return correlation


def create_scatter_plots(data: pd.DataFrame) -> None:
    pairs = [
        ("transfers_from_cbp_to_hhs", "hhs_care_load"),
        ("hhs_discharges", "hhs_care_load"),
        ("cbp_daily_intake", "transfers_from_cbp_to_hhs"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (x_column, y_column) in zip(axes, pairs):
        ax.scatter(data[x_column], data[y_column], alpha=0.55, s=25)
        ax.set_xlabel(COLUMN_LABELS[x_column])
        ax.set_ylabel(COLUMN_LABELS[y_column])
        ax.set_title(f"{COLUMN_LABELS[x_column]} vs.\n{COLUMN_LABELS[y_column]}")
    save_figure(fig, "06_relationship_scatterplots.png")


def create_reporting_pattern_plots(data: pd.DataFrame) -> dict:
    date_differences = data[DATE_COLUMN].diff().dt.days.dropna().astype(int)
    weekday_counts = (
        data.assign(weekday=data[DATE_COLUMN].dt.day_name())
        .groupby("weekday").size()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fill_value=0)
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    interval_counts = date_differences.value_counts().sort_index()
    axes[0].bar(interval_counts.index.astype(str), interval_counts.values)
    axes[0].set_title("Days Between Consecutive Observations")
    axes[0].set_xlabel("Interval (days)")
    axes[0].set_ylabel("Number of intervals")
    axes[1].bar(weekday_counts.index, weekday_counts.values)
    axes[1].set_title("Observations by Weekday")
    axes[1].set_xlabel("Weekday")
    axes[1].set_ylabel("Number of observations")
    axes[1].tick_params(axis="x", rotation=35)
    save_figure(fig, "07_reporting_pattern.png")
    return {
        "interval_day_counts": {str(int(k)): int(v) for k, v in interval_counts.items()},
        "weekday_observation_counts": {str(k): int(v) for k, v in weekday_counts.items()},
        "median_interval_days": float(date_differences.median()) if not date_differences.empty else None,
        "mean_interval_days": float(date_differences.mean()) if not date_differences.empty else None,
    }


def build_summary(data: pd.DataFrame, correlation: pd.DataFrame, reporting_patterns: dict) -> dict:
    descriptive_stats = data[NUMERIC_COLUMNS].describe().round(3)
    statistics = {
        column: {str(index): float(value) for index, value in descriptive_stats[column].items()}
        for column in NUMERIC_COLUMNS
    }
    strongest_pairs = []
    for i, first in enumerate(NUMERIC_COLUMNS):
        for second in NUMERIC_COLUMNS[i + 1:]:
            value = float(correlation.loc[first, second])
            strongest_pairs.append({"variable_1": first, "variable_2": second, "pearson_correlation": round(value, 4)})
    strongest_pairs.sort(key=lambda item: abs(item["pearson_correlation"]), reverse=True)

    summary = {
        "dataset_rows": int(len(data)),
        "dataset_columns": int(len(data.columns)),
        "start_date": data[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "end_date": data[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "observed_date_count": int(data[DATE_COLUMN].nunique()),
        "numeric_summary_statistics": statistics,
        "pearson_correlations": {
            first: {second: float(correlation.loc[first, second]) for second in NUMERIC_COLUMNS}
            for first in NUMERIC_COLUMNS
        },
        "strongest_pairwise_correlations": strongest_pairs,
        "reporting_patterns": reporting_patterns,
        "analysis_notes": [
            "The observations are irregularly spaced; line charts connect observed records only and do not imply values were recorded on every intervening day.",
            "Correlation is descriptive and does not establish causation.",
            "Outliers should be investigated with domain context rather than removed automatically.",
            "Forecasting validation must respect chronological order and account for the observation cadence.",
        ],
    }
    return summary


def run_eda() -> dict:
    data = load_cleaned_data()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    create_time_series_plot(data)
    create_flow_plot(data)
    create_distributions(data)
    create_boxplots(data)
    correlation = create_correlation_heatmap(data)
    create_scatter_plots(data)
    reporting_patterns = create_reporting_pattern_plots(data)
    summary = build_summary(data, correlation, reporting_patterns)

    summary_stats = data[NUMERIC_COLUMNS].describe().round(3)
    summary_stats.to_csv(PROCESSED_DIR / "descriptive_statistics.csv")
    with EDA_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("EXPLORATORY DATA ANALYSIS COMPLETED")
    print("=" * 55)
    print(f"Observed records: {summary['dataset_rows']:,}")
    print(f"Date range: {summary['start_date']} to {summary['end_date']}")
    print(f"Unique observed dates: {summary['observed_date_count']:,}")
    print(f"Median days between observations: {reporting_patterns['median_interval_days']}")
    print("\nStrongest pairwise correlations (descriptive only):")
    for pair in summary["strongest_pairwise_correlations"][:5]:
        print(
            f"  {pair['variable_1']} vs {pair['variable_2']}: "
            f"{pair['pearson_correlation']:.3f}"
        )
    print("\nSaved figures:")
    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    print(f"\nEDA summary: {EDA_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("Note: observed dates are not interpolated or filled for EDA.")
    return summary


if __name__ == "__main__":
    run_eda()
