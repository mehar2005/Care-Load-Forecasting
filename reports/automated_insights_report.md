# Predictive Forecasting of Care Load & Placement Demand

## Executive summary

This project analyses 720 observations across 6 fields, covering 2023-01-12 to 2025-12-21. It prepares the operational time series, explores historical patterns, engineers forecasting features, compares baseline and machine-learning approaches, and produces forecast and risk-context outputs.

## Dataset and data quality

- Observed records: 720.
- Calendar days in the date range: 1075.
- Calendar dates without observations: 355.
- Missing cells after cleaning: 0.
- Missing calendar dates were preserved rather than automatically imputed.

## Descriptive findings

- HHS care load ranged from 1,972 to 11,516; its mean was 6,061.3.
- HHS discharges ranged from 0 to 505; their mean was 173.4 per observed report.
- Strong correlations between some operational variables are useful for exploration but do not establish causal relationships.

## Forecast model comparison

| Target | Best model by holdout MAE | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| `hhs_care_load` | Naive Persistence | 10.42 | 14.21 | 0.993 |
| `hhs_discharges` | Naive Persistence | 5.63 | 7.64 | -0.442 |

The naïve persistence model is a strong benchmark on the current chronological holdout. More complex models should only be selected if they consistently improve performance under appropriate time-aware validation.

## Recommendations

1. Obtain a newer source-data snapshot before presenting forecasts as current.
2. Confirm the operational meaning and reporting cadence of missing calendar dates with the data provider.
3. Evaluate each required calendar-day horizon separately using rolling-origin validation.
4. Compare all models with simple baselines and report MAE and RMSE alongside R².
5. Use capacity-specific thresholds only when authoritative capacity data are available.
6. Re-run the complete pipeline whenever the source dataset is updated.

## Limitations and responsible interpretation

- The source ends on 2025-12-21; forecasts from this snapshot are not current operational estimates.
- There are missing calendar dates. They were not imputed, and a next-observation forecast is not equivalent to a next-calendar-day forecast.
- Pearson correlations are descriptive and do not imply causation; time trends and correlated variables may affect them.
- Historical percentile risk flags are not capacity alerts because facility capacity and staffing thresholds are not provided.
- Empirical error bands are not guaranteed prediction intervals and should not be interpreted as calibrated coverage guarantees.

## Reproducibility

Run `python -m src.automated_insights` from the project root to regenerate this report and its machine-readable summary.
