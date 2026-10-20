from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / 'data' / 'processed'
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)


def read_json(name):
    path = PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(f'Required input not found: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(name):
    path = PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(f'Required input not found: {path}')
    return pd.read_csv(path)


def generate_insights():
    quality = read_json('data_quality_report.json')
    eda = read_json('eda_summary.json')
    metrics = read_csv('forecast_model_metrics.csv')
    validation_path = PROCESSED / 'rolling_validation_metrics.csv'
    validation = pd.read_csv(validation_path) if validation_path.exists() else pd.DataFrame()
    risk_path = PROCESSED / 'risk_analysis_report.json'
    risk = json.loads(risk_path.read_text(encoding='utf-8')) if risk_path.exists() else {}

    corr = eda.get('pearson_correlations', {})
    care_stats = eda['numeric_summary_statistics']['hhs_care_load']
    discharge_stats = eda['numeric_summary_statistics']['hhs_discharges']
    strongest = eda.get('strongest_pairwise_correlations', [])
    best_models = {}
    for target, group in metrics.groupby('target'):
        best = group.sort_values('mae').iloc[0]
        best_models[target] = {
            'model': str(best['model']),
            'mae': float(best['mae']),
            'rmse': float(best['rmse']),
            'r2': float(best['r2']),
            'test_observations': int(best['test_observations']),
            'test_start_date': str(best['test_start_date']),
            'test_end_date': str(best['test_end_date']),
        }

    insights = {
        'project_title': 'Predictive Forecasting of Care Load & Placement Demand',
        'dataset': {
            'rows': quality['original_rows'],
            'columns': len(quality['columns_after_standardization']),
            'start_date': quality['start_date'],
            'end_date': quality['end_date'],
            'calendar_days_in_range': quality['calendar_days_in_range'],
            'observed_dates': quality['observed_unique_dates'],
            'dates_without_observations': quality['calendar_dates_without_observations'],
            'missing_cells_after_cleaning': quality['remaining_missing_cells'],
        },
        'descriptive_findings': {
            'hhs_care_load_mean': care_stats['mean'],
            'hhs_care_load_min': care_stats['min'],
            'hhs_care_load_max': care_stats['max'],
            'hhs_discharges_mean': discharge_stats['mean'],
            'hhs_discharges_min': discharge_stats['min'],
            'hhs_discharges_max': discharge_stats['max'],
            'strongest_pairwise_correlations': strongest[:5],
        },
        'holdout_best_models_by_mae': best_models,
        'rolling_validation_available': not validation.empty,
        'risk_summary_available': bool(risk),
        'limitations': [
            'The source ends on 2025-12-21; forecasts from this snapshot are not current operational estimates.',
            'There are missing calendar dates. They were not imputed, and a next-observation forecast is not equivalent to a next-calendar-day forecast.',
            'Pearson correlations are descriptive and do not imply causation; time trends and correlated variables may affect them.',
            'Historical percentile risk flags are not capacity alerts because facility capacity and staffing thresholds are not provided.',
            'Empirical error bands are not guaranteed prediction intervals and should not be interpreted as calibrated coverage guarantees.',
        ],
    }

    lines = [
        '# Predictive Forecasting of Care Load & Placement Demand',
        '',
        '## Executive summary',
        '',
        f"This project analyses {insights['dataset']['rows']} observations across {insights['dataset']['columns']} fields, covering {insights['dataset']['start_date']} to {insights['dataset']['end_date']}. It prepares the operational time series, explores historical patterns, engineers forecasting features, compares baseline and machine-learning approaches, and produces forecast and risk-context outputs.",
        '',
        '## Dataset and data quality',
        '',
        f"- Observed records: {insights['dataset']['observed_dates']}.",
        f"- Calendar days in the date range: {insights['dataset']['calendar_days_in_range']}.",
        f"- Calendar dates without observations: {insights['dataset']['dates_without_observations']}.",
        f"- Missing cells after cleaning: {insights['dataset']['missing_cells_after_cleaning']}.",
        '- Missing calendar dates were preserved rather than automatically imputed.',
        '',
        '## Descriptive findings',
        '',
        f"- HHS care load ranged from {care_stats['min']:,.0f} to {care_stats['max']:,.0f}; its mean was {care_stats['mean']:,.1f}.",
        f"- HHS discharges ranged from {discharge_stats['min']:,.0f} to {discharge_stats['max']:,.0f}; their mean was {discharge_stats['mean']:,.1f} per observed report.",
        '- Strong correlations between some operational variables are useful for exploration but do not establish causal relationships.',
        '',
        '## Forecast model comparison',
        '',
        '| Target | Best model by holdout MAE | MAE | RMSE | R² |',
        '|---|---|---:|---:|---:|',
    ]
    for target, result in best_models.items():
        lines.append(f"| `{target}` | {result['model']} | {result['mae']:.2f} | {result['rmse']:.2f} | {result['r2']:.3f} |")
    lines += [
        '',
        'The naïve persistence model is a strong benchmark on the current chronological holdout. More complex models should only be selected if they consistently improve performance under appropriate time-aware validation.',
        '',
        '## Recommendations',
        '',
        '1. Obtain a newer source-data snapshot before presenting forecasts as current.',
        '2. Confirm the operational meaning and reporting cadence of missing calendar dates with the data provider.',
        '3. Evaluate each required calendar-day horizon separately using rolling-origin validation.',
        '4. Compare all models with simple baselines and report MAE and RMSE alongside R².',
        '5. Use capacity-specific thresholds only when authoritative capacity data are available.',
        '6. Re-run the complete pipeline whenever the source dataset is updated.',
        '',
        '## Limitations and responsible interpretation',
        '',
    ]
    lines.extend(f'- {item}' for item in insights['limitations'])
    lines += ['', '## Reproducibility', '', 'Run `python -m src.automated_insights` from the project root to regenerate this report and its machine-readable summary.', '']

    markdown_path = REPORTS / 'automated_insights_report.md'
    json_path = PROCESSED / 'automated_insights_summary.json'
    markdown_path.write_text('\n'.join(lines), encoding='utf-8')
    json_path.write_text(json.dumps(insights, indent=2), encoding='utf-8')
    print(f'Report written: {markdown_path}')
    print(f'Summary written: {json_path}')
    print('Automated insights generated successfully.')
    return insights


if __name__ == '__main__':
    generate_insights()
