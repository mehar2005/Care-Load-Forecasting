from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = PROJECT_ROOT / 'data' / 'processed' / 'cleaned_care_load.csv'
FORECAST_PATH = PROJECT_ROOT / 'data' / 'processed' / 'future_forecasts.csv'
FIGURE_DIR = PROJECT_ROOT / 'reports' / 'figures'
REPORT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'forecast_visualization_report.json'
TARGETS = {
    'hhs_care_load': ('HHS Care Load', 'Children in care'),
    'hhs_discharges': ('HHS Discharges', 'Children discharged per report')
}


def load_inputs():
    if not HISTORY_PATH.exists():
        raise FileNotFoundError('Run `python -m src.data_preparation` first.')
    if not FORECAST_PATH.exists():
        raise FileNotFoundError('Run `python -m src.forecast_generation` first.')
    history = pd.read_csv(HISTORY_PATH, parse_dates=['date']).sort_values('date')
    forecasts = pd.read_csv(FORECAST_PATH, parse_dates=['date'])
    required_history = {'date', *TARGETS.keys()}
    required_forecast = {'date', 'target', 'forecast', 'lower_90_empirical', 'upper_90_empirical', 'horizon_days'}
    if not required_history.issubset(history.columns):
        raise ValueError(f'History is missing columns: {sorted(required_history - set(history.columns))}')
    if not required_forecast.issubset(forecasts.columns):
        raise ValueError(f'Forecasts are missing columns: {sorted(required_forecast - set(forecasts.columns))}')
    return history, forecasts


def plot_target(history, forecasts, target, title, ylabel):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(history['date'], history[target], label='Historical observations', linewidth=1.7)
    # Show one non-duplicated forecast trajectory; the 7/14/30-day outputs share the same baseline.
    f = forecasts[(forecasts['target'] == target) & (forecasts['horizon_days'] == 30)].sort_values('date')
    ax.plot(f['date'], f['forecast'], label='30-calendar-day baseline forecast', linewidth=2, linestyle='--')
    if f[['lower_90_empirical', 'upper_90_empirical']].notna().all(axis=None):
        ax.fill_between(f['date'], f['lower_90_empirical'], f['upper_90_empirical'], alpha=0.18, label='Empirical error band (not horizon-calibrated)')
    ax.axvline(history['date'].max(), linestyle=':', linewidth=1.4, label='Forecast origin')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')
    fig.tight_layout()
    path = FIGURE_DIR / f'08_{target}_historical_forecast.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_horizon_comparison(forecasts):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (target, (title, ylabel)) in zip(axes, TARGETS.items()):
        for horizon in (7, 14, 30):
            f = forecasts[(forecasts['target'] == target) & (forecasts['horizon_days'] == horizon)].sort_values('date')
            # Keep each horizon visually distinct by its actual endpoint.
            ax.plot(f['forecast_step_day'], f['forecast'], label=f'{horizon}-day horizon', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Day ahead')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle('Forecast horizon comparison (baseline forecasts)')
    fig.tight_layout()
    path = FIGURE_DIR / '10_forecast_horizon_comparison.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_forecast_table(forecasts):
    rows = []
    for target, (title, _) in TARGETS.items():
        for horizon in (7, 14, 30):
            f = forecasts[(forecasts['target'] == target) & (forecasts['horizon_days'] == horizon)].sort_values('date')
            if f.empty:
                continue
            rows.append({'Target': title, 'Horizon': f'{horizon} days', 'Start': f['date'].min().strftime('%Y-%m-%d'), 'End': f['date'].max().strftime('%Y-%m-%d'), 'Baseline forecast': f'{f["forecast"].iloc[0]:,.1f}', 'Lower empirical band': f'{f["lower_90_empirical"].iloc[0]:,.1f}', 'Upper empirical band': f'{f["upper_90_empirical"].iloc[0]:,.1f}'})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis('off')
    ax.table(cellText=table.values, colLabels=table.columns, loc='center', cellLoc='center')
    ax.set_title('Forecast summary by target and horizon', pad=20)
    fig.tight_layout()
    path = FIGURE_DIR / '09_forecast_summary_table.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    csv_path = PROJECT_ROOT / 'data' / 'processed' / 'forecast_horizon_summary.csv'
    table.to_csv(csv_path, index=False)
    return path, csv_path, len(table)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    history, forecasts = load_inputs()
    generated = []
    for target, (title, ylabel) in TARGETS.items():
        generated.append(plot_target(history, forecasts, target, f'{title}: historical observations and forecast', ylabel))
    table_png, table_csv, rows = plot_forecast_table(forecasts)
    generated.extend([table_png, plot_horizon_comparison(forecasts)])
    report = {
        'history_start_date': history['date'].min().strftime('%Y-%m-%d'),
        'history_end_date': history['date'].max().strftime('%Y-%m-%d'),
        'historical_observations': int(len(history)),
        'forecast_rows': int(len(forecasts)),
        'horizons_calendar_days': sorted(int(x) for x in forecasts['horizon_days'].unique()),
        'summary_rows': rows,
        'figures': [str(p.relative_to(PROJECT_ROOT)) for p in generated],
        'caveats': [
            'The forecast origin is the final date in the supplied dataset, not the current date.',
            'Forecasts are simple baselines, not operationally validated predictions.',
            'Empirical bands are based on next-observed-report errors and are not calibrated by calendar-day horizon.',
            'The source data has missing calendar dates; no missing historical values were invented.'
        ]
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('FORECAST VISUALIZATION COMPLETED')
    print(f'History: {report["history_start_date"]} to {report["history_end_date"]} ({len(history)} observations)')
    print(f'Forecast rows visualized: {len(forecasts)}')
    print(f'Figures generated: {len(generated)}')
    for p in generated:
        print(f' - {p.relative_to(PROJECT_ROOT)}')
    print(f'Summary CSV: {table_csv.relative_to(PROJECT_ROOT)}')
    print(f'Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}')

if __name__ == '__main__':
    main()
