# Predictive Forecasting of Care Load & Placement Demand

A Python and Streamlit data-science project using the HHS Unaccompanied Alien Children Program dataset. It includes data preparation, exploratory data analysis, feature engineering, baseline forecasting, chronological validation, forecast visualization, historical-percentile risk context, model comparison, and an interactive dashboard.

> **Important:** The included source snapshot ends on **21 December 2025**. Its forecasts are historical baseline projections from that date, not current operational forecasts. Missing calendar dates are retained as missing rather than automatically imputed. The dataset does not provide facility capacity limits, so percentile-based indicators are not capacity alerts.

## 1. Project structure

```text
care_load_forecasting/
├── app.py
├── requirements.txt
├── DEPLOYMENT.md
├── data/
│   ├── raw/                 # Original CSV; keep unchanged
│   └── processed/           # Cleaned data, features, forecasts, metrics
├── src/                     # Reusable analysis and modelling modules
├── reports/
│   ├── figures/             # Generated charts
│   ├── final_research_report.docx
│   ├── final_research_report.md
│   ├── presentation_and_viva_guide.md
│   └── other QA and analysis reports
└── notebooks/               # Space for exploratory notebooks
```

## 2. Requirements

Python 3.10 or newer is recommended. Install dependencies in a virtual environment.

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run the dashboard

From the project root, run:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal (usually `http://localhost:8501`). The dashboard uses the processed data already included in `data/processed/`.

## 4. Rebuild outputs

Run modules from the project root so Python can resolve the `src` package and relative project paths correctly. The general sequence is:

```bash
python -m src.data_preparation
python -m src.eda
python -m src.feature_engineering
python -m src.forecasting
python -m src.validation
python -m src.forecast_generation
python -m src.forecast_visualization
python -m src.risk_analysis
python -m src.model_explainability
python -m src.automated_insights
```

After rebuilding outputs, launch the dashboard again. Review the generated reports before using results in a submission. The modules are separate stages; rerunning an earlier stage does not necessarily execute every later stage automatically.

## 5. Main deliverables

- `reports/final_research_report.docx` — editable research report.
- `reports/final_research_report.md` — Markdown report.
- `reports/presentation_and_viva_guide.md` — demonstration and viva preparation.
- `reports/figures/` — EDA, forecast, risk-context and model-explainability charts.
- `data/processed/forecast_model_metrics.csv` — holdout model metrics.
- `data/processed/rolling_validation_metrics.csv` — rolling-validation metrics.
- `data/processed/future_forecasts.csv` — baseline forecast rows.
- `data/processed/forecast_risk_indicators.csv` — historical-percentile context indicators.

## 6. Methodology notes and limitations

- Evaluation uses chronological splits; random shuffling is inappropriate for time-ordered forecasting.
- The source contains missing calendar dates. A lag of one observed row does not necessarily mean one calendar day.
- The supplied forecast generator creates calendar-date rows using simple baselines; these should not be interpreted as validated, operational multi-step forecasts.
- Empirical error bands are not guaranteed calibrated prediction intervals, and are not separately calibrated for each forecast horizon.
- Historical percentile comparisons describe where forecasts sit relative to observed history; they do not establish operational risk or facility capacity exceedance.
- Correlation and feature importance are descriptive/predictive, not causal.
- Forecasts should be regenerated and validated against a newer source snapshot before being used for current planning.

## 7. Data source

HHS Unaccompanied Alien Children Program dataset. Check the official catalogue for the latest available source file before updating the analysis: https://catalog.data.gov/dataset/hhs-unaccompanied-alien-children-program

## 8. Final local verification

Before submission:

1. Install the requirements in a fresh virtual environment.
2. Run `streamlit run app.py`.
3. Confirm the dashboard loads without errors.
4. Test both target views, all three forecast horizons, charts, and CSV downloads.
5. Confirm the report and figures are present in the extracted folder.
6. Be ready to explain the stale data date, missing dates, baseline model choice, and uncertainty limitations honestly.
