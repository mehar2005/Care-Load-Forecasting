# Section 14 — Quality Assurance and Integration Checks

## Test summary

- Python syntax compilation: **PASS**
- Individual module executions: **10/10 passed**

| Check | Status |
|---|---|
| `data_preparation` | PASS |
| `eda` | PASS |
| `feature_engineering` | PASS |
| `forecasting` | PASS |
| `validation` | PASS |
| `forecast_generation` | PASS |
| `forecast_visualization` | PASS |
| `risk_analysis` | PASS |
| `model_explainability` | PASS |
| `automated_insights` | PASS |
| `compileall` | PASS |

## Required artifact checks

| Artifact | Exists | Size (bytes) |
|---|---:|---:|
| `data/raw/HHS_Unaccompanied_Alien_Children_Program.csv` | Yes | 23742 |
| `data/processed/cleaned_care_load.csv` | Yes | 21710 |
| `data/processed/future_forecasts.csv` | Yes | 25765 |
| `data/processed/forecast_risk_indicators.csv` | Yes | 21666 |
| `reports/final_research_report.docx` | Yes | 810019 |
| `app.py` | Yes | 7937 |

## Scope and limitations

Each listed Python module was executed successfully in this session; compileall also passed. The combined batch execution exceeded the time limit while running model explainability, which then passed when rerun individually. This does not constitute a live browser/UI acceptance test of the Streamlit dashboard or independent validation of forecasting assumptions.

## Run the application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Before submission, open the dashboard locally and inspect the charts and controls. If you update the source dataset, rerun the pipeline in dependency order and review the generated reports.
