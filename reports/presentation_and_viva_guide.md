# Project Presentation and Viva Guide

## Project title
**Predictive Forecasting of Care Load & Placement Demand**

## 60–90 second project introduction

My project analyses the U.S. Department of Health and Human Services Unaccompanied Alien Children Program dataset. The goal is to explore historical operational measures and build a reproducible forecasting workflow for HHS care load and daily discharges. The workflow includes data validation, exploratory data analysis, time-aware feature engineering, chronological model evaluation, forecast visualizations, historical-percentile risk context, and a Streamlit dashboard. I compared simple baseline forecasts with machine-learning models rather than assuming that a complex model would perform better. In the current experiments, the persistence baseline achieved the lowest mean absolute error on the held-out period for both targets. The results are exploratory and should not be treated as current operational forecasts because the supplied dataset snapshot ends on 21 December 2025 and contains missing calendar dates.

## Suggested 5-minute demonstration

1. **Problem and dataset (30–45 seconds):** State the forecasting goals and introduce the six source variables.
2. **Data quality (45 seconds):** Show chronological sorting, numeric conversion, and the missing-calendar-date issue. Explain why absent dates were not blindly interpolated.
3. **EDA and feature engineering (45 seconds):** Show trends and explain lagged values, rolling statistics, and calendar features.
4. **Model comparison (60 seconds):** Compare the persistence baseline with Linear Regression, Random Forest, and HistGradientBoosting. Emphasize MAE and chronological evaluation.
5. **Forecasts and risk context (60 seconds):** Show the 7-, 14-, and 30-calendar-day baseline outputs and empirical error bands. Clearly distinguish descriptive percentile flags from actual capacity alerts.
6. **Dashboard and limitations (45 seconds):** Open the Streamlit app, demonstrate filters/tables/downloads, and explain the data freshness and uncertainty limitations.

## Likely evaluator questions and model answers

### 1. What is the problem statement?
The project explores whether historical HHS operational data can support forecasts of the number of children in HHS care and daily discharges, and presents the results in an accessible analytical dashboard.

### 2. Why is this a time-series problem?
The observations are ordered by reporting date, and the aim is to estimate future values from past observations. Evaluation must therefore preserve temporal order.

### 3. Why did you not randomly split the data?
A random split can place later observations in training and earlier observations in testing, leaking future information into evaluation. A chronological split better simulates forecasting unseen future observations.

### 4. What is MAE?
Mean Absolute Error is the average absolute difference between actual and predicted values. It is easy to interpret in the target's units, and lower values indicate smaller average absolute errors.

### 5. Why include a naïve persistence baseline?
It provides a simple benchmark. A complex model is useful only if it improves meaningfully over a transparent baseline on held-out data.

### 6. Why did the baseline perform better?
The observed care-load series is relatively smooth over short intervals, so the latest observed value can be a strong short-term estimate. The result is specific to the supplied dataset and evaluation period; it does not prove that persistence will always be best.

### 7. How did you handle missing dates?
The data-quality workflow identifies gaps and preserves absent calendar dates as missing rather than automatically inventing observations. The reporting cadence must be understood before constructing a complete daily series.

### 8. Are the forecasts current?
No. The supplied dataset snapshot ends on 21 December 2025. The generated forecast dates are based on that snapshot and must not be represented as current operational forecasts.

### 9. What do the risk indicators mean?
They compare forecasts with historical percentiles. They are descriptive context only; they are not official capacity alerts because the dataset does not provide capacity or staffing thresholds.

### 10. Are the uncertainty bands guaranteed prediction intervals?
No. The current bands are empirical error bands based on historical validation errors. Their coverage is not guaranteed and can vary over time and across horizons.

### 11. What are the main limitations?
The data snapshot is not current, calendar dates are missing, a limited set of forecasting models was tested, and the empirical uncertainty bands are not calibrated guarantees. More recent data and horizon-specific rolling-origin evaluation would improve the analysis.

### 12. What would you improve next?
Obtain a current data snapshot, confirm the reporting schedule, define operationally meaningful forecast horizons, run horizon-specific rolling-origin evaluation, calibrate prediction intervals, and obtain approved capacity thresholds if capacity alerts are required.

## Demo-day checklist

- [ ] Install dependencies in a clean virtual environment.
- [ ] Run `streamlit run app.py` and confirm the app opens.
- [ ] Verify both care-load and discharge charts.
- [ ] Check the 7-, 14-, and 30-day forecast selections.
- [ ] Test CSV download controls.
- [ ] Confirm report and chart files are included in the package.
- [ ] Mention the dataset end date and missing-date limitation.
- [ ] Do not describe historical percentile flags as capacity alerts.
- [ ] Do not claim forecast intervals have guaranteed coverage.

## Recommended closing statement

This project demonstrates an end-to-end, reproducible forecasting workflow and shows why a baseline must be evaluated before adopting complex models. The current results are a starting point for decision support, not a substitute for current operational data, validated uncertainty estimates, or official capacity planning information.
