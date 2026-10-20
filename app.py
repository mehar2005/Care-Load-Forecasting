from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
HISTORY_PATH = DATA / "cleaned_care_load.csv"
FORECAST_PATH = DATA / "future_forecasts.csv"
RISK_PATH = DATA / "forecast_risk_indicators.csv"
REPORT_PATH = DATA / "forecast_generation_report.json"

st.set_page_config(page_title="Care Load Forecasting", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    history = pd.read_csv(HISTORY_PATH, parse_dates=["date"]).sort_values("date")
    forecasts = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    risks = pd.read_csv(RISK_PATH, parse_dates=["date"]) if RISK_PATH.exists() else pd.DataFrame()
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else {}
    for col in ["hhs_care_load", "hhs_discharges"]:
        history[col] = pd.to_numeric(history[col], errors="coerce")
    forecasts["forecast"] = pd.to_numeric(forecasts["forecast"], errors="coerce")
    for col in ["lower_90_empirical", "upper_90_empirical"]:
        forecasts[col] = pd.to_numeric(forecasts[col], errors="coerce")
    return history, forecasts, risks, report

st.title("Care Load & Placement Demand Forecasting")
st.caption("HHS Unaccompanied Alien Children Program · Historical analysis and baseline forecasts")

if not HISTORY_PATH.exists() or not FORECAST_PATH.exists():
    st.error("Required processed files are missing. Run the preparation and forecasting modules first.")
    st.code("python -m src.data_preparation\npython -m src.forecast_generation\npython -m src.risk_analysis")
    st.stop()

history, all_forecasts, risks, report = load_data()

with st.sidebar:
    st.header("Dashboard controls")
    horizon = st.selectbox("Forecast horizon", [7, 14, 30], index=2, format_func=lambda x: f"{x} calendar days")
    history_days = st.slider("Historical context (days)", min_value=30, max_value=720, value=180, step=30)
    target_choice = st.radio("Primary view", ["HHS care load", "HHS discharges"], index=0)
    st.divider()
    st.caption("The forecast origin is fixed to the last date in the supplied dataset. Update the source data to produce current forecasts.")

forecast = all_forecasts[all_forecasts["horizon_days"] == horizon].copy()
last_date = history["date"].max()
first_date = history["date"].min()
care_latest = history.dropna(subset=["hhs_care_load"]).iloc[-1]
discharge_latest = history.dropna(subset=["hhs_discharges"]).iloc[-1]

if last_date < pd.Timestamp.today().normalize() - pd.Timedelta(days=30):
    st.warning(f"The dataset ends on {last_date:%d %b %Y}. Forecasts shown are historical projections from that date, not current operational forecasts.")

st.info("These are transparent baseline projections: care load uses persistence and discharges use the median of recent observed reports. Error bands are empirical and are not calibrated separately for each calendar-day horizon.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observed records", f"{len(history):,}")
c2.metric("Historical coverage", f"{first_date:%d %b %Y} – {last_date:%d %b %Y}")
c3.metric("Latest HHS care load", f"{care_latest['hhs_care_load']:,.0f}")
c4.metric("Latest discharges", f"{discharge_latest['hhs_discharges']:,.0f}")

st.subheader(f"{horizon}-day forecast overview")
care_fc = forecast[forecast["target"] == "hhs_care_load"].sort_values("date")
disch_fc = forecast[forecast["target"] == "hhs_discharges"].sort_values("date")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Care-load baseline", f"{care_fc['forecast'].iloc[-1]:,.0f}" if not care_fc.empty else "N/A")
k2.metric("Discharge baseline", f"{disch_fc['forecast'].iloc[-1]:,.0f}" if not disch_fc.empty else "N/A")
k3.metric("Forecast starts", f"{forecast['date'].min():%d %b %Y}" if not forecast.empty else "N/A")
k4.metric("Forecast ends", f"{forecast['date'].max():%d %b %Y}" if not forecast.empty else "N/A")


def forecast_chart(target, title, color):
    hist = history[["date", target]].dropna().tail(history_days)
    fut = forecast[forecast["target"] == target].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist[target], mode="lines", name="Observed history", line=dict(width=2)))
    if not fut.empty:
        fig.add_trace(go.Scatter(x=fut["date"], y=fut["upper_90_empirical"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fut["date"], y=fut["lower_90_empirical"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(100, 149, 237, 0.20)", name="Empirical error band"))
        fig.add_trace(go.Scatter(x=fut["date"], y=fut["forecast"], mode="lines", name="Baseline forecast", line=dict(dash="dash", width=3)))
        fig.add_vline(x=last_date, line_dash="dot", annotation_text="Forecast origin")
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Children", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig

left, right = st.columns(2)
with left:
    st.plotly_chart(forecast_chart("hhs_care_load", "HHS care load: history and forecast", "blue"), use_container_width=True)
with right:
    st.plotly_chart(forecast_chart("hhs_discharges", "HHS discharges: history and forecast", "green"), use_container_width=True)

st.subheader("Forecast details")
tab1, tab2, tab3 = st.tabs(["Care load", "Discharges", "Risk indicators"])
with tab1:
    st.dataframe(care_fc[["date", "forecast", "lower_90_empirical", "upper_90_empirical"]].rename(columns={"date":"Date", "forecast":"Forecast", "lower_90_empirical":"Empirical lower band", "upper_90_empirical":"Empirical upper band"}), use_container_width=True, hide_index=True)
    st.download_button("Download care-load forecast CSV", care_fc.to_csv(index=False).encode("utf-8"), file_name=f"care_load_forecast_{horizon}d.csv", mime="text/csv")
with tab2:
    st.dataframe(disch_fc[["date", "forecast", "lower_90_empirical", "upper_90_empirical"]].rename(columns={"date":"Date", "forecast":"Forecast", "lower_90_empirical":"Empirical lower band", "upper_90_empirical":"Empirical upper band"}), use_container_width=True, hide_index=True)
    st.download_button("Download discharge forecast CSV", disch_fc.to_csv(index=False).encode("utf-8"), file_name=f"discharge_forecast_{horizon}d.csv", mime="text/csv")
with tab3:
    if risks.empty:
        st.warning("Risk indicators are not available. Run `python -m src.risk_analysis`.")
    else:
        risk_view = risks[risks["horizon_days"] == horizon].copy()
        if target_choice == "HHS care load":
            risk_view = risk_view[risk_view["target"] == "hhs_care_load"]
        else:
            risk_view = risk_view[risk_view["target"] == "hhs_discharges"]
        st.caption("Historical percentile comparisons are descriptive only; they are not facility capacity or safety alerts.")
        st.dataframe(risk_view, use_container_width=True, hide_index=True)
        st.download_button("Download risk indicators CSV", risk_view.to_csv(index=False).encode("utf-8"), file_name=f"risk_indicators_{horizon}d.csv", mime="text/csv")

with st.expander("Model and data limitations"):
    st.markdown("""
    - The supplied dataset ends on **21 December 2025**; forecasts are projections from that historical date.
    - Calendar dates missing from the source are not imputed.
    - Care load uses a persistence baseline; discharges use the median of the latest 28 observed reports.
    - Error bands use rolling out-of-fold absolute errors for the next observed report, not horizon-specific calibrated intervals.
    - Historical percentile thresholds are not operational capacity limits.
    - Forecasts are for exploratory planning and should not be interpreted as individual-level predictions or operational advice.
    """)

st.caption("Built with Python, Pandas, Plotly and Streamlit.")
