"""
Streamlit app: OpenSnow-style ensemble for Tamarack Resort.
Uses Open-Meteo forecast for future, and historical API for past data.
Run with: streamlit run tamarack_streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# ====== Config ======
LAT, LON = 44.671, -116.123
BASE_FT, SUMMIT_FT = 4900, 7700
VALID_MODELS = {"gfs", "icon", "ecmwf"}

# ====== Helpers ======
def feet_to_m(ft): return ft * 0.3048
def mm_to_inches(mm): return mm / 25.4
def slr_from_temp(tc):
    if tc <= -12: return 22.0
    if tc <= -6: return 18.0
    if tc <= -2: return 14.0
    if tc <= 0: return 10.0
    return 7.0

def _safe_get(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        try: msg = r.json().get("reason", "")
        except: msg = ""
        raise requests.HTTPError(f"{e}. {msg}".strip()) from None
    return r.json()

def fetch_forecast(model, start_dt, end_dt):
    base = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=temperature_2m,precipitation&start_date={start_dt.date()}&end_date={end_dt.date()}&timezone=UTC"
    url = base + f"&models={model}" if model else base
    j = _safe_get(url)
    times = pd.to_datetime(j["hourly"]["time"])
    temps = j["hourly"]["temperature_2m"]
    qpf = j["hourly"]["precipitation"]
    elev = j.get("elevation", feet_to_m(BASE_FT))
    df = pd.DataFrame({"time_utc": times, "t_C": temps, "qpf_mm": qpf})
    df = df[(df["time_utc"] >= start_dt) & (df["time_utc"] <= end_dt)].reset_index(drop=True)
    return df, elev

def fetch_historical(start_dt, end_dt):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&hourly=temperature_2m,precipitation&start_date={start_dt.date()}&end_date={end_dt.date()}&timezone=UTC&models=era5"
    j = _safe_get(url)
    times = pd.to_datetime(j["hourly"]["time"])
    temps = j["hourly"]["temperature_2m"]
    qpf = j["hourly"]["precipitation"]
    elev = j.get("elevation", feet_to_m(BASE_FT))
    df = pd.DataFrame({"time_utc": times, "t_C": temps, "qpf_mm": qpf})
    df = df[(df["time_utc"] >= start_dt) & (df["time_utc"] <= end_dt)].reset_index(drop=True)
    return df, elev

# ====== UI ======
st.set_page_config(page_title="Tamarack — OpenSnow-style Forecast", layout="wide")
st.title("Tamarack Resort — OpenSnow-style Forecast (Demo)")

with st.sidebar:
    st.header("Options")
    resort_elev_ft = st.number_input("Resort elevation (ft)", value=6600, min_value=BASE_FT, max_value=SUMMIT_FT)
    models_selected = [m for m in st.multiselect("Models to include", list(VALID_MODELS), default=["gfs", "icon"]) if m in VALID_MODELS]
    hours = st.slider("Hours ahead (forecast)", 24, 384, 168, step=24)
    show_history = st.checkbox("Show previous days snow totals", value=True)
    history_days = st.number_input("History days to fetch", 1, 30, 7)
    run_click = st.button("Run Forecast")

if not run_click:
    st.info("Choose models and options in the sidebar and click **Run Forecast**.")
    st.stop()

# ====== Dates ======
NOW_UTC = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
FORECAST_END_UTC = NOW_UTC + datetime.timedelta(hours=hours)
HIST_START_UTC = NOW_UTC - datetime.timedelta(days=history_days)
HIST_END_UTC = NOW_UTC - datetime.timedelta(hours=1)

# ====== History ======
if show_history:
    try:
        df_hist, elev_hist = fetch_historical(HIST_START_UTC, HIST_END_UTC)
        lapse = 0.0065
        temp_adj = -(feet_to_m(resort_elev_ft) - elev_hist) * lapse
        df_hist["t_C_adj"] = df_hist["t_C"] + temp_adj
        df_hist["qpf_in"] = df_hist["qpf_mm"].apply(mm_to_inches)
        df_hist["slr"] = df_hist["t_C_adj"].apply(slr_from_temp)
        df_hist["snow_in"] = df_hist["qpf_in"] * df_hist["slr"]
        df_hist["model"] = "era5"
        hist_daily = df_hist.groupby(df_hist["time_utc"].dt.date).agg({"snow_in":"sum"})
        st.subheader(f"Previous {history_days} days snow totals (in)")
        st.table(hist_daily.style.format({"snow_in":"{:.2f}"}))
    except Exception as e:
        st.warning(f"Historical fetch failed: {e}")

# ====== Forecast ======
model_frames = []
for m in models_selected or [None]:
    try:
        df_fcst, elev_fcst = fetch_forecast(m, NOW_UTC, FORECAST_END_UTC)
        lapse = 0.0065
        temp_adj = -(feet_to_m(resort_elev_ft) - elev_fcst) * lapse
        df_fcst["t_C_adj"] = df_fcst["t_C"] + temp_adj
        df_fcst["qpf_in"] = df_fcst["qpf_mm"].apply(mm_to_inches)
        df_fcst["slr"] = df_fcst["t_C_adj"].apply(slr_from_temp)
        df_fcst["snow_in"] = df_fcst["qpf_in"] * df_fcst["slr"]
        df_fcst["model"] = m or "auto"
        model_frames.append(df_fcst[["time_utc","t_C_adj","qpf_in","slr","snow_in","model"]])
    except Exception as e:
        st.warning(f"Forecast model {m or 'auto'} failed: {e}")

if not model_frames:
    st.error("No model data available. Try shorter forecast window or check network.")
    st.stop()

# ====== Ensemble ======
ensemble_df = pd.concat(model_frames, ignore_index=True)
pivot = ensemble_df.pivot_table(index="time_utc", columns="model", values="snow_in")
out = pd.DataFrame(index=pivot.index)
out["snow_mean"] = pivot.mean(axis=1)
out["snow_median"] = pivot.median(axis=1)
out["snow_std"] = pivot.std(axis=1).fillna(0.0)
out["liq_mean"] = ensemble_df.pivot_table(index="time_utc", columns="model", values="qpf_in").mean(axis=1)

# ====== Display ======
st.subheader("Quick stats")
c1, c2, c3 = st.columns(3)
c1.metric("Forecast window (UTC)", f"{NOW_UTC.isoformat()} → {FORECAST_END_UTC.isoformat()}", "")
c2.metric("Models used", ", ".join([m or "auto" for m in models_selected]), "")
c3.metric("Resort elev (ft)", f"{resort_elev_ft}", "")

st.subheader("Forecast plot")
fig, ax = plt.subplots(figsize=(12,4))
t = out.index
ax.bar(t, out["snow_mean"], width=0.03, label="Hourly snow (in)", alpha=0.6)
ax.plot(t, out["snow_mean"].rolling(24, min_periods=1).sum(), linewidth=2, label="24‑hr rolling snow (in)")
ax.fill_between(t, out["snow_mean"]-out["snow_std"], out["snow_mean"]+out["snow_std"], alpha=0.2, label="Ensemble ±1σ")
ax.set_xlabel("UTC")
ax.set_ylabel("Snow (in)")
ax.legend()
st.pyplot(fig)

st.subheader("Hourly table (first 120 rows)")
display_df = out.reset_index().rename(columns={"index":"time_utc"}).head(120)
st.dataframe(display_df.style.format({"snow_mean
