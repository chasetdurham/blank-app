# tamarack_streamlit_app.py
"""
Streamlit app: OpenSnow-style ensemble for Tamarack Resort.
Uses Open-Meteo forecast for future, and historical API for past data.
Run with: streamlit run tamarack_streamlit_app.py
Requirements: streamlit, requests, pandas, numpy, matplotlib
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tamarack — OpenSnow-style Forecast", layout="wide")
st.title("Tamarack Resort — OpenSnow-style Forecast (Demo)")

# Default config (Tamarack)
LAT, LON = 44.671, -116.123
BASE_FT = 4900
SUMMIT_FT = 7700

with st.sidebar:
    st.header("Options")
    resort_elev_ft = st.number_input("Resort elevation (ft)", value=6600, min_value=BASE_FT, max_value=SUMMIT_FT)
    models_selected = st.multiselect("Models to include (Open-Meteo)", options=['gfs','icon','ecmwf'], default=['gfs','icon'])
    hours = st.slider("Hours ahead (forecast)", min_value=24, max_value=384, value=168, step=24,
                      help="Forecast uses Open-Meteo — may be limited by model availability.")
    show_history = st.checkbox("Show previous days snow totals", value=True)
    history_days = st.number_input("History days to fetch", min_value=1, max_value=30, value=7)
    run_click = st.button("Run Forecast")

if not run_click:
    st.info("Choose models and options in the sidebar and click **Run Forecast**.")
    st.stop()

def feet_to_m(ft):
    return ft * 0.3048

def mm_to_inches(x):
    return x / 25.4

def slr_from_temp(tc_c):
    if tc_c <= -12: return 22.0
    if tc_c <= -6: return 18.0
    if tc_c <= -2: return 14.0
    if tc_c <= 0: return 10.0
    return 7.0

# ====== Date ranges ======
NOW_UTC = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
FORECAST_END_UTC = NOW_UTC + datetime.timedelta(hours=hours)

if show_history:
    HIST_START_UTC = NOW_UTC - datetime.timedelta(days=history_days)
    HIST_END_UTC = NOW_UTC - datetime.timedelta(hours=1)
else:
    HIST_START_UTC = HIST_END_UTC = None

# ====== Fetch functions =====
def fetch_forecast(model_name, start_dt, end_dt):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,precipitation"
        f"&start_date={start_dt.date().isoformat()}"
        f"&end_date={end_dt.date().isoformat()}"
        f"&timezone=UTC&models={model_name}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    times = pd.to_datetime(j['hourly']['time'])
    temps = j['hourly']['temperature_2m']
    qpf = j['hourly']['precipitation']
    elev = j.get('elevation', feet_to_m(BASE_FT))
    df = pd.DataFrame({'time_utc': times, 't_C': temps, 'qpf_mm': qpf})
    df = df[(df['time_utc'] >= start_dt) & (df['time_utc'] <= end_dt)].reset_index(drop=True)
    return df, elev

def fetch_historical(model_name, start_dt, end_dt):
    # Use the historical-weather endpoint
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,precipitation"
        f"&start_date={start_dt.date().isoformat()}"
        f"&end_date={end_dt.date().isoformat()}"
        f"&timezone=UTC&models={model_name}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    times = pd.to_datetime(j['hourly']['time'])
    temps = j['hourly']['temperature_2m']
    qpf = j['hourly']['precipitation']
    elev = j.get('elevation', feet_to_m(BASE_FT))
    df = pd.DataFrame({'time_utc': times, 't_C': temps, 'qpf_mm': qpf})
    df = df[(df['time_utc'] >= start_dt) & (df['time_utc'] <= end_dt)].reset_index(drop=True)
    return df, elev

# ====== Optionally fetch history =====
if show_history:
    history_frames = []
    for m in models_selected:
        try:
            df_hist, elev_hist = fetch_historical(m, HIST_START_UTC, HIST_END_UTC)
            model_elev_m = elev_hist
            lapse = 0.0065
            temp_adj = -(feet_to_m(resort_elev_ft) - model_elev_m) * lapse
            df_hist['t_C_adj'] = df_hist['t_C'] + temp_adj
            df_hist['qpf_in'] = df_hist['qpf_mm'].apply(mm_to_inches)
            df_hist['slr'] = df_hist['t_C_adj'].apply(slr_from_temp)
            df_hist['snow_in'] = df_hist['qpf_in'] * df_hist['slr']
            df_hist['model'] = m
            history_frames.append(df_hist[['time_utc','t_C_adj','qpf_in','slr','snow_in','model']])
        except Exception as e:
            st.warning(f"History model {m} failed: {e}")

    if history_frames:
        history_df = pd.concat(history_frames, ignore_index=True)
        hist_daily = history_df.groupby(history_df['time_utc'].dt.date).agg({'snow_in':'sum'}).rename(columns={'snow_in':'snow_in'})
        st.subheader(f"Previous {history_days} days snow totals (in)")
        st.table(hist_daily.style.format({"snow_in":"{:.2f}"}))

# ====== Fetch forecast =====
model_frames = []
for m in models_selected:
    try:
        df_fcst, elev_fcst = fetch_forecast(m, NOW_UTC, FORECAST_END_UTC)
        model_elev_m = elev_fcst
        lapse = 0.0065
        temp_adj = -(feet_to_m(resort_elev_ft) - model_elev_m) * lapse
        df_fcst['t_C_adj'] = df_fcst['t_C'] + temp_adj
        df_fcst['qpf_in'] = df_fcst['qpf_mm'].apply(mm_to_inches)
        df_fcst['slr'] = df_fcst['t_C_adj'].apply(slr_from_temp)
        df_fcst['snow_in'] = df_fcst['qpf_in'] * df_fcst['slr']
        df_fcst['model'] = m
        model_frames.append(df_fcst[['time_utc','t_C_adj','qpf_in','slr','snow_in','model']])
    except Exception as e:
        st.warning(f"Forecast model {m} failed: {e}")

if not model_frames:
    st.error("No model data available. Try different models or check network.")
    st.stop()

ensemble_df = pd.concat(model_frames, ignore_index=True)
pivot = ensemble_df.pivot_table(index='time_utc', columns='model', values='snow_in')
out = pd.DataFrame(index=pivot.index)
out['snow_mean'] = pivot.mean(axis=1)
out['snow_median'] = pivot.median(axis=1)
out['snow_std'] = pivot.std(axis=1).fillna(0.0)
out['liq_mean'] = ensemble_df.pivot_table(index='time_utc', columns='model', values='qpf_in').mean(axis=1)

# Display stats, plot, tables as before
st.subheader("Quick stats")
c1, c2, c3 = st.columns(3)
c1.metric("Forecast window (UTC)", f"{NOW_UTC.isoformat()} → {FORECAST_END_UTC.isoformat()}", "")
c2.metric("Models used", ", ".join(models_selected), "")
c3.metric("Resort elev (ft)", f"{resort_elev_ft}", "")

st.subheader("Forecast plot")
fig, ax = plt.subplots(figsize=(12,4))
t = out.index
ax.bar(t, out['snow_mean'], width=0.03, label='Hourly snow (in)', alpha=0.6)
rolling = out['snow_mean'].rolling(window=24, min_periods=1).sum()
ax.plot(t, rolling, linewidth=2, label='24‑hr rolling snow (in)')
ax.fill_between(t, out['snow_mean']-out['snow_std'], out['snow_mean']+out['snow_std'], alpha=0.2, label='Ensemble ±1σ')
ax.set_xlabel('UTC')
ax.set_ylabel('Snow (in)')
ax.legend()
st.pyplot(fig)

st.subheader("Hourly table (first 120 rows)")
display_df = out.reset_index().rename(columns={'index':'time_utc'}).head(120)
st.dataframe(display_df.style.format({"snow_mean":"{:.2f}", "snow_std":"{:.2f}", "liq_mean":"{:.2f}"}))
csv = display_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV (hourly)", data=csv, file_name='tamarack_forecast_hourly.csv', mime='text/csv')

st.subheader("10-day daily summary")
display_df['local_date'] = display_df['time_utc'].dt.date
daily = display_df.groupby('local_date').agg({'liq_mean':'sum','snow_mean':'sum','snow_std':'mean'}).rename(columns={'liq_mean':'liq_in','snow_mean':'snow_in'})
st.table(daily.head(10).style.format({"liq_in":"{:.2f}","snow_in":"{:.2f}","snow_std":"{:.2f}"}))

st.info("Note: Historical data comes from Open‑Meteo archive API; forecast from standard API.")
