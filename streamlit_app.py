# tamarack_streamlit_app.py
"""
Streamlit app: OpenSnow-style ensemble for Tamarack Resort.
Run with: streamlit run tamarack_streamlit_app.py
Requirements: streamlit, requests, pandas, numpy, matplotlib
Install: pip install streamlit requests pandas numpy matplotlib
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# ------------------ UI ------------------
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
    hours = st.slider("Hours ahead", min_value=48, max_value=240, value=168, step=24)
    show_history = st.checkbox("Show previous days snow totals", value=True)
    run_click = st.button("Run Forecast")

if not run_click:
    st.info("Choose models and options in the sidebar and click **Run Forecast**.")
    st.stop()

# ------------------ Helper functions ------------------
def feet_to_m(ft):
    return ft * 0.3048

def mm_to_inches(x):
    return x / 25.4

def slr_from_temp(tc_c):
    """Simple snow-to-liquid ratio based on temperature (C)"""
    if tc_c <= -12: return 22.0
    if tc_c <= -6: return 18.0
    if tc_c <= -2: return 14.0
    if tc_c <= 0: return 10.0
    return 7.0

# ------------------ Date ranges ------------------
START_UTC = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
END_UTC = START_UTC + datetime.timedelta(hours=hours)

HISTORY_DAYS = 7
history_start = START_UTC - datetime.timedelta(days=HISTORY_DAYS)
history_end = START_UTC - datetime.timedelta(hours=1)

# ------------------ Fetch function ------------------
def fetch_open_meteo(model_name, start_dt, end_dt):
    start_date = start_dt.date().isoformat()
    end_date = end_dt.date().isoformat()
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,precipitation"
        f"&start_date={start_date}&end_date={end_date}"
        f"&timezone=UTC&models={model_name}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    times = pd.to_datetime(j['hourly']['time'])
    temps = j['hourly']['temperature_2m']
    qpf = j['hourly']['precipitation']
    model_elev = j.get('elevation', feet_to_m(BASE_FT))
    df = pd.DataFrame({'time_utc': times, 't_C': temps, 'qpf_mm': qpf})
    df = df[(df['time_utc'] >= start_dt) & (df['time_utc'] <= end_dt)].reset_index(drop=True)
    return df, model_elev

# ------------------ Fetch previous days ------------------
if show_history:
    history_frames = []
    for m in models_selected:
        try:
            df, elev = fetch_open_meteo(m, history_start, history_end)
            model_elev_m = elev
            lapse_deg_per_m = 0.0065
            temp_adj = -(feet_to_m(resort_elev_ft) - model_elev_m) * lapse_deg_per_m
            df['t_C_adj'] = df['t_C'] + temp_adj
            df['qpf_in'] = df['qpf_mm'].apply(mm_to_inches)
            df['slr'] = df['t_C_adj'].apply(slr_from_temp)
            df['snow_in'] = df['qpf_in'] * df['slr']
            df['model'] = m
            history_frames.append(df[['time_utc','t_C_adj','qpf_in','slr','snow_in','model']])
        except Exception as e:
            st.warning(f"History model {m} failed: {e}")
    if history_frames:
        history_df = pd.concat(history_frames, ignore_index=True)
        hist_daily = history_df.groupby(history_df['time_utc'].dt.date).agg({'snow_in':'sum'}).rename(columns={'snow_in':'snow_in'})
        st.subheader(f"Previous {HISTORY_DAYS} days snow totals (in)")
        st.table(hist_daily)

# ------------------ Fetch forecast ------------------
model_frames = []
model_elevs = {}

for m in models_selected:
    try:
        df, elev = fetch_open_meteo(m, START_UTC, END_UTC)
        df = df.set_index('time_utc').reindex(pd.date_range(START_UTC, END_UTC, freq='H')).reset_index().rename(columns={'index':'time_utc'})
        # temp adj (simple lapse-rate)
        model_elevs[m] = elev
        model_elev_m = elev
        lapse_deg_per_m = 0.0065
        temp_adj = -(feet_to_m(resort_elev_ft) - model_elev_m) * lapse_deg_per_m
        df['t_C_adj'] = df['t_C'] + temp_adj
        df['qpf_in'] = df['qpf_mm'].apply(mm_to_inches)
        df['slr'] = df['t_C_adj'].apply(slr_from_temp)
        df['snow_in'] = df['qpf_in'] * df['slr']
        df['model'] = m
        model_frames.append(df[['time_utc','t_C_adj','qpf_in','slr','snow_in','model']])
    except Exception as e:
        st.warning(f"Model {m} failed: {e}")

if not model_frames:
    st.error("No model data available. Try different models or check network.")
    st.stop()

# ------------------ Build ensemble ------------------
ensemble_df = pd.concat(model_frames, ignore_index=True)
pivot_snow = ensemble_df.pivot_table(index='time_utc', columns='model', values='snow_in')
out = pd.DataFrame(index=pivot_snow.index)
out['snow_mean'] = pivot_snow.mean(axis=1)
out['snow_median'] = pivot_snow.median(axis=1)
out['snow_std'] = pivot_snow.std(axis=1).fillna(0.0)
out['liq_mean'] = ensemble_df.pivot_table(index='time_utc',columns='model',values='qpf_in').mean(axis=1)

# ------------------ Quick stats ------------------
st.subheader("Quick stats")
col1, col2, col3 = st.columns(3)
col1.metric("Forecast window (UTC)", f"{START_UTC.isoformat()} → {END_UTC.isoformat()}", "")
col2.metric("Models used", ", ".join(models_selected), "")
col3.metric("Resort elev (ft)", f"{resort_elev_ft}", "")

# ------------------ Plot ------------------
st.subheader("Forecast plot")
fig, ax = plt.subplots(figsize=(12,4))
t = out.index
ax.bar(t, out['snow_mean'], width=0.03, label='Hourly snow (in)', alpha=0.6)
rolling = out['snow_mean'].rolling(window=24, min_periods=1).sum()
ax.plot(t, rolling, linewidth=2, label='24-hr rolling snow (in)')
ax.fill_between(t, out['snow_mean']-out['snow_std'], out['snow_mean']+out['snow_std'], alpha=0.2, label='Ensemble ±1σ')
ax.set_xlabel('UTC')
ax.set_ylabel('Snow (in)')
ax.legend()
st.pyplot(fig)

# ------------------ Data table and downloads ------------------
st.subheader("Hourly table (first 120 rows)")
display_df = out.reset_index().rename(columns={'index':'time_utc'}).head(120)
st.dataframe(display_df.style.format({"snow_mean":"{:.2f}", "snow_std":"{:.2f}", "liq_mean":"{:.2f}"}))
csv = display_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV (hourly)", data=csv, file_name='tamarack_forecast_hourly.csv', mime='text/csv')

# ------------------ Daily summary ------------------
st.subheader("10-day daily summary")
display_df['local_date'] = display_df['time_utc'].dt.date
daily = display_df.groupby('local_date').agg({'liq_mean':'sum','snow_mean':'sum','snow_std':'mean'}).rename(columns={'liq_mean':'liq_in','snow_mean':'snow_in'})
st.table(daily.head(10).style.format({"liq_in":"{:.2f}","snow_in":"{:.2f}","snow_std":"{:.2f}"}))

st.info("This app uses Open-Meteo free model endpoints for demo. For production, add station-based bias correction, vertical-sounding based SLR, and higher-res local models for nowcast skill.")
