"""
Streamlit app: Chase's Pow Outlook for Idaho & nearby resorts.
Uses Open-Meteo forecast for future, and historical API for past data.
Run with: streamlit run idaho_resorts_forecast.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo

# ====== Resort Config ======
# Timezones: Most of Idaho is Mountain (America/Boise), Panhandle is Pacific (America/Los_Angeles).
RESORTS = {
    # Mountain Time (America/Boise)
    "Tamarack (ID)": {
        "lat": 44.671, "lon": -116.123,
        "base_ft": 4900, "mid_ft": 6600, "summit_ft": 7700,
        "tz": ZoneInfo("America/Boise"),
        "colors": {"primary": "#1f77b4", "accent": "#0d3d56", "fill": "#a6cbe3"},
        "logo_url": "https://upload.wikimedia.org/wikipedia/en/2/2a/Tamarack_Resort_logo.png"
    },
    "Sun Valley (ID)": {
        "lat": 43.697, "lon": -114.351,
        "base_ft": 5750, "mid_ft": 7200, "summit_ft": 9150,
        "tz": ZoneInfo("America/Boise"),
        "colors": {"primary": "#e67e22", "accent": "#873600", "fill": "#f5cba7"},
        "logo_url": "https://upload.wikimedia.org/wikipedia/en/3/3a/Sun_Valley_Resort_logo.png"
    },
    "Bogus Basin (ID)": {
        "lat": 43.767, "lon": -116.101,
        "base_ft": 5800, "mid_ft": 7000, "summit_ft": 7600,
        "tz": ZoneInfo("America/Boise"),
        "colors": {"primary": "#27ae60", "accent": "#145a32", "fill": "#abebc6"},
        "logo_url": "https://upload.wikimedia.org/wikipedia/en/7/7a/Bogus_Basin_logo.png"
    },
    "Brundage (ID)": {
        "lat": 45.004, "lon": -116.155,
        "base_ft": 5776, "mid_ft": 7000, "summit_ft": 7640,
        "tz": ZoneInfo("America/Boise"),
        "colors": {"primary": "#3498db", "accent": "#1a5276", "fill": "#aed6f1"},
        "logo_url": "https://upload.wikimedia.org/wikipedia/en/8/8a/Brundage_Mountain_logo.png"
    },
    "Pomerelle (ID)": {
        "lat": 42.314, "lon": -113.563,
        "base_ft": 8100, "mid_ft": 8600, "summit_ft": 9000,
        "tz": ZoneInfo("America/Boise"),
        "colors": {"primary": "#2ecc71", "accent": "#1d8348", "fill": "#d5f5e3"},
        "logo_url": ""
    },
    "Soldier Mountain (ID)": {
        "lat": 43.481, "lon": -114.920,
        "base_ft": 5700, "mid_ft": 6800, "summit_ft": 7100,
        "tz": ZoneInfo("America/Boise"),
        "colors": {"primary": "#7f8c8d", "accent": "#34495e", "fill": "#d6dbdf"},
        "logo_url": ""
    },
    "Kelly Canyon (ID)": {
        "lat": 43.605, "lon": -111.587,
        "base_ft": 5700, "mid_ft": 6100, "summit_ft": 6600,
        "tz": ZoneInfo("America/Boise"),
        "colors": {"primary": "#16a085", "accent": "#0b5345", "fill": "#a2d9ce"},
        "logo_url": ""
    },
    # Pacific Time (America/Los_Angeles) — Idaho Panhandle
    "Schweitzer (ID)": {
        "lat": 48.369, "lon": -116.623,
        "base_ft": 3900, "mid_ft": 5000, "summit_ft": 6400,
        "tz": ZoneInfo("America/Los_Angeles"),
        "colors": {"primary": "#9b59b6", "accent": "#4a235a", "fill": "#d7bde2"},
        "logo_url": "https://upload.wikimedia.org/wikipedia/en/9/9a/Schweitzer_logo.png"
    },
    "Silver Mountain (ID)": {
        "lat": 47.529, "lon": -116.120,
        "base_ft": 4700, "mid_ft": 5500, "summit_ft": 6300,
        "tz": ZoneInfo("America/Los_Angeles"),
        "colors": {"primary": "#95a5a6", "accent": "#2c3e50", "fill": "#ccd1d1"},
        "logo_url": ""
    },
    "Lookout Pass (ID/MT)": {
        "lat": 47.456, "lon": -115.713,
        "base_ft": 4500, "mid_ft": 5000, "summit_ft": 5600,
        "tz": ZoneInfo("America/Los_Angeles"),
        "colors": {"primary": "#e74c3c", "accent": "#922b21", "fill": "#f5b7b1"},
        "logo_url": ""
    },
    # Nearby (Wyoming) requested
    "Grand Targhee (WY)": {
        "lat": 43.789, "lon": -110.957,
        "base_ft": 8000, "mid_ft": 8500, "summit_ft": 9920,
        "tz": ZoneInfo("America/Denver"),
        "colors": {"primary": "#f1c40f", "accent": "#7d6608", "fill": "#f9e79f"},
        "logo_url": "https://upload.wikimedia.org/wikipedia/en/6/6a/Grand_Targhee_logo.png"
    },
    "Jackson Hole (WY)": {
        "lat": 43.587, "lon": -110.827,
        "base_ft": 6311, "mid_ft": 8000, "summit_ft": 10450,
        "tz": ZoneInfo("America/Denver"),
        "colors": {"primary": "#c0392b", "accent": "#641e16", "fill": "#f5b7b1"},
        "logo_url": "https://upload.wikimedia.org/wikipedia/en/5/5a/Jackson_Hole_logo.png"
    },
}

VALID_MODELS = {"gfs_seamless", "icon_seamless", "ecmwf_ifs04"}

# ====== Helpers ======
def feet_to_m(ft): return ft * 0.3048
def mm_to_inches(mm): return mm / 25.4
def slr_from_temp(tc):
    # Simple snow-to-liquid ratio estimation by temperature
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
        try:
            msg = r.json().get("reason", "")
        except Exception:
            msg = ""
        raise requests.HTTPError(f"{e}. {msg}".strip()) from None
    return r.json()

def fetch_forecast(lat, lon, tz, model, start_dt, end_dt, base_ft):
    base = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,precipitation"
        f"&start_date={start_dt.date()}"
        f"&end_date={end_dt.date()}"
        f"&timezone=UTC"
    )
    url = base + (f"&models={model}" if model else "")
    j = _safe_get(url)
    times = pd.to_datetime(j["hourly"]["time"]).tz_localize("UTC").tz_convert(tz)
    temps = j["hourly"]["temperature_2m"]
    qpf = j["hourly"]["precipitation"]
    elev = j.get("elevation", feet_to_m(base_ft))
    df = pd.DataFrame({"time_local": times, "t_C": temps, "qpf_mm": qpf})
    df = df[(df["time_local"] >= start_dt) & (df["time_local"] <= end_dt)].reset_index(drop=True)
    return df, elev

def fetch_historical(lat, lon, tz, start_dt, end_dt, base_ft):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,precipitation"
        f"&start_date={start_dt.date()}"
        f"&end_date={end_dt.date()}"
        f"&timezone=UTC&models=era5"
    )
    j = _safe_get(url)
    times = pd.to_datetime(j["hourly"]["time"]).tz_localize("UTC").tz_convert(tz)
    temps = j["hourly"]["temperature_2m"]
    qpf = j["hourly"]["precipitation"]
    elev = j.get("elevation", feet_to_m(base_ft))
    df = pd.DataFrame({"time_local": times, "t_C": temps, "qpf_mm": qpf})
    df = df[(df["time_local"] >= start_dt) & (df["time_local"] <= end_dt)].reset_index(drop=True)
    return df, elev

# ====== UI ======
st.set_page_config(page_title="Chase's Pow Outlook", layout="wide")

with st.sidebar:
    resort_choice = st.selectbox("Choose resort", sorted(list(RESORTS.keys())))
    resort = RESORTS[resort_choice]

    elev_choice = st.radio("Choose elevation", ["Base", "Mid", "Summit"], index=1)
    resort_elev_ft = (
        resort["base_ft"] if elev_choice == "Base" else
        resort["mid_ft"] if elev_choice == "Mid" else
        resort["summit_ft"]
    )

    models_selected = st.multiselect(
        "Models to include",
        sorted(list(VALID_MODELS)),
        default=["gfs_seamless", "icon_seamless"]
    )
    hours = st.slider("Hours ahead (forecast)", 24, 384, 168, step=24)
    show_history = st.checkbox("Show previous days snow totals", value=True)
    history_days = st.number_input("History days to fetch", 1, 30, 7)
    run_click = st.button("Run Forecast")

if not run_click:
    st.stop()

# Logo + dynamic title
if resort.get("logo_url"):
    st.image(resort["logo_url"], width=200)
st.title(f"Chase's {resort_choice} Pow Outlook")

# ====== Dates (local to resort) ======
NOW_LOCAL = datetime.datetime.now(resort["tz"]).replace(minute=0, second=0, microsecond=0)
FORECAST_END_LOCAL = NOW_LOCAL + datetime.timedelta(hours=hours)
HIST_START_LOCAL = NOW_LOCAL - datetime.timedelta(days=history_days)
HIST_END_LOCAL = NOW_LOCAL - datetime.timedelta(hours=1)

# ====== History ======
if show_history:
    try:
        df_hist, elev_hist = fetch_historical(
            resort["lat"], resort["lon"], resort["tz"],
            HIST_START_LOCAL, HIST_END_LOCAL, resort["base_ft"]
        )
        lapse = 0.0065  # K per meter
        temp_adj = -(feet_to_m(resort_elev_ft) - elev_hist) * lapse
        df_hist["t_C_adj"] = df_hist["t_C"] + temp_adj
        df_hist["qpf_in"] = df_hist["qpf_mm"].apply(mm_to_inches)
        df_hist["slr"] = df_hist["t_C_adj"].apply(slr_from_temp)
        df_hist["snow_in"] = df_hist["qpf_in"] * df_hist["slr"]

        hist_daily = df_hist.groupby(df_hist["time_local"].dt.date).agg({"snow_in": "sum"})
        hist_daily.index = pd.to_datetime(hist_daily.index).strftime("%m/%d/%y")
        hist_daily = hist_daily.rename(columns={"snow_in": "Snow (in.)"})
        hist_daily = hist_daily.reset_index().rename(columns={"index": "Date"})

        st.subheader(f"{resort_choice} — Previous {history_days} Days")
        st.table(
            hist_daily.style
            .format({"Snow (in.)": "{:.2f}"})
        )
    except Exception as e:
        st.warning(f"Historical fetch failed: {e}")

# ====== Forecast ======
model_frames = []
for m in models_selected or [None]:
    try:
        df_fcst, elev_fcst = fetch_forecast(
            resort["lat"], resort["lon"], resort["tz"],
            m, NOW_LOCAL, FORECAST_END_LOCAL, resort["base_ft"]
        )
        lapse = 0.0065  # K per meter
        temp_adj = -(feet_to_m(resort_elev_ft) - elev_fcst) * lapse
        df_fcst["t_C_adj"] = df_fcst["t_C"] + temp_adj
        df_fcst["qpf_in"] = df_fcst["qpf_mm"].apply(mm_to_inches)
        df_fcst["slr"] = df_fcst["t_C_adj"].apply(slr_from_temp)
        df_fcst["snow_in"] = df_fcst["qpf_in"] * df_fcst["slr"]
        df_fcst["model"] = m or "auto"
        model_frames.append(df_fcst[["time_local", "t_C_adj", "qpf_in", "slr", "snow_in", "model"]])
    except Exception as e:
        st.warning(f"Forecast model {m or 'auto'} failed: {e}")

if not model_frames:
    st.error("No model data available. Try shorter forecast window or check network.")
    st.stop()

# ====== Ensemble ======
ensemble_df = pd.concat(model_frames, ignore_index=True)
pivot = ensemble_df.pivot_table(index="time_local", columns="model", values="snow_in")
out = pd.DataFrame(index=pivot.index)
out["snow_mean"] = pivot.mean(axis=1)
out["snow_std"] = pivot.std(axis=1).fillna(0.0)

# ====== Forecast plot (Hourly, local time) ======
colors = resort.get("colors", {"primary": "#1f77b4", "accent": "#0d3d56", "fill": "#a6cbe3"})
st.subheader(f"{resort_choice} — Forecast (Hourly, Local Time)")
fig, ax = plt.subplots(figsize=(12, 4))
t = out.index
ax.bar(t, out["snow_mean"], width=0.03, label="Hourly snow (in)", color=colors["primary"], alpha=0.7)
ax.plot(t, out["snow_mean"].rolling(24, min_periods=1).sum(), linewidth=2, label="24‑hr rolling snow (in)", color=colors["accent"])
ax.fill_between(t, out["snow_mean"] - out["snow_std"], out["snow_mean"] + out["snow_std"],
                alpha=0.25, color=colors["fill"], label="Ensemble ±1σ")
ax.set_xlabel("Local time")
ax.set_ylabel("Snow (in)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()
fig.set_facecolor("white")
ax.set_facecolor("#fbfdff")
st.pyplot(fig)

# ====== Daily totals summary (local dates) ======
st.subheader("Daily Totals Summary")
out_daily = out.copy()
out_daily["local_date"] = out_daily.index.date
daily_totals = out_daily.groupby("local_date").agg({"snow_mean": "sum"}).rename(columns={"snow_mean": "Snow (in.)"})
daily_totals.index = pd.to_datetime(daily_totals.index).strftime("%m/%d/%y")

# Table with desired labels
daily_table = daily_totals.reset_index().rename(columns={"index": "Date"})
st.table(
    daily_table.style
    .format({"Snow (in.)": "{:.2f}"})
)

# Bar chart (daily snow only) with diagonal x labels
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.bar(daily_totals.index, daily_totals["Snow (in.)"], color=colors["primary"], label="Daily Snow (in.)", alpha=0.9)
ax2.set_xlabel("Date (MM/DD/YY)")
ax2.set_ylabel("Snow (in.)")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.xticks(rotation=30, ha="right")  # diagonal-ish for readability
ax2.legend()
fig2.set_facecolor("white")
ax2.set_facecolor("#fbfdff")
st.pyplot(fig2)
