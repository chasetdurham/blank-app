"""
Streamlit app: Chase's Pow Outlook for Idaho & nearby resorts.
Uses Open-Meteo forecast for future, and historical API for past data.
Run with: streamlit run idaho_resorts_forecast.py
"""

import streamlit as st
import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo

# ====== Resort config ======
RESORTS = {
    "Tamarack (ID)": {
        "lat": 44.671, "lon": -116.123,
        "base_ft": 4900, "mid_ft": 6600, "summit_ft": 7700,
        "tz": ZoneInfo("America/Boise"),
        "colors": {"primary": "#1f77b4", "accent": "#0d3d56", "fill": "#a6cbe3"},
        # Two YouTube cameras
        "webcam_urls": [
            "https://www.youtube.com/embed/TuXoTi6Y5Uc?si=Y4VNHqFwmGlxIPtc",
            "https://www.youtube.com/embed/3fJt-3O3bec?si=ZeGBlBpRqEFz6jH2"
        ]
    },
    "Brundage (ID)": {
        "lat": 45.004, "lon": -116.155,
        "base_ft": 5776, "mid_ft": 7000, "summit_ft": 7640,
        "tz": ZoneInfo("America/Boise"),
        "colors": {"primary": "#3498db", "accent": "#1a5276", "fill": "#aed6f1"},
        "webcam_url": "https://brundage.com/live-cams/?title=base-area-snow-cam"
    },
    "Bogus Basin (ID)": {
        "lat": 43.767, "lon": -116.101,
        "base_ft": 5800, "mid_ft": 7000, "summit_ft": 7600,
        "tz": ZoneInfo("America/Boise"),
        "colors": {"primary": "#27ae60", "accent": "#145a32", "fill": "#abebc6"},
        # Use the YouTube embed link you provided
    "webcam_url": "https://www.youtube.com/embed/g714uvR8Rjg?si=ranKNhTUPB_jbmYs"
},

    "Sun Valley (ID)": {
        "lat": 43.697, "lon": -114.351,
        "base_ft": 5750, "mid_ft": 7200, "summit_ft": 9150,
        "tz": ZoneInfo("America/Boise"),
        "colors": {"primary": "#e67e22", "accent": "#873600", "fill": "#f5cba7"},
        "webcam_url": "https://www.sunvalley.com/mountain/webcams"
    },
    "Schweitzer (ID)": {
        "lat": 48.369, "lon": -116.623,
        "base_ft": 3900, "mid_ft": 5000, "summit_ft": 6400,
        "tz": ZoneInfo("America/Los_Angeles"),
        "colors": {"primary": "#9b59b6", "accent": "#4a235a", "fill": "#d7bde2"},
       # Use the YouTube embed link you provided
    "webcam_url": "https://www.youtube.com/embed/osT7v3ZS9bI?si=zQ6mfkiuhcY_2Dv-"
},

    "Silver Mountain (ID)": {
        "lat": 47.529, "lon": -116.120,
        "base_ft": 4700, "mid_ft": 5500, "summit_ft": 6300,
        "tz": ZoneInfo("America/Los_Angeles"),
        "colors": {"primary": "#95a5a6", "accent": "#2c3e50", "fill": "#ccd1d1"},
        "webcam_url": "https://silvermt.com/webcams"
    },
    "Lookout Pass (ID/MT)": {
        "lat": 47.456, "lon": -115.713,
        "base_ft": 4500, "mid_ft": 5000, "summit_ft": 5600,
        "tz": ZoneInfo("America/Los_Angeles"),
        "colors": {"primary": "#e74c3c", "accent": "#922b21", "fill": "#f5b7b1"},
        "webcam_url": "https://skilookout.com/webcams"
    },
    "Grand Targhee (WY)": {
        "lat": 43.789, "lon": -110.957,
        "base_ft": 8000, "mid_ft": 8500, "summit_ft": 9920,
        "tz": ZoneInfo("America/Denver"),
        "colors": {"primary": "#f1c40f", "accent": "#7d6608", "fill": "#f9e79f"},
        # Use the YouTube embed link you provided
    "webcam_url": "https://www.youtube.com/embed/B4KTi91qL-4?si=qNu8NHUdGn-M9tOr"
},

    "Jackson Hole (WY)": {
        "lat": 43.587, "lon": -110.827,
        "base_ft": 6311, "mid_ft": 8000, "summit_ft": 10450,
        "tz": ZoneInfo("America/Denver"),
        "colors": {"primary": "#c0392b", "accent": "#641e16", "fill": "#f5b7b1"},
        "webcam_url": "https://www.jacksonhole.com/live-mountain-cams"
    },
"Niseko Grand Hirafu (JP)": {
    "lat": 42.856, "lon": 140.704,   # approximate coordinates
    "base_ft": 1000, "mid_ft": 2500, "summit_ft": 4000,  # adjust if you want more precise
    "tz": ZoneInfo("Asia/Tokyo"),
    "colors": {"primary": "#2ecc71", "accent": "#145a32", "fill": "#abebc6"},
    # YouTube live cam link you provided
    "webcam_url": "https://www.youtube.com/embed/taDZzy8yMTw?si=RM06dEVWWjFaZRDg"
},
    "Kiroro (JP)": {
    "lat": 43.06, "lon": 140.99,   # approximate coordinates for Kiroro Ski Resort
    "base_ft": 1900, "mid_ft": 2600, "summit_ft": 3800,  # adjust if you want more precise
    "tz": ZoneInfo("Asia/Tokyo"),
    "colors": {"primary": "#e74c3c", "accent": "#922b21", "fill": "#f5b7b1"},
    # YouTube live cam link you provided
    "webcam_url": "https://www.youtube.com/embed/YlUSsQruxAM?si=D87jcxjDJTSNqXMo"
},
"Mt. Bachelor (OR)": {
    "lat": 43.983, "lon": -121.688,   # approximate coordinates for Mt. Bachelor Ski Resort
    "base_ft": 5700, "mid_ft": 7000, "summit_ft": 9065,
    "tz": ZoneInfo("America/Los_Angeles"),
    "colors": {"primary": "#2980b9", "accent": "#1a5276", "fill": "#aed6f1"},
    # YouTube live cam link you provided
    "webcam_url": "https://www.youtube.com/embed/jF9f7hsdlJg?si=8Y7OdUnMglvY9_Xx"
},
"Snowbird (UT)": {
    "lat": 40.581, "lon": -111.654,   # approximate coordinates for Snowbird Ski Resort
    "base_ft": 7760, "mid_ft": 9000, "summit_ft": 11000,
    "tz": ZoneInfo("America/Denver"),
    "colors": {"primary": "#16a085", "accent": "#0b5345", "fill": "#a3e4d7"},
    # Official Snowstake webcam page
    "webcam_url": "https://www.snowbird.com/the-mountain/webcams/view-all-webcams/snowstake-webcam/"
},
"Alta (UT)": {
    "lat": 40.588, "lon": -111.638,   # approximate coordinates for Alta Ski Area
    "base_ft": 8530, "mid_ft": 9500, "summit_ft": 10550,
    "tz": ZoneInfo("America/Denver"),
    "colors": {"primary": "#8e44ad", "accent": "#4a235a", "fill": "#d7bde2"},
    # Direct Collins Snow Stake webcam image
    "webcam_url": "https://altaskiarea.s3-us-west-2.amazonaws.com/mountain-cams/Collins_Snow_Stake.jpg"
},


}

DEFAULT_MODELS = ["gfs_seamless", "icon_seamless"]

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
    r.raise_for_status()
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

with st.sidebar ("Filters"):
    resort_keys_sorted = sorted(list(RESORTS.keys()))
    default_index = resort_keys_sorted.index("Tamarack (ID)") if "Tamarack (ID)" in resort_keys_sorted else 0

    resort_choice = st.selectbox("Choose resort", resort_keys_sorted, index=default_index)
    resort = RESORTS[resort_choice]

    elev_choice = st.radio("Choose elevation", ["Base", "Mid", "Summit"], index=1)
    resort_elev_ft = (
        resort["base_ft"] if elev_choice == "Base"
        else resort["mid_ft"] if elev_choice == "Mid"
        else resort["summit_ft"]
    )

    days = st.slider("Days ahead (forecast)", 1, 16, 7)
    show_history = st.checkbox("Show previous days snow totals", value=True)
    history_days = st.number_input("History days to fetch", 1, 30, 7)


# Dynamic title
st.title(f"Chase's {resort_choice} Pow Outlook")

# ====== Dates ======
NOW_LOCAL = datetime.datetime.now(resort["tz"]).replace(minute=0, second=0, microsecond=0)
FORECAST_END_LOCAL = NOW_LOCAL + datetime.timedelta(days=days)
HIST_START_LOCAL = NOW_LOCAL - datetime.timedelta(days=history_days)
HIST_END_LOCAL = NOW_LOCAL - datetime.timedelta(hours=1)

# ====== Forecast (ensemble across default models) ======
model_frames = []
for m in DEFAULT_MODELS or [None]:
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
    st.error("No model data available. Try fewer days or check network.")
    st.stop()

# ====== Ensemble aggregation ======
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
ax.fill_between(t, out["snow_mean"] - out["snow_std"], out["snow_mean"] + out["snow_std"], alpha=0.25, color=colors["fill"], label="Ensemble ±1σ")
ax.set_xlabel("Local time")
ax.set_ylabel("Snow (in)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()
fig.set_facecolor("white")
ax.set_facecolor("#fbfdff")
st.pyplot(fig)

# ====== Daily Forecast Totals Summary (local dates) ======
st.subheader("Daily Forecast Totals")
out_daily = out.copy()
out_daily["Date"] = out_daily.index.date
daily_totals = (
    out_daily.groupby("Date")
             .agg({"snow_mean": "sum"})
             .reset_index()
             .rename(columns={"snow_mean": "Snow (in.)"})
)
daily_totals["Date"] = pd.to_datetime(daily_totals["Date"]).dt.strftime("%m/%d/%y")

# Display table without index column (no Styler to avoid index reappearing)
daily_totals_display = daily_totals[["Date", "Snow (in.)"]].reset_index(drop=True)
daily_totals_display["Snow (in.)"] = daily_totals_display["Snow (in.)"].map("{:.1f}".format)
st.dataframe(daily_totals_display, hide_index=True)

# Bar chart (daily snow only) with diagonal x labels and data labels
fig2, ax2 = plt.subplots(figsize=(10, 4))
bars = ax2.bar(daily_totals["Date"], daily_totals["Snow (in.)"], color=colors["primary"], alpha=0.9, label="Daily Snow (in.)")
ax2.set_xlabel("Date (MM/DD/YY)")
ax2.set_ylabel("Snow (in.)")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.xticks(rotation=30, ha="right")
ax2.legend()

for bar in bars:
    h = bar.get_height()
    ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                 textcoords="offset points", ha="center", va="bottom", fontsize=9, color="black")

fig2.set_facecolor("white")
ax2.set_facecolor("#fbfdff")
st.pyplot(fig2)

# ====== History ======
if show_history:
    try:
        df_hist, elev_hist = fetch_historical(resort["lat"], resort["lon"], resort["tz"],
                                              HIST_START_LOCAL, HIST_END_LOCAL, resort["base_ft"])
        lapse = 0.0065  # K per meter
        temp_adj = -(feet_to_m(resort_elev_ft) - elev_hist) * lapse
        df_hist["t_C_adj"] = df_hist["t_C"] + temp_adj
        df_hist["qpf_in"] = df_hist["qpf_mm"].apply(mm_to_inches)
        df_hist["slr"] = df_hist["t_C_adj"].apply(slr_from_temp)
        df_hist["snow_in"] = df_hist["qpf_in"] * df_hist["slr"]

        hist_daily = (
            df_hist.groupby(df_hist["time_local"].dt.date)
                  .agg({"snow_in": "sum"})
                  .reset_index()
                  .rename(columns={"time_local": "Date", "snow_in": "Snow (in.)"})
        )
        hist_daily["Date"] = pd.to_datetime(hist_daily["Date"]).dt.strftime("%m/%d/%y")

        st.subheader(f"{resort_choice} — Previous {history_days} Days")

        # Display table without index column
        hist_daily_display = hist_daily[["Date", "Snow (in.)"]].reset_index(drop=True)
        hist_daily_display["Snow (in.)"] = hist_daily_display["Snow (in.)"].map("{:.1f}".format)
        st.dataframe(hist_daily_display, hide_index=True)

        # Previous days bar chart
        colors = resort.get("colors", {"primary": "#1f77b4", "accent": "#0d3d56", "fill": "#a6cbe3"})
        fig_prev, ax_prev = plt.subplots(figsize=(10, 3.5))
        bars_prev = ax_prev.bar(hist_daily["Date"], hist_daily["Snow (in.)"], color=colors["primary"], alpha=0.9, label="Daily Snow (in.)")
        ax_prev.set_xlabel("Date (MM/DD/YY)")
        ax_prev.set_ylabel("Snow (in.)")
        plt.xticks(rotation=30, ha="right")
        ax_prev.spines["top"].set_visible(False)
        ax_prev.spines["right"].set_visible(False)
        ax_prev.legend()
        for bar in bars_prev:
            h = bar.get_height()
            ax_prev.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                             textcoords="offset points", ha="center", va="bottom", fontsize=9)
        st.pyplot(fig_prev)

    except Exception as e:
        st.warning(f"Historical fetch failed: {e}")

# ====== Live Cams (at bottom) ======
if "webcam_urls" in resort and resort["webcam_urls"]:
    st.subheader("Live Cams")
    for url in resort["webcam_urls"]:
        st.video(url)
elif "webcam_url" in resort and resort["webcam_url"]:
    st.subheader("Live Cam")
    if "youtube.com" in resort["webcam_url"]:
        st.video(resort["webcam_url"])
    else:
        st.components.v1.iframe(resort["webcam_url"], height=400)
