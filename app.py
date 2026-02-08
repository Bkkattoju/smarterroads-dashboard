import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

TZINFO = timezone(timedelta(hours=-5))  # ET (fixed offset, avoids cloud tz bugs)

EVENTS_CSV = "event_filtered_clean.csv"
WEATHER_WIDE_CSV = "Weather_sensor_observations_wide.csv"

REFRESH_SECONDS = 60

METRIC_LABELS = {
    "weather-data__air-temperature": "Air Temperature",
    "weather-data__avg-wind-speed": "Avg Wind Speed",
    "weather-data__relative-humidity": "Relative Humidity",
    "weather-data__visibility-data": "Visibility",
    "surface-data__surface-temperature": "Surface Temperature",
}

st.set_page_config(layout="wide", page_title="SmarterRoads Dashboard")

def et_now():
    return datetime.now(timezone.utc).astimezone(TZINFO)

def safe_read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def parse_datetime(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = df[c].dt.tz_convert(TZINFO)
    return df

def latest_weather_snapshot(df, minutes=30):
    if df.empty:
        return pd.DataFrame()
    cutoff = et_now() - timedelta(minutes=minutes)
    df = df[df["obs_iso8601"] >= cutoff]
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("obs_iso8601", ascending=False)
    return df.drop_duplicates("station_device_id", keep="first")

def network_trend(df, metric):
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["val"] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["val"])
    if df.empty:
        return pd.DataFrame()
    df["bucket"] = df["obs_iso8601"].dt.floor("30min")
    return df.groupby("bucket")["val"].mean().reset_index()

def events_trend(df):
    if df.empty or "update_time" not in df.columns:
        return pd.DataFrame()
    df["bucket"] = df["update_time"].dt.floor("30min")
    return df.groupby("bucket").size().reset_index(name="count")

st.title("SmarterRoads Real-Time Monitoring Dashboard")

st.caption(f"Auto-refresh every {REFRESH_SECONDS}s")
st_autorefresh = st.experimental_rerun if False else None

events_df = safe_read_csv(EVENTS_CSV)
weather_df = safe_read_csv(WEATHER_WIDE_CSV)

events_df = parse_datetime(events_df, ["update_time", "start_time", "clear_time"])
weather_df = parse_datetime(weather_df, ["obs_iso8601"])

tabs = st.tabs(["Overview", "Weather", "Events"])

with tabs[0]:
    snap = latest_weather_snapshot(weather_df)
    k1, k2, k3, k4 = st.columns(4)

    total_events = len(events_df)
    active_events = len(events_df[events_df["status"].str.contains("active", na=False)])

    avg_temp = (
        pd.to_numeric(snap["weather-data__air-temperature"], errors="coerce").mean()
        if not snap.empty and "weather-data__air-temperature" in snap.columns
        else None
    )

    avg_vis = (
        pd.to_numeric(snap["weather-data__visibility-data"], errors="coerce").mean()
        if not snap.empty and "weather-data__visibility-data" in snap.columns
        else None
    )

    k1.metric("Total Incidents", total_events)
    k2.metric("Active Incidents", active_events)
    k3.metric("Avg Air Temp (raw)", "—" if avg_temp is None else f"{avg_temp:.2f}")
    k4.metric("Avg Visibility (raw)", "—" if avg_vis is None else f"{avg_vis:.2f}")

with tabs[1]:
    st.subheader("Weather Monitoring")

    metric_options = [
        m for m in METRIC_LABELS if m in weather_df.columns
    ]
    metric_labels = {METRIC_LABELS[m]: m for m in metric_options}

    metric_label = st.selectbox("Metric", list(metric_labels.keys()))
    metric = metric_labels[metric_label]

    snap = latest_weather_snapshot(weather_df)
    map_df = snap.dropna(subset=["lat", "lon", metric]).copy()
    map_df["val"] = pd.to_numeric(map_df[metric], errors="coerce")
    map_df = map_df.dropna(subset=["val"])

    if not map_df.empty:
        fig_map = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            color="val",
            hover_name="station_device_name",
            color_continuous_scale="Turbo",
            zoom=5,
        )
        fig_map.update_layout(mapbox_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True, key="weather_map")

    trend = network_trend(weather_df, metric)
    if not trend.empty:
        fig_trend = px.line(
            trend,
            x="bucket",
            y="val",
            title=f"Network Avg {metric_label} (last 24h)",
        )
        st.plotly_chart(fig_trend, use_container_width=True, key="weather_trend")

with tabs[2]:
    st.subheader("Events Monitoring")

    trend = events_trend(events_df)
    if not trend.empty:
        fig_evt = px.line(
            trend,
            x="bucket",
            y="count",
            title="Incidents per 30 minutes (last 24h)",
        )
        st.plotly_chart(fig_evt, use_container_width=True, key="events_trend")

st.caption(f"Last refresh: {et_now().strftime('%Y-%m-%d %H:%M:%S ET')}")

