import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

TZINFO = timezone(timedelta(hours=-5))

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
        df = pd.read_csv(path)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def parse_datetime(df, cols):
    for c in cols:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = dt.dt.tz_convert(TZINFO)
    return df


def find_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def latest_weather_snapshot(df, minutes=30):
    if df.empty or "obs_iso8601" not in df.columns:
        return pd.DataFrame()
    cutoff = et_now() - timedelta(minutes=minutes)
    df2 = df[df["obs_iso8601"] >= cutoff].copy()
    if df2.empty:
        return pd.DataFrame()
    df2 = df2.sort_values("obs_iso8601", ascending=False)
    id_col = "station_device_id" if "station_device_id" in df2.columns else None
    if not id_col:
        return df2.head(500)
    return df2.drop_duplicates(id_col, keep="first")


def network_trend(df, metric):
    if df.empty or "obs_iso8601" not in df.columns or metric not in df.columns:
        return pd.DataFrame()
    df2 = df.copy()
    df2["val"] = pd.to_numeric(df2[metric], errors="coerce")
    df2 = df2.dropna(subset=["val", "obs_iso8601"])
    if df2.empty:
        return pd.DataFrame()
    cutoff = et_now() - timedelta(hours=24)
    df2 = df2[df2["obs_iso8601"] >= cutoff]
    if df2.empty:
        return pd.DataFrame()
    df2["bucket"] = df2["obs_iso8601"].dt.floor("30min")
    return df2.groupby("bucket")["val"].mean().reset_index()


def events_trend(df, time_col):
    if df.empty or time_col is None or time_col not in df.columns:
        return pd.DataFrame()
    df2 = df.dropna(subset=[time_col]).copy()
    cutoff = et_now() - timedelta(hours=24)
    df2 = df2[df2[time_col] >= cutoff]
    if df2.empty:
        return pd.DataFrame()
    df2["bucket"] = df2[time_col].dt.floor("30min")
    return df2.groupby("bucket").size().reset_index(name="count")


st.title("SmarterRoads Real-Time Monitoring Dashboard")
st.caption(f"Auto-refresh every {REFRESH_SECONDS}s")

events_df = safe_read_csv(EVENTS_CSV)
weather_df = safe_read_csv(WEATHER_WIDE_CSV)

events_time_col = find_first_existing(events_df, ["update_time", "updateTime", "start_time", "startTime"])
status_col = find_first_existing(events_df, ["status", "event_status", "roadStatus", "eventStatus"])

events_df = parse_datetime(events_df, ["update_time", "updateTime", "start_time", "startTime", "clear_time", "clearTime"])
weather_df = parse_datetime(weather_df, ["obs_iso8601"])

tabs = st.tabs(["Overview", "Weather", "Events"])

with tabs[0]:
    snap = latest_weather_snapshot(weather_df)

    total_events = int(len(events_df)) if not events_df.empty else 0

    active_events = None
    if not events_df.empty and status_col:
        s = events_df[status_col].astype(str).str.lower().str.strip()
        active_events = int(s.str.contains("active", na=False).sum())

    avg_temp = None
    if not snap.empty and "weather-data__air-temperature" in snap.columns:
        avg_temp = pd.to_numeric(snap["weather-data__air-temperature"], errors="coerce").mean()

    avg_vis = None
    if not snap.empty and "weather-data__visibility-data" in snap.columns:
        avg_vis = pd.to_numeric(snap["weather-data__visibility-data"], errors="coerce").mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Incidents", total_events)
    k2.metric("Active Incidents", "—" if active_events is None else active_events)
    k3.metric("Avg Air Temp (raw)", "—" if avg_temp is None or pd.isna(avg_temp) else f"{avg_temp:.2f}")
    k4.metric("Avg Visibility (raw)", "—" if avg_vis is None or pd.isna(avg_vis) else f"{avg_vis:.2f}")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Incidents Trend (last 24h)")
        trend = events_trend(events_df, events_time_col)
        if trend.empty:
            st.info("No events trend data available yet.")
        else:
            fig = px.line(trend, x="bucket", y="count", markers=True)
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Time (ET)", yaxis_title="Count")
            st.plotly_chart(fig, width="stretch", key="ov_events_trend")

    with c2:
        st.subheader("Weather Trend (Network Avg)")
        metric = "weather-data__air-temperature" if "weather-data__air-temperature" in weather_df.columns else None
        if metric is None:
            st.info("Weather trend metric not available yet.")
        else:
            wtrend = network_trend(weather_df, metric)
            if wtrend.empty:
                st.info("No weather trend data available yet.")
            else:
                fig = px.line(wtrend, x="bucket", y="val", markers=True)
                fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Time (ET)", yaxis_title="Avg Air Temp")
                st.plotly_chart(fig, width="stretch", key="ov_weather_trend")

with tabs[1]:
    st.subheader("Weather Monitoring")

    metric_options = [m for m in METRIC_LABELS if m in weather_df.columns]
    if not metric_options:
        st.info("No weather metrics available yet.")
    else:
        metric_label_to_key = {METRIC_LABELS[m]: m for m in metric_options}
        metric_label = st.selectbox("Metric", list(metric_label_to_key.keys()), key="w_metric")
        metric = metric_label_to_key[metric_label]

        snap = latest_weather_snapshot(weather_df)
        lat_col = find_first_existing(snap, ["lat", "latitude"])
        lon_col = find_first_existing(snap, ["lon", "longitude"])
        name_col = find_first_existing(snap, ["station_device_name", "station_device_id"])

        if snap.empty or lat_col is None or lon_col is None:
            st.info("No station coordinates available yet.")
        else:
            map_df = snap.dropna(subset=[lat_col, lon_col]).copy()
            map_df["val"] = pd.to_numeric(map_df.get(metric), errors="coerce")
            map_df = map_df.dropna(subset=["val"])

            if map_df.empty:
                st.info("No mappable numeric values yet for this metric.")
            else:
                fig = px.scatter_mapbox(
                    map_df,
                    lat=lat_col,
                    lon=lon_col,
                    color="val",
                    hover_name=name_col,
                    zoom=5,
                    height=520,
                    color_continuous_scale="Turbo",
                )
                fig.update_layout(mapbox_style="carto-positron", margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch", key="weather_map")

        wtrend = network_trend(weather_df, metric)
        if wtrend.empty:
            st.info("No trend data yet (need more time-series samples).")
        else:
            fig = px.line(wtrend, x="bucket", y="val", markers=True, title=f"Network Avg {metric_label} (last 24h)")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Time (ET)", yaxis_title=metric_label)
            st.plotly_chart(fig, width="stretch", key="weather_trend")

with tabs[2]:
    st.subheader("Events Monitoring")

    trend = events_trend(events_df, events_time_col)
    if trend.empty:
        st.info("No events data available yet.")
    else:
        fig = px.line(trend, x="bucket", y="count", markers=True)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Time (ET)", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch", key="events_trend")

    if not events_df.empty:
        type_col = find_first_existing(events_df, ["event_type", "type", "category"])
        if type_col:
            s = events_df[type_col].astype(str).str.strip()
            s = s[(s != "") & (s.str.lower() != "nan") & (s.str.lower() != "none")]
            top = s.value_counts().head(10).reset_index()
            top.columns = ["event_type", "count"]
            if not top.empty:
                fig = px.bar(top, x="event_type", y="count", title="Top Event Types")
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=80), xaxis_title="", yaxis_title="Count")
                fig.update_xaxes(tickangle=30)
                st.plotly_chart(fig, width="stretch", key="events_top_types")

st.caption(f"Last refresh: {et_now().strftime('%Y-%m-%d %H:%M:%S ET')}")
time.sleep(REFRESH_SECONDS)
st.experimental_rerun()

