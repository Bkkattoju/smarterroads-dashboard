import os
import sys
import time
import subprocess
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

APP_TZ = "America/New_York"

EVENTS_CSV = "event_filtered_clean.csv"
WEATHER_LONG_CSV = "Weather_sensor_observations_long.csv"
WEATHER_WIDE_CSV = "Weather_sensor_observations_wide.csv"

INGEST_INTERVAL_SEC = 60
UI_REFRESH_MS = 60_000

EVENTS_SCRIPT = "eventfiltered_tmdd_ingest.py"
WEATHER_SCRIPT = "weather_long_ingest.py"

st.set_page_config(page_title="SmarterRoads Real-Time Monitoring Dashboard", layout="wide")
st_autorefresh(interval=UI_REFRESH_MS, key="ui_refresh")


def get_tzinfo():
    if ZoneInfo is None:
        return None
    try:
        return ZoneInfo(APP_TZ)
    except Exception:
        return None


TZINFO = get_tzinfo()


def now_et_str() -> str:
    if TZINFO is None:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(TZINFO).strftime("%Y-%m-%d %H:%M:%S ET")


def read_secret(name: str) -> Optional[str]:
    v = None
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            v = str(st.secrets[name]).strip()
    except Exception:
        v = None
    if not v:
        v = os.getenv(name)
        if v:
            v = v.strip()
    return v if v else None


def run_ingestion_if_due() -> Tuple[List[str], List[str]]:
    if "last_ingest_ts" not in st.session_state:
        st.session_state["last_ingest_ts"] = 0.0

    now_ts = time.time()
    if now_ts - float(st.session_state["last_ingest_ts"]) < INGEST_INTERVAL_SEC:
        return ([], [])

    missing = []
    if not read_secret("ITERIS_TOKEN_EVENTFILTERED") and not read_secret("ITERIS_TOKEN_EVENTS"):
        missing.append("ITERIS_TOKEN_EVENTFILTERED (or ITERIS_TOKEN_EVENTS)")
    if not read_secret("ITERIS_TOKEN_WEATHER"):
        missing.append("ITERIS_TOKEN_WEATHER")

    errors = []
    if missing:
        return (missing, errors)

    st.session_state["last_ingest_ts"] = now_ts

    try:
        if os.path.exists(EVENTS_SCRIPT):
            subprocess.run([sys.executable, EVENTS_SCRIPT], check=False)
        else:
            errors.append(f"Missing ingestion script: {EVENTS_SCRIPT}")
    except Exception as e:
        errors.append(f"Events ingest failed: {e}")

    try:
        if os.path.exists(WEATHER_SCRIPT):
            subprocess.run([sys.executable, WEATHER_SCRIPT], check=False)
        else:
            errors.append(f"Missing ingestion script: {WEATHER_SCRIPT}")
    except Exception as e:
        errors.append(f"Weather ingest failed: {e}")

    return (missing, errors)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, engine="python")
        except Exception:
            return pd.DataFrame()


def parse_dt_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def to_et(ts_utc: pd.Series) -> pd.Series:
    if TZINFO is None:
        return ts_utc
    try:
        return ts_utc.dt.tz_convert(TZINFO)
    except Exception:
        return ts_utc


def clean_category(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    s2 = s2.replace({"": None, "None": None, "nan": None, "NaN": None})
    return s2


def safe_plotly(fig, key: str, height: int = 360):
    st.plotly_chart(fig, key=key, width="stretch", height=height)


def latest_file_mtime_et(paths: List[str]) -> Optional[str]:
    mt = None
    for p in paths:
        if os.path.exists(p):
            t = os.path.getmtime(p)
            mt = t if mt is None else max(mt, t)
    if mt is None:
        return None
    if TZINFO is None:
        return pd.Timestamp.utcfromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S UTC")
    return pd.Timestamp.utcfromtimestamp(mt).tz_localize("UTC").tz_convert(TZINFO).strftime("%Y-%m-%d %H:%M:%S ET")


def prepare_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for c in ["update_time", "start_time", "clear_time"]:
        if c in df.columns:
            df[c] = parse_dt_utc(df[c])

    if "event_type" in df.columns:
        df["event_type"] = clean_category(df["event_type"])
    if "status" in df.columns:
        df["status"] = clean_category(df["status"])
    if "district" in df.columns:
        df["district"] = clean_category(df["district"])
    if "region" in df.columns:
        df["region"] = clean_category(df["region"])

    for c in ["lat", "lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "update_time" in df.columns:
        df = df.sort_values("update_time", ascending=False)

    return df


def prepare_weather_long(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "obs_iso8601" in df.columns:
        df["obs_iso8601"] = parse_dt_utc(df["obs_iso8601"])

    for c in ["lat", "lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["station_device_id", "station_device_name", "metric_full"]:
        if c in df.columns:
            df[c] = clean_category(df[c])

    if "value" in df.columns:
        df["value_num"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["obs_iso8601"])
    return df


def prepare_weather_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "obs_iso8601" in df.columns:
        df["obs_iso8601"] = parse_dt_utc(df["obs_iso8601"])

    for c in ["lat", "lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["station_device_id", "station_device_name"]:
        if c in df.columns:
            df[c] = clean_category(df[c])

    return df


missing, ingest_errors = run_ingestion_if_due()

st.title("SmarterRoads Real-Time Monitoring Dashboard")

if missing:
    st.error("Missing tokens: set secrets " + ", ".join(missing))
if ingest_errors:
    for e in ingest_errors:
        st.error(e)

events_last = latest_file_mtime_et([EVENTS_CSV])
weather_last = latest_file_mtime_et([WEATHER_LONG_CSV, WEATHER_WIDE_CSV])

top_a, top_b, top_c = st.columns([1, 1, 1])
with top_a:
    st.caption("Events last updated (ET)")
    st.subheader(events_last or "—")
with top_b:
    st.caption("Weather last updated (ET)")
    st.subheader(weather_last or "—")
with top_c:
    st.caption("Live update")
    st.subheader("Auto-ingest ~every 60s | UI refresh every 60s")

events_df = prepare_events(load_csv(EVENTS_CSV))
wlong_df = prepare_weather_long(load_csv(WEATHER_LONG_CSV))
wwide_df = prepare_weather_wide(load_csv(WEATHER_WIDE_CSV))

tabs = st.tabs(["Overview", "Weather", "Events"])

with tabs[0]:
    st.subheader("Key Metrics")

    total_incidents = int(len(events_df)) if not events_df.empty else 0
    active_incidents = 0
    if not events_df.empty and "status" in events_df.columns:
        active_incidents = int((events_df["status"].fillna("") == "event active").sum())

    avg_air = None
    avg_vis = None
    if not wwide_df.empty:
        if "weather-data__air-temperature" in wwide_df.columns:
            avg_air = pd.to_numeric(wwide_df["weather-data__air-temperature"], errors="coerce").mean()
        if "weather-data__visibility-data" in wwide_df.columns:
            avg_vis = pd.to_numeric(wwide_df["weather-data__visibility-data"], errors="coerce").mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Incidents", f"{total_incidents}")
    with k2:
        st.metric("Active Incidents", f"{active_incidents}")
    with k3:
        st.metric("Avg Air Temp (raw)", "—" if avg_air is None or pd.isna(avg_air) else f"{avg_air:.2f}")
    with k4:
        st.metric("Avg Visibility (raw)", "—" if avg_vis is None or pd.isna(avg_vis) else f"{avg_vis:.2f}")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Incidents Trend (last 24h)")
        if events_df.empty or "update_time" not in events_df.columns:
            st.info("No events available yet.")
        else:
            t = events_df.copy()
            t["update_time_et"] = to_et(t["update_time"])
            cutoff = (pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(hours=24))
            t = t[t["update_time"] >= cutoff]
            if t.empty:
                st.info("No events in the last 24 hours.")
            else:
                t["hour_et"] = t["update_time_et"].dt.floor("H")
                s = t.groupby("hour_et").size().reset_index(name="count").sort_values("hour_et")
                fig = px.line(s, x="hour_et", y="count", markers=True)
                fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Hour (ET)", yaxis_title="Incidents")
                safe_plotly(fig, key="ov_events_trend", height=360)

    with c2:
        st.subheader("Weather Trend (station metric)")
        if wlong_df.empty:
            st.info("No weather data available yet.")
        else:
            stations = (
                wlong_df[["station_device_id", "station_device_name"]]
                .dropna()
                .drop_duplicates()
            )
            stations["label"] = stations["station_device_name"].fillna(stations["station_device_id"]).astype(str)
            station_labels = stations.sort_values("label")["label"].tolist()

            metrics = sorted([m for m in wlong_df["metric_full"].dropna().unique().tolist() if str(m).strip() != ""])
            default_metric = "weather-data__air-temperature" if "weather-data__air-temperature" in metrics else (metrics[0] if metrics else None)

            colA, colB = st.columns(2)
            with colA:
                station_choice = st.selectbox("Station", station_labels, key="ov_station")
            with colB:
                metric_choice = st.selectbox("Metric", metrics, index=(metrics.index(default_metric) if default_metric in metrics else 0), key="ov_metric")

            station_row = stations[stations["label"] == station_choice].head(1)
            station_id = station_row["station_device_id"].iloc[0] if not station_row.empty else None

            t = wlong_df.copy()
            if station_id is not None:
                t = t[t["station_device_id"] == station_id]
            t = t[t["metric_full"] == metric_choice]
            t = t.dropna(subset=["value_num", "obs_iso8601"])
            if t.empty:
                st.info("No datapoints for that selection yet.")
            else:
                t["obs_et"] = to_et(t["obs_iso8601"])
                cutoff = (pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(hours=24))
                t = t[t["obs_iso8601"] >= cutoff]
                t = t.sort_values("obs_et")
                fig = px.line(t, x="obs_et", y="value_num", markers=False)
                fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Time (ET)", yaxis_title=metric_choice)
                safe_plotly(fig, key="ov_weather_trend", height=360)

    st.divider()

    m1, m2 = st.columns(2)
    with m1:
        st.subheader("Incidents Map")
        if events_df.empty or "lat" not in events_df.columns or "lon" not in events_df.columns:
            st.info("No mappable events yet.")
        else:
            mdf = events_df.dropna(subset=["lat", "lon"]).copy()
            mdf = mdf[(mdf["lat"].between(-90, 90)) & (mdf["lon"].between(-180, 180))]
            if mdf.empty:
                st.info("No valid coordinates available.")
            else:
                fig = px.scatter_mapbox(
                    mdf.head(2000),
                    lat="lat",
                    lon="lon",
                    hover_name="event_type" if "event_type" in mdf.columns else None,
                    hover_data=["status", "district"] if "status" in mdf.columns and "district" in mdf.columns else None,
                    zoom=6,
                    height=420,
                )
                fig.update_layout(mapbox_style="open-street-map", margin=dict(l=10, r=10, t=10, b=10))
                safe_plotly(fig, key="ov_events_map", height=420)

    with m2:
        st.subheader("Weather Stations Map (latest snapshot)")
        if wwide_df.empty or "lat" not in wwide_df.columns or "lon" not in wwide_df.columns:
            st.info("No weather station map yet.")
        else:
            sdf = wwide_df.dropna(subset=["lat", "lon"]).copy()
            sdf = sdf[(sdf["lat"].between(-90, 90)) & (sdf["lon"].between(-180, 180))]
            if "obs_iso8601" in sdf.columns:
                sdf = sdf.sort_values("obs_iso8601", ascending=False)
                sdf = sdf.drop_duplicates(subset=["station_device_id"], keep="first")
            if sdf.empty:
                st.info("No valid station coordinates available.")
            else:
                fig = px.scatter_mapbox(
                    sdf.head(2000),
                    lat="lat",
                    lon="lon",
                    hover_name="station_device_name" if "station_device_name" in sdf.columns else "station_device_id",
                    zoom=6,
                    height=420,
                )
                fig.update_layout(mapbox_style="open-street-map", margin=dict(l=10, r=10, t=10, b=10))
                safe_plotly(fig, key="ov_weather_map", height=420)

with tabs[1]:
    st.subheader("Weather Analytics (professional view)")

    if wlong_df.empty:
        st.info("No weather data yet.")
    else:
        st.caption(f"Displayed in {APP_TZ}. Current time: {now_et_str()}")

        metrics = sorted([m for m in wlong_df["metric_full"].dropna().unique().tolist() if str(m).strip() != ""])
        metric_default = "weather-data__air-temperature" if "weather-data__air-temperature" in metrics else (metrics[0] if metrics else None)

        stations = (
            wlong_df[["station_device_id", "station_device_name"]]
            .dropna()
            .drop_duplicates()
        )
        stations["label"] = stations["station_device_name"].fillna(stations["station_device_id"]).astype(str)
        station_labels = stations.sort_values("label")["label"].tolist()

        r1, r2, r3 = st.columns([2, 2, 1])
        with r1:
            metric_choice = st.selectbox("Metric", metrics, index=(metrics.index(metric_default) if metric_default in metrics else 0), key="w_metric")
        with r2:
            station_choice = st.selectbox("Station", station_labels, key="w_station")
        with r3:
            hours = st.selectbox("Window", [6, 12, 24, 48], index=2, key="w_hours")

        station_row = stations[stations["label"] == station_choice].head(1)
        station_id = station_row["station_device_id"].iloc[0] if not station_row.empty else None

        t = wlong_df.copy()
        if station_id is not None:
            t = t[t["station_device_id"] == station_id]
        t = t[t["metric_full"] == metric_choice]
        t = t.dropna(subset=["value_num", "obs_iso8601"])
        cutoff = (pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(hours=int(hours)))
        t = t[t["obs_iso8601"] >= cutoff]
        t = t.sort_values("obs_iso8601")

        if t.empty:
            st.info("No data in the selected window.")
        else:
            t["obs_et"] = to_et(t["obs_iso8601"])
            fig = px.line(t, x="obs_et", y="value_num", markers=False)
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Time (ET)", yaxis_title=metric_choice)
            safe_plotly(fig, key="w_station_metric_line", height=420)

        st.divider()

        st.subheader("Network Snapshot (latest wide file)")
        if wwide_df.empty:
            st.info("Wide weather snapshot not available.")
        else:
            snapshot = wwide_df.copy()
            if "obs_iso8601" in snapshot.columns:
                snapshot = snapshot.sort_values("obs_iso8601", ascending=False)
                latest_ts = snapshot["obs_iso8601"].iloc[0]
                snapshot = snapshot[snapshot["obs_iso8601"] == latest_ts]

            candidate_metrics = [
                "weather-data__air-temperature",
                "weather-data__relative-humidity",
                "weather-data__avg-wind-speed",
                "weather-data__precipitation-one-hour",
                "weather-data__visibility-data",
                "surface-data__surface-temperature",
            ]
            available = [c for c in candidate_metrics if c in snapshot.columns]

            if not available:
                st.info("No common snapshot metrics available yet.")
            else:
                cA, cB = st.columns(2)
                with cA:
                    snap_metric = st.selectbox("Snapshot metric", available, key="w_snap_metric")
                with cB:
                    topn = st.selectbox("Top N stations", [10, 20, 30, 50], index=1, key="w_topn")

                s = snapshot[["station_device_name", "station_device_id", snap_metric]].copy()
                s["station"] = s["station_device_name"].fillna(s["station_device_id"]).astype(str)
                s["val"] = pd.to_numeric(s[snap_metric], errors="coerce")
                s = s.dropna(subset=["val"])
                s = s.sort_values("val", ascending=False).head(int(topn))

                if s.empty:
                    st.info("No numeric values for that metric at the latest timestamp.")
                else:
                    fig = px.bar(s, x="station", y="val")
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=30, b=80),
                        xaxis_title="Station",
                        yaxis_title=snap_metric,
                    )
                    fig.update_xaxes(tickangle=45)
                    safe_plotly(fig, key="w_snapshot_bar", height=420)

with tabs[2]:
    st.subheader("Events Analytics (professional view)")

    if events_df.empty:
        st.info("No events available yet.")
    else:
        st.caption(f"Displayed in {APP_TZ}. Current time: {now_et_str()}")

        df = events_df.copy()
        if "update_time" in df.columns:
            df["update_time_et"] = to_et(df["update_time"])

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Top Event Types")
            if "event_type" not in df.columns:
                st.info("Missing event_type.")
            else:
                s = df["event_type"].dropna()
                s = s[s.astype(str).str.strip() != ""]
                counts = s.value_counts().head(12).reset_index()
                counts.columns = ["event_type", "count"]
                if counts.empty:
                    st.info("No event_type values.")
                else:
                    fig = px.bar(counts, x="event_type", y="count")
                    fig.update_layout(margin=dict(l=10, r=10, t=30, b=80), xaxis_title="", yaxis_title="Count")
                    fig.update_xaxes(tickangle=30)
                    safe_plotly(fig, key="e_top_types", height=420)

        with c2:
            st.subheader("Status Breakdown")
            if "status" not in df.columns:
                st.info("Missing status.")
            else:
                s = df["status"].dropna()
                s = s[s.astype(str).str.strip() != ""]
                counts = s.value_counts().reset_index()
                counts.columns = ["status", "count"]
                if counts.empty:
                    st.info("No status values.")
                else:
                    fig = px.pie(counts, names="status", values="count", hole=0.4)
                    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                    safe_plotly(fig, key="e_status_pie", height=420)

        st.divider()

        st.subheader("Incidents by District (Active only)")
        if "district" not in df.columns or "status" not in df.columns:
            st.info("Missing district/status fields.")
        else:
            active = df[df["status"] == "event active"].copy()
            s = active["district"].dropna()
            s = s[s.astype(str).str.strip() != ""]
            counts = s.value_counts().reset_index()
            counts.columns = ["district", "count"]
            if counts.empty:
                st.info("No active incidents by district.")
            else:
                fig = px.bar(counts, x="district", y="count")
                fig.update_layout(margin=dict(l=10, r=10, t=30, b=80), xaxis_title="", yaxis_title="Active incidents")
                fig.update_xaxes(tickangle=30)
                safe_plotly(fig, key="e_active_by_district", height=420)

        st.divider()

        st.subheader("Events Timeline (last 48h by update time)")
        if "update_time" not in df.columns:
            st.info("Missing update_time.")
        else:
            cutoff = (pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(hours=48))
            t = df[df["update_time"] >= cutoff].copy()
            if t.empty:
                st.info("No updates in last 48h.")
            else:
                t["update_time_et"] = to_et(t["update_time"])
                t["hour_et"] = t["update_time_et"].dt.floor("H")
                s = t.groupby(["hour_et", "status"]).size().reset_index(name="count")
                fig = px.area(s, x="hour_et", y="count", color="status")
                fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Hour (ET)", yaxis_title="Updates")
                safe_plotly(fig, key="e_timeline_area", height=420)

