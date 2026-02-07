import os
import sys
import subprocess
from datetime import datetime

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="SmarterRoads Dashboard", layout="wide")
st_autorefresh(interval=60_000, key="refresh")

st.title("SmarterRoads Dashboard (VDOT Live Data)")
st.write(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

EVENTS_SCRIPT = "eventfiltered_tmdd_ingest.py"
WEATHER_SCRIPT = "weather_long_ingest.py"

EVENTS_CSV = "event_filtered_clean.csv"
WEATHER_LONG_CSV = "Weather_sensor_observations_long.csv"
WEATHER_WIDE_CSV = "Weather_sensor_observations_wide.csv"

st.sidebar.header("Controls")

run_now = st.sidebar.button("Run ingestion now")
if run_now:
    with st.spinner("Running ingestion scripts..."):
        try:
            subprocess.run([sys.executable, EVENTS_SCRIPT], check=False)
        except Exception as e:
            st.sidebar.error(f"Failed to run {EVENTS_SCRIPT}: {e}")

        try:
            subprocess.run([sys.executable, WEATHER_SCRIPT], check=False)
        except Exception as e:
            st.sidebar.error(f"Failed to run {WEATHER_SCRIPT}: {e}")

tab_events, tab_weather_long, tab_weather_wide = st.tabs(["Events / Incidents", "Weather (Long)", "Weather (Wide)"])

with tab_events:
    st.subheader("Events / Incidents")
    if os.path.exists(EVENTS_CSV):
        df = pd.read_csv(EVENTS_CSV)
        for c in ["update_time", "start_time", "clear_time"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        if "update_time" in df.columns:
            df = df.sort_values("update_time", ascending=False)
        st.write(f"Rows: {len(df)}")
        st.dataframe(df.head(200), use_container_width=True)
        st.download_button(
            "Download events CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=EVENTS_CSV,
            mime="text/csv",
        )
    else:
        st.warning(f"{EVENTS_CSV} not found. Click 'Run ingestion now'.")

with tab_weather_long:
    st.subheader("Weather Sensors (Long)")
    if os.path.exists(WEATHER_LONG_CSV):
        w = pd.read_csv(WEATHER_LONG_CSV)
        if "obs_iso8601" in w.columns:
            w["obs_iso8601"] = pd.to_datetime(w["obs_iso8601"], errors="coerce", utc=True)
            w = w.sort_values("obs_iso8601", ascending=False)
        st.write(f"Rows: {len(w)}")
        st.dataframe(w.head(200), use_container_width=True)
        st.download_button(
            "Download weather long CSV",
            data=w.to_csv(index=False).encode("utf-8"),
            file_name=WEATHER_LONG_CSV,
            mime="text/csv",
        )
    else:
        st.warning(f"{WEATHER_LONG_CSV} not found. Click 'Run ingestion now'.")

with tab_weather_wide:
    st.subheader("Weather Sensors (Wide)")
    if os.path.exists(WEATHER_WIDE_CSV):
        ww = pd.read_csv(WEATHER_WIDE_CSV)
        if "obs_iso8601" in ww.columns:
            ww["obs_iso8601"] = pd.to_datetime(ww["obs_iso8601"], errors="coerce", utc=True)
            ww = ww.sort_values("obs_iso8601", ascending=False)
        st.write(f"Rows: {len(ww)}")
        st.dataframe(ww.head(200), use_container_width=True)
        st.download_button(
            "Download weather wide CSV",
            data=ww.to_csv(index=False).encode("utf-8"),
            file_name=WEATHER_WIDE_CSV,
            mime="text/csv",
        )
    else:
        st.warning(f"{WEATHER_WIDE_CSV} not found. Click 'Run ingestion now'.")

