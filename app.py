import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Iterable, Tuple

import pandas as pd
import pytz
import requests
import streamlit as st
import xml.etree.ElementTree as ET
import plotly.express as px
from streamlit_autorefresh import st_autorefresh


EVENTS_URL = "https://data.511-atis-ttrip-prod.iteriscloud.com/smarterRoads/eventFiltered/eventFilteredTMDD/current/eventFiltered_tmdd.xml"
WEATHER_URL = "https://data.511-atis-ttrip-prod.iteriscloud.com/smarterRoads/weather/weatherTMDD/current/weather_tmdd.xml"

EVENTS_CSV = "event_filtered_clean.csv"
WEATHER_LONG_CSV = "Weather_sensor_observations_long.csv"
WEATHER_WIDE_CSV = "Weather_sensor_observations_wide.csv"

AUTO_INGEST_SECONDS = 60
AUTO_REFRESH_MS = 60_000
LOCK_FILE = ".ingest.lock"

HEADERS = {"Accept": "application/xml", "User-Agent": "smarterroads-streamlit-dashboard"}

DEFAULT_WEATHER_METRIC = "weather-data__air-temperature"
DEFAULT_VIS_METRIC = "weather-data__visibility-data"
DEFAULT_WIND_METRIC = "weather-data__avg-wind-speed"

EASTERN = pytz.timezone("US/Eastern")

st.set_page_config(page_title="SmarterRoads Real-Time Dashboard", layout="wide")
st_autorefresh(interval=AUTO_REFRESH_MS, key="auto_refresh_main")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def et_now() -> datetime:
    return datetime.now(EASTERN)


def to_eastern(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(EASTERN)


def fmt_et(dt: Optional[datetime]) -> str:
    d = to_eastern(dt)
    return d.strftime("%Y-%m-%d %H:%M:%S %Z") if d else "—"


def file_mtime_utc(path: str) -> Optional[datetime]:
    if not os.path.exists(path):
        return None
    return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)


def newest_timestamp(ts_list: List[Optional[datetime]]) -> Optional[datetime]:
    vals = [t for t in ts_list if t is not None]
    return max(vals) if vals else None


def acquire_lock(lock_path: str, max_age_seconds: int = 300) -> bool:
    if os.path.exists(lock_path):
        try:
            age = time.time() - os.path.getmtime(lock_path)
            if age > max_age_seconds:
                os.remove(lock_path)
            else:
                return False
        except Exception:
            return False
    try:
        with open(lock_path, "w") as f:
            f.write(str(time.time()))
        return True
    except Exception:
        return False


def release_lock(lock_path: str) -> None:
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


def get_token(name: str) -> Optional[str]:
    try:
        if name in st.secrets:
            v = str(st.secrets[name]).strip()
            return v if v else None
    except Exception:
        pass
    v2 = os.getenv(name)
    if v2 and v2.strip():
        return v2.strip()
    return None


def fetch_xml(url: str, token: str) -> bytes:
    r = requests.get(url, headers=HEADERS, params={"token": token}, timeout=90)
    if r.status_code == 403:
        raise RuntimeError("HTTP 403 (token rejected). Check token for this endpoint.")
    r.raise_for_status()
    return r.content


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def text_clean(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = x.strip()
    return s if s else None


def find_first_by_local(elem: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if elem is None:
        return None
    for n in elem.iter():
        if local_name(n.tag) == name:
            return n
    return None


def find_all_by_local(elem: Optional[ET.Element], name: str) -> List[ET.Element]:
    if elem is None:
        return []
    out: List[ET.Element] = []
    for n in elem.iter():
        if local_name(n.tag) == name:
            out.append(n)
    return out


def text_by_local(elem: Optional[ET.Element], name: str) -> Optional[str]:
    n = find_first_by_local(elem, name)
    return text_clean(n.text) if n is not None else None


def texts_by_local(elem: Optional[ET.Element], name: str) -> List[str]:
    nodes = find_all_by_local(elem, name)
    out: List[str] = []
    for n in nodes:
        t = text_clean(n.text)
        if t is not None:
            out.append(t)
    return out


def uniq_join(values: List[str], sep: str = " | ") -> Optional[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        vv = v.strip()
        if vv and vv not in seen:
            seen.add(vv)
            out.append(vv)
    return sep.join(out) if out else None


def to_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def to_degree(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return v / 1e6 if abs(v) > 180 else v


def extract_point_latlon_any(rec: ET.Element) -> Tuple[Optional[float], Optional[float]]:
    lat = to_degree(to_float(text_by_local(rec, "latitude")))
    lon = to_degree(to_float(text_by_local(rec, "longitude")))
    if lat is not None or lon is not None:
        return lat, lon
    pt = find_first_by_local(rec, "point")
    if pt is None:
        return None, None
    lat2 = to_degree(to_float(text_by_local(pt, "latitude")))
    lon2 = to_degree(to_float(text_by_local(pt, "longitude")))
    return lat2, lon2


def is_event_record(elem: ET.Element) -> bool:
    needles = {
        "id",
        "updateTime",
        "startTime",
        "clearTime",
        "locationName",
        "travelDirection",
        "eventType",
        "eventStatus",
        "latitude",
        "longitude",
        "five11Message",
        "description",
    }
    hits = 0
    for n in elem.iter():
        if local_name(n.tag) in needles:
            hits += 1
        if hits >= 2:
            return True
    return False


def extract_event_records(root: ET.Element) -> List[ET.Element]:
    candidates: List[ET.Element] = []
    for e in root.iter():
        for c in list(e):
            if is_event_record(c):
                candidates.append(c)
    if candidates:
        return candidates
    fallback: List[ET.Element] = []
    for e in root.iter():
        if is_event_record(e) and find_first_by_local(e, "id") is not None:
            fallback.append(e)
    seen = set()
    uniq: List[ET.Element] = []
    for e in fallback:
        if id(e) not in seen:
            seen.add(id(e))
            uniq.append(e)
    return uniq


def parse_events_xml(xml_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)
    records = extract_event_records(root)
    rows: List[Dict[str, Any]] = []
    for rec in records:
        incident_id = text_by_local(rec, "id") or text_by_local(rec, "eventID") or text_by_local(rec, "incidentID")
        sender_incident_id = text_by_local(rec, "senderIncidentID") or text_by_local(rec, "sender_incident_id")
        update_time = text_by_local(rec, "updateTime")
        start_time = text_by_local(rec, "startTime")
        clear_time = text_by_local(rec, "clearTime")
        event_type = text_by_local(rec, "eventType") or text_by_local(rec, "type") or text_by_local(rec, "category")
        status = text_by_local(rec, "eventStatus") or text_by_local(rec, "roadStatus") or text_by_local(rec, "status")
        description = text_by_local(rec, "description") or uniq_join(texts_by_local(rec, "five11Message")) or uniq_join(
            texts_by_local(rec, "eventDescription")
        )
        county = text_by_local(rec, "county")
        city = text_by_local(rec, "city")
        region = text_by_local(rec, "region")
        district = text_by_local(rec, "district")
        route = text_by_local(rec, "route") or text_by_local(rec, "routeLocation")
        location_name = text_by_local(rec, "locationName") or text_by_local(rec, "routeLocation")
        travel_direction = text_by_local(rec, "travelDirection")
        lat, lon = extract_point_latlon_any(rec)

        if not incident_id and not description and lat is None and lon is None:
            continue

        rows.append(
            {
                "incident_id": incident_id,
                "sender_incident_id": sender_incident_id,
                "update_time": update_time,
                "start_time": start_time,
                "clear_time": clear_time,
                "event_type": event_type,
                "status": status,
                "description": description,
                "county": county,
                "city": city,
                "region": region,
                "district": district,
                "route": route,
                "location_name": location_name,
                "travel_direction": travel_direction,
                "lat": lat,
                "lon": lon,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in ["update_time", "start_time", "clear_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_convert("US/Eastern")

    df = df.drop_duplicates(subset=["incident_id", "update_time", "lat", "lon"], keep="last")
    return df


NS = {
    "orci": "http://www.openroadsconsulting.org/weather",
    "ess": "http://www.openroadsconsulting.org/orci_ess",
    "qfree": "http://www.qfree.com/common",
}


def iter_by_local(elem: Optional[ET.Element], name: str) -> Iterable[ET.Element]:
    if elem is None:
        return []
    for child in elem.iter():
        if local_name(child.tag) == name:
            yield child


def to_iso(date_val: Optional[str], time_val: Optional[str], offset: Optional[str]) -> Optional[str]:
    if not date_val or not time_val or not offset:
        return None
    try:
        dt = datetime.strptime(f"{date_val}{time_val}", "%Y%m%d%H%M%S")
        sign = 1 if offset.startswith("+") else -1
        hours = int(offset[1:3])
        minutes = int(offset[3:5])
        tz = timezone(sign * timedelta(hours=hours, minutes=minutes))
        return dt.replace(tzinfo=tz).astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def to_float_degree(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        num = float(value)
    except Exception:
        return None
    if abs(num) > 180:
        return num / 1e6
    return num


def parse_weather_xml(xml_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = ET.fromstring(xml_bytes)
    rows_long: List[Dict[str, Any]] = []
    for station in root.findall("orci:station", NS):
        inventory = station.find("orci:inventory", NS)
        if inventory is None:
            continue

        inv_header = find_first_by_local(inventory, "device-inventory-header")
        org_info = find_first_by_local(inventory, "organization-information")

        org_id = text_by_local(org_info, "organization-id")
        device_id = text_by_local(inv_header, "device-id")
        device_name = text_by_local(inv_header, "device-name")
        lat = to_float_degree(text_by_local(inv_header, "latitude"))
        lon = to_float_degree(text_by_local(inv_header, "longitude"))

        data = station.find("orci:data", NS)
        if data is None:
            continue

        for sensor in iter_by_local(data, "ess-sensor"):
            ess_id = text_by_local(sensor, "ess-sensor-id")
            timestamp_elem = find_first_by_local(sensor, "ess-observation-timestamp")
            obs_iso = None
            dt_holder = find_first_by_local(sensor, "dateTimeHolder")
            if dt_holder is not None:
                obs_iso = text_by_local(dt_holder, "iso-8601")
            if obs_iso is None and timestamp_elem is not None:
                obs_iso = to_iso(
                    text_by_local(timestamp_elem, "date"),
                    text_by_local(timestamp_elem, "time"),
                    text_by_local(timestamp_elem, "offset"),
                )

            base = {
                "org_id": org_id,
                "station_device_id": device_id,
                "station_device_name": device_name,
                "lat": lat,
                "lon": lon,
                "ess_sensor_id": ess_id,
                "obs_iso8601": obs_iso,
            }

            obs_type = find_first_by_local(sensor, "ess-observation-type")
            if obs_type is not None:
                for bucket_name in ("weather-data", "surface-data", "subsurface-data"):
                    bucket = next((child for child in obs_type if local_name(child.tag) == bucket_name), None)
                    if bucket is None:
                        continue
                    for metric in list(bucket):
                        if metric.text and metric.text.strip():
                            rows_long.append(
                                {**base, "metric_full": f"{bucket_name}__{local_name(metric.tag)}", "value": metric.text.strip()}
                            )

            alert = find_first_by_local(sensor, "sensor-alert")
            if alert is not None:
                update_time = text_by_local(alert, "update-time")
                threshold = text_by_local(alert, "threshold")
                if update_time is not None:
                    rows_long.append({**base, "metric_full": "alert__update-time", "value": update_time})
                if threshold is not None:
                    rows_long.append({**base, "metric_full": "alert__threshold", "value": threshold})

    long_df = pd.DataFrame(rows_long)
    index_cols = ["org_id", "station_device_id", "station_device_name", "lat", "lon", "obs_iso8601"]

    if long_df.empty:
        wide_df = pd.DataFrame(columns=index_cols)
        return long_df, wide_df

    long_df["obs_iso8601"] = pd.to_datetime(long_df["obs_iso8601"], errors="coerce", utc=True).dt.tz_convert("US/Eastern")

    wide_df = long_df.pivot_table(index=index_cols, columns="metric_full", values="value", aggfunc="first").reset_index()
    for col in wide_df.columns:
        if col in index_cols:
            continue
        wide_df[col] = pd.to_numeric(wide_df[col], errors="coerce")

    return long_df, wide_df


def ingest_if_due() -> None:
    last_events = file_mtime_utc(EVENTS_CSV)
    last_weather = newest_timestamp([file_mtime_utc(WEATHER_LONG_CSV), file_mtime_utc(WEATHER_WIDE_CSV)])
    last_any = newest_timestamp([last_events, last_weather])

    due = last_any is None or (utc_now() - last_any) >= timedelta(seconds=AUTO_INGEST_SECONDS)
    if not due:
        return

    if not acquire_lock(LOCK_FILE):
        return

    try:
        tok_events = get_token("ITERIS_TOKEN_EVENTFILTERED") or get_token("ITERIS_TOKEN_EVENTS")
        tok_weather = get_token("ITERIS_TOKEN_WEATHER")
        if not tok_events or not tok_weather:
            raise RuntimeError("Missing tokens: set secrets ITERIS_TOKEN_EVENTFILTERED (or ITERIS_TOKEN_EVENTS) and ITERIS_TOKEN_WEATHER")

        ev_xml = fetch_xml(EVENTS_URL, tok_events)
        ev_df = parse_events_xml(ev_xml)
        ev_df.to_csv(EVENTS_CSV, index=False)

        wx_xml = fetch_xml(WEATHER_URL, tok_weather)
        wlong, wwide = parse_weather_xml(wx_xml)
        wlong.to_csv(WEATHER_LONG_CSV, index=False)
        wwide.to_csv(WEATHER_WIDE_CSV, index=False)
    finally:
        release_lock(LOCK_FILE)


@st.cache_data(ttl=55)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def ensure_dt_et(df: pd.DataFrame, col: str) -> None:
    if df is None or df.empty or col not in df.columns:
        return
    s = pd.to_datetime(df[col], errors="coerce", utc=True)
    try:
        df[col] = s.dt.tz_convert("US/Eastern")
    except Exception:
        df[col] = s


def parse_loaded_events(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    for c in ["update_time", "start_time", "clear_time"]:
        ensure_dt_et(df, c)
    return df


def parse_loaded_weather_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    ensure_dt_et(df, "obs_iso8601")
    return df


def numeric_metric_columns(wwide: pd.DataFrame) -> List[str]:
    if wwide is None or wwide.empty:
        return []
    reserved = {"org_id", "station_device_id", "station_device_name", "lat", "lon", "obs_iso8601"}
    candidates = [c for c in wwide.columns if c not in reserved]
    keep: List[str] = []
    for c in candidates:
        s = pd.to_numeric(wwide[c], errors="coerce")
        if s.notna().any():
            keep.append(c)
    return sorted(keep)


def choose_default_metric(metrics: List[str], preferred: str) -> str:
    if preferred in metrics:
        return preferred
    return metrics[0] if metrics else preferred


def scale_series_auto(s: pd.Series) -> Tuple[pd.Series, str]:
    v = pd.to_numeric(s, errors="coerce")
    med = float(v.dropna().median()) if v.notna().any() else 0.0
    if med > 1_000_000:
        return v / 1e6, "÷ 1e6"
    if med > 10_000:
        return v / 1e3, "÷ 1e3"
    return v, "raw"


def safe_plotly(fig, height: int, key: str, bottom_margin: int = 70) -> None:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=bottom_margin))
    st.plotly_chart(fig, width="stretch", key=key)


def latest_weather_snapshot(wwide: pd.DataFrame) -> pd.DataFrame:
    if wwide is None or wwide.empty or "obs_iso8601" not in wwide.columns:
        return pd.DataFrame()
    t = wwide["obs_iso8601"].max()
    if pd.isna(t):
        return pd.DataFrame()
    return wwide[wwide["obs_iso8601"] == t].copy()


def weather_timeseries_for_station(wwide: pd.DataFrame, station_id: str, metric: str, hours: int) -> Tuple[pd.DataFrame, str]:
    if wwide is None or wwide.empty:
        return pd.DataFrame(), "raw"
    if "obs_iso8601" not in wwide.columns or "station_device_id" not in wwide.columns or metric not in wwide.columns:
        return pd.DataFrame(), "raw"

    cutoff = et_now() - timedelta(hours=int(hours))
    sub = wwide[wwide["station_device_id"].astype(str) == str(station_id)].copy()
    sub = sub.dropna(subset=["obs_iso8601"])
    sub = sub[sub["obs_iso8601"] >= cutoff]

    scaled, scale_note = scale_series_auto(sub[metric])
    sub["val"] = scaled
    sub = sub.dropna(subset=["val"]).sort_values("obs_iso8601")
    return sub[["obs_iso8601", "val"]], scale_note


def weather_network_avg_timeseries(wwide: pd.DataFrame, metric: str, hours: int) -> Tuple[pd.DataFrame, str]:
    if wwide is None or wwide.empty:
        return pd.DataFrame(), "raw"
    if "obs_iso8601" not in wwide.columns or metric not in wwide.columns:
        return pd.DataFrame(), "raw"
    cutoff = et_now() - timedelta(hours=int(hours))
    tmp = wwide.dropna(subset=["obs_iso8601"]).copy()
    tmp = tmp[tmp["obs_iso8601"] >= cutoff]
    scaled, scale_note = scale_series_auto(tmp[metric])
    tmp["val"] = scaled
    tmp = tmp.dropna(subset=["val"])
    if tmp.empty:
        return pd.DataFrame(), scale_note
    agg = tmp.groupby("obs_iso8601")["val"].mean().reset_index()
    agg = agg.sort_values("obs_iso8601")
    return agg, scale_note


def topn_with_other(series: pd.Series, n: int, other_label: str = "Other") -> pd.DataFrame:
    s = series.fillna("Unknown").astype(str)
    vc = s.value_counts()
    if vc.empty:
        return pd.DataFrame(columns=["label", "count"])
    top = vc.head(n)
    rest = vc.iloc[n:].sum()
    out = top.reset_index()
    out.columns = ["label", "count"]
    if rest > 0:
        out = pd.concat([out, pd.DataFrame([{"label": other_label, "count": rest}])], ignore_index=True)
    return out


try:
    ingest_if_due()
except Exception as e:
    st.error(str(e))

events_df = parse_loaded_events(load_csv(EVENTS_CSV))
wwide = parse_loaded_weather_wide(load_csv(WEATHER_WIDE_CSV))

events_updated = file_mtime_utc(EVENTS_CSV)
weather_updated = newest_timestamp([file_mtime_utc(WEATHER_LONG_CSV), file_mtime_utc(WEATHER_WIDE_CSV)])

st.title("SmarterRoads Real-Time Monitoring Dashboard")

h1, h2, h3 = st.columns([1.3, 1.3, 2.0])
with h1:
    st.caption("Events last updated (ET)")
    st.write(fmt_et(events_updated))
with h2:
    st.caption("Weather last updated (ET)")
    st.write(fmt_et(weather_updated))
with h3:
    st.caption("Live update")
    st.write(f"Auto-ingest ~every {AUTO_INGEST_SECONDS}s | UI refresh every {AUTO_REFRESH_MS//1000}s")

tab_overview, tab_weather, tab_events = st.tabs(["Overview", "Weather", "Events"])

weather_metrics = numeric_metric_columns(wwide)
default_metric = choose_default_metric(weather_metrics, DEFAULT_WEATHER_METRIC)
default_vis = choose_default_metric(weather_metrics, DEFAULT_VIS_METRIC)
default_wind = choose_default_metric(weather_metrics, DEFAULT_WIND_METRIC)

with tab_overview:
    st.subheader("Key Metrics")

    total_incidents = int(len(events_df)) if events_df is not None and not events_df.empty else 0
    if events_df is not None and not events_df.empty and "status" in events_df.columns:
        active_incidents = int(events_df["status"].fillna("").str.contains("active", case=False, na=False).sum())
    else:
        active_incidents = total_incidents

    new_last_hour = 0
    if events_df is not None and not events_df.empty and "update_time" in events_df.columns and pd.api.types.is_datetime64_any_dtype(events_df["update_time"]):
        cutoff = et_now() - timedelta(hours=1)
        new_last_hour = int((events_df["update_time"] >= cutoff).sum())

    snap = latest_weather_snapshot(wwide)
    avg_temp = None
    avg_vis = None
    temp_note = "raw"
    vis_note = "raw"

    if snap is not None and not snap.empty and weather_metrics:
        if default_metric in snap.columns:
            v, temp_note = scale_series_auto(snap[default_metric])
            avg_temp = float(v.dropna().mean()) if v.notna().any() else None
        if default_vis in snap.columns:
            v, vis_note = scale_series_auto(snap[default_vis])
            avg_vis = float(v.dropna().mean()) if v.notna().any() else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Incidents", f"{total_incidents:,}")
    k2.metric("Active Incidents", f"{active_incidents:,}")
    k3.metric(f"Avg Air Temp ({temp_note})", "—" if avg_temp is None else f"{avg_temp:.2f}")
    k4.metric(f"Avg Visibility ({vis_note})", "—" if avg_vis is None else f"{avg_vis:.2f}")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Incidents Trend (last 24h)")
        if events_df is not None and not events_df.empty and "update_time" in events_df.columns and pd.api.types.is_datetime64_any_dtype(events_df["update_time"]):
            tmp = events_df.dropna(subset=["update_time"]).copy()
            cutoff = et_now() - timedelta(hours=24)
            tmp = tmp[tmp["update_time"] >= cutoff]
            if not tmp.empty:
                tmp["bucket"] = tmp["update_time"].dt.floor("10min")
                trend = tmp.groupby("bucket").size().reset_index(name="count").sort_values("bucket")
                fig = px.line(trend, x="bucket", y="count", markers=True, title="Incidents per 10 minutes (ET)")
                safe_plotly(fig, 360, key="ov_incidents_trend", bottom_margin=10)
            else:
                st.info("No recent events in the last 24h window.")
        else:
            st.info("No events available yet.")

    with c2:
        st.subheader("Weather Trend (Network Avg)")
        if wwide is not None and not wwide.empty and weather_metrics:
            met = st.selectbox(
                "Metric",
                weather_metrics,
                index=weather_metrics.index(default_metric) if default_metric in weather_metrics else 0,
                key="ov_net_metric",
            )
            hours = st.slider("Time window (hours)", 1, 48, 12, key="ov_net_hours")
            ts, scale_note = weather_network_avg_timeseries(wwide, met, hours)
            if not ts.empty:
                fig = px.line(ts, x="obs_iso8601", y="val", markers=True, title=f"Network Avg: {met} ({scale_note})")
                safe_plotly(fig, 360, key="ov_weather_net_trend", bottom_margin=10)
            else:
                st.info("Weather trend metric not available yet.")
        else:
            st.info("No weather data available yet.")

    st.divider()

    b1, b2 = st.columns(2)
    with b1:
        st.subheader("Incidents by Type (Top 12 + Other)")
        if events_df is not None and not events_df.empty and "event_type" in events_df.columns:
            top = topn_with_other(events_df["event_type"], 12, other_label="Other")
            if not top.empty:
                fig = px.bar(top, x="label", y="count", title="Incident types")
                safe_plotly(fig, 360, key="ov_incidents_type", bottom_margin=130)
            else:
                st.info("No incident types available yet.")
        else:
            st.info("No incident types available yet.")

    with b2:
        st.subheader("Incidents by District (Top 12 + Other)")
        if events_df is not None and not events_df.empty and "district" in events_df.columns:
            top = topn_with_other(events_df["district"], 12, other_label="Other")
            if not top.empty:
                fig = px.bar(top, x="label", y="count", title="Districts")
                safe_plotly(fig, 360, key="ov_incidents_district", bottom_margin=130)
            else:
                st.info("No district data available yet.")
        else:
            st.info("No district data available yet.")

with tab_weather:
    st.subheader("Weather Monitoring")
    if wwide is None or wwide.empty or not weather_metrics:
        st.info("Weather data not available yet.")
    else:
        snap = latest_weather_snapshot(wwide)
        stations = sorted(wwide["station_device_id"].dropna().astype(str).unique().tolist())

        st.sidebar.header("Weather Controls")
        met = st.sidebar.selectbox(
            "Weather metric",
            weather_metrics,
            index=weather_metrics.index(default_metric) if default_metric in weather_metrics else 0,
            key="w_metric",
        )
        stn = st.sidebar.selectbox("Station", stations, index=0, key="w_station")
        hours = st.sidebar.slider("Trend window (hours)", 1, 48, 12, key="w_hours")

        top_row = st.columns(4)
        if snap is not None and not snap.empty and met in snap.columns:
            snap2 = snap.dropna(subset=["lat", "lon"]).copy()
            snap2["val"], scale_note = scale_series_auto(snap2[met])
            station_val = None
            sv = snap2[snap2["station_device_id"].astype(str) == str(stn)]["val"]
            if not sv.dropna().empty:
                station_val = float(sv.dropna().iloc[0])

            top_row[0].metric("Selected metric", met)
            top_row[1].metric(f"Latest @ station ({scale_note})", "—" if station_val is None else f"{station_val:.3f}")
            top_row[2].metric(
                f"Avg (latest snapshot) ({scale_note})",
                "—" if snap2["val"].dropna().empty else f"{float(snap2['val'].dropna().mean()):.3f}",
            )
            top_row[3].metric("Stations reporting", f"{int(snap2['val'].notna().sum()):,}")
        else:
            st.warning("Latest snapshot not available yet for this metric.")

        c1, c2 = st.columns([1.35, 1.65])

        with c1:
            st.subheader("Trend (Station)")
            ts, scale_note = weather_timeseries_for_station(wwide, stn, met, hours)
            if not ts.empty:
                fig = px.line(ts, x="obs_iso8601", y="val", markers=True, title=f"{met} @ {stn} ({scale_note})")
                safe_plotly(fig, 420, key="weather_station_trend", bottom_margin=10)
            else:
                st.info("No usable numeric data for this station/metric in the selected window.")

        with c2:
            st.subheader("Map (Latest Snapshot)")
            if snap is not None and not snap.empty and met in snap.columns:
                map_df = snap.dropna(subset=["lat", "lon"]).copy()
                map_df["val"], _ = scale_series_auto(map_df[met])
                map_df = map_df.dropna(subset=["val"])
                if not map_df.empty:
                    fig = px.scatter_mapbox(
                        map_df,
                        lat="lat",
                        lon="lon",
                        color="val",
                        hover_name="station_device_name" if "station_device_name" in map_df.columns else "station_device_id",
                        zoom=6,
                        height=420,
                    )
                    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, width="stretch", key="weather_map")
                else:
                    st.info("No mappable values for this metric at the latest snapshot.")
            else:
                st.info("No map data available yet.")

        st.divider()

        d1, d2 = st.columns(2)
        with d1:
            st.subheader("Top Stations (Latest)")
            if snap is not None and not snap.empty and met in snap.columns:
                df = snap.copy()
                df["val"], _ = scale_series_auto(df[met])
                df = df.dropna(subset=["val"])
                if not df.empty:
                    df = df.sort_values("val", ascending=False).head(10)
                    df["label"] = df["station_device_name"].fillna(df["station_device_id"])
                    fig = px.bar(df, x="label", y="val", title=f"Top 10 stations by {met}")
                    safe_plotly(fig, 380, key="weather_top_stations", bottom_margin=130)
                else:
                    st.info("No values available to rank.")
            else:
                st.info("No snapshot values available.")

        with d2:
            st.subheader("Distribution (Latest)")
            if snap is not None and not snap.empty and met in snap.columns:
                df = snap.copy()
                df["val"], _ = scale_series_auto(df[met])
                df = df.dropna(subset=["val"])
                if not df.empty:
                    fig = px.histogram(df, x="val", nbins=30, title=f"Distribution of {met} across stations")
                    safe_plotly(fig, 380, key="weather_distribution", bottom_margin=70)
                else:
                    st.info("No values to plot.")
            else:
                st.info("No snapshot values available.")

with tab_events:
    st.subheader("Events / Incidents Monitoring")
    if events_df is None or events_df.empty:
        st.info("Events data not available yet.")
    else:
        m1, m2, m3, m4 = st.columns(4)

        total = int(len(events_df))
        active = int(events_df["status"].fillna("").str.contains("active", case=False, na=False).sum()) if "status" in events_df.columns else total

        last_hour = 0
        if "update_time" in events_df.columns and pd.api.types.is_datetime64_any_dtype(events_df["update_time"]):
            last_hour = int((events_df["update_time"] >= (et_now() - timedelta(hours=1))).sum())

        most_recent = None
        if "update_time" in events_df.columns and pd.api.types.is_datetime64_any_dtype(events_df["update_time"]):
            if not events_df["update_time"].dropna().empty:
                most_recent = events_df["update_time"].max()

        m1.metric("Total incidents", f"{total:,}")
        m2.metric("Active incidents", f"{active:,}")
        m3.metric("New (last 60 min)", f"{last_hour:,}")
        m4.metric("Most recent update (ET)", "—" if most_recent is None or pd.isna(most_recent) else most_recent.strftime("%Y-%m-%d %H:%M:%S %Z"))

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Incidents Over Time (last 24h)")
            if "update_time" in events_df.columns and pd.api.types.is_datetime64_any_dtype(events_df["update_time"]):
                tmp = events_df.dropna(subset=["update_time"]).copy()
                cutoff = et_now() - timedelta(hours=24)
                tmp = tmp[tmp["update_time"] >= cutoff]
                if not tmp.empty:
                    tmp["bucket"] = tmp["update_time"].dt.floor("10min")
                    trend = tmp.groupby("bucket").size().reset_index(name="count").sort_values("bucket")
                    fig = px.line(trend, x="bucket", y="count", markers=True, title="Incidents per 10 minutes (ET)")
                    safe_plotly(fig, 380, key="events_trend", bottom_margin=10)
                else:
                    st.info("No events in the last 24h window.")
            else:
                st.info("No update_time available yet.")

        with c2:
            st.subheader("Incidents by Type (Top 15 + Other)")
            if "event_type" in events_df.columns:
                top = topn_with_other(events_df["event_type"], 15, other_label="Other")
                if not top.empty:
                    fig = px.bar(top, x="label", y="count", title="Incident types")
                    safe_plotly(fig, 380, key="events_type_bar", bottom_margin=130)
                else:
                    st.info("No event_type values available yet.")
            else:
                st.info("No event_type column found in events data.")

        st.divider()

        d1, d2 = st.columns(2)
        with d1:
            st.subheader("Incidents by District (Top 12 + Other)")
            if "district" in events_df.columns:
                top = topn_with_other(events_df["district"], 12, other_label="Other")
                if not top.empty:
                    fig = px.bar(top, x="label", y="count", title="Districts")
                    safe_plotly(fig, 360, key="events_district_bar", bottom_margin=130)
                else:
                    st.info("No district values available yet.")
            else:
                st.info("No district column found in events data.")

        with d2:
            st.subheader("Incidents by Status")
            if "status" in events_df.columns:
                s = events_df["status"].fillna("Unknown").astype(str).value_counts().reset_index()
                s.columns = ["status", "count"]
                fig = px.bar(s, x="status", y="count", title="Incidents by status")
                safe_plotly(fig, 360, key="events_status_bar", bottom_margin=130)
            else:
                st.info("No status column found in events data.")

