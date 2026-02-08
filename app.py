import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import pytz
import requests
import streamlit as st
import xml.etree.ElementTree as ET
import plotly.express as px
import plotly.graph_objects as go
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

EASTERN = pytz.timezone("US/Eastern")

DEFAULT_WEATHER_METRIC = "weather-data__air-temperature"
DEFAULT_VIS_METRIC = "weather-data__visibility-data"
DEFAULT_WIND_METRIC = "weather-data__avg-wind-speed"
DEFAULT_RH_METRIC = "weather-data__relative-humidity"

METRIC_LABELS = {
    "weather-data__air-temperature": "Air Temperature",
    "weather-data__avg-wind-speed": "Avg Wind Speed",
    "weather-data__spot-wind-speed": "Spot Wind Speed",
    "weather-data__relative-humidity": "Relative Humidity",
    "weather-data__visibility-data": "Visibility",
    "weather-data__atmospheric-pressure": "Atmospheric Pressure",
    "weather-data__precipitation-rate": "Precipitation Rate",
    "weather-data__precipitation-one-hour": "Precipitation (1h)",
    "weather-data__precipitation-three-hour": "Precipitation (3h)",
    "weather-data__precipitation-six-hour": "Precipitation (6h)",
    "weather-data__precipitation-twelve-hour": "Precipitation (12h)",
    "weather-data__precipitation-24-hour": "Precipitation (24h)",
    "surface-data__surface-temperature": "Surface Temperature",
    "surface-data__surface-friction-index": "Surface Friction Index",
    "surface-data__surface-status": "Surface Status",
    "subsurface-data__subsurface-temperature": "Subsurface Temperature",
}

UNIT_HINTS = {
    "weather-data__air-temperature": "°C (raw)",
    "weather-data__avg-wind-speed": "raw",
    "weather-data__spot-wind-speed": "raw",
    "weather-data__relative-humidity": "% (raw)",
    "weather-data__visibility-data": "raw",
    "weather-data__atmospheric-pressure": "raw",
    "surface-data__surface-temperature": "°C (raw)",
    "subsurface-data__subsurface-temperature": "°C (raw)",
}

st.set_page_config(page_title="SmarterRoads Real-Time Monitoring Dashboard", layout="wide")
st_autorefresh(interval=AUTO_REFRESH_MS, key="auto_refresh")


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


def normalize_lane_count(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = x.strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


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
        lanes_affected = text_by_local(rec, "lanesAffected") or text_by_local(rec, "lanes_affected")
        lane_count = normalize_lane_count(text_by_local(rec, "laneCount") or text_by_local(rec, "lane_count"))
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
                "lanes_affected": lanes_affected,
                "lane_count": lane_count,
                "lat": lat,
                "lon": lon,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in ["update_time", "start_time", "clear_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
            if pd.api.types.is_datetime64tz_dtype(df[c]):
                df[c] = df[c].dt.tz_convert("US/Eastern")

    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    if "lane_count" in df.columns:
        df["lane_count"] = pd.to_numeric(df["lane_count"], errors="coerce")

    df = df.drop_duplicates(subset=["incident_id", "update_time", "lat", "lon"], keep="last")
    return df


NS = {
    "orci": "http://www.openroadsconsulting.org/weather",
    "ess": "http://www.openroadsconsulting.org/orci_ess",
    "qfree": "http://www.qfree.com/common",
}


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


def iter_by_local(elem: Optional[ET.Element], name: str):
    if elem is None:
        return
    for child in elem.iter():
        if local_name(child.tag) == name:
            yield child


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
                                {
                                    **base,
                                    "metric_full": f"{bucket_name}__{local_name(metric.tag)}",
                                    "value": metric.text.strip(),
                                }
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

    long_df["lat"] = pd.to_numeric(long_df["lat"], errors="coerce")
    long_df["lon"] = pd.to_numeric(long_df["lon"], errors="coerce")
    long_df["obs_iso8601"] = pd.to_datetime(long_df["obs_iso8601"], errors="coerce", utc=True)
    if pd.api.types.is_datetime64tz_dtype(long_df["obs_iso8601"]):
        long_df["obs_iso8601"] = long_df["obs_iso8601"].dt.tz_convert("US/Eastern")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    wide_df = long_df.pivot_table(index=index_cols, columns="metric_full", values="value", aggfunc="first").reset_index()

    for c in ["lat", "lon"]:
        if c in wide_df.columns:
            wide_df[c] = pd.to_numeric(wide_df[c], errors="coerce")

    if "obs_iso8601" in wide_df.columns:
        wide_df["obs_iso8601"] = pd.to_datetime(wide_df["obs_iso8601"], errors="coerce", utc=True)
        if pd.api.types.is_datetime64tz_dtype(wide_df["obs_iso8601"]):
            wide_df["obs_iso8601"] = wide_df["obs_iso8601"].dt.tz_convert("US/Eastern")

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


def parse_loaded_events(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    for c in ["update_time", "start_time", "clear_time"]:
        if c not in df.columns:
            continue
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        if pd.api.types.is_datetime64tz_dtype(df[c]):
            df[c] = df[c].dt.tz_convert("US/Eastern")
    for c in ["lat", "lon", "lane_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_loaded_weather_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    for c in ["lat", "lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "obs_iso8601" in df.columns:
        df["obs_iso8601"] = pd.to_datetime(df["obs_iso8601"], errors="coerce", utc=True)
        if pd.api.types.is_datetime64tz_dtype(df["obs_iso8601"]):
            df["obs_iso8601"] = df["obs_iso8601"].dt.tz_convert("US/Eastern")
    reserved = {"org_id", "station_device_id", "station_device_name", "lat", "lon", "obs_iso8601"}
    for c in df.columns:
        if c in reserved:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
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
    keep = [m for m in keep if not m.startswith("alert__")]
    return sorted(keep)


def choose_default_metric(metrics: List[str], preferred: str) -> str:
    return preferred if preferred in metrics else (metrics[0] if metrics else preferred)


def metric_label(m: str) -> str:
    base = METRIC_LABELS.get(m, m)
    u = UNIT_HINTS.get(m)
    return f"{base} ({u})" if u else base


def safe_plotly(fig, height: int, key: str, bottom_margin: int = 70) -> None:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=bottom_margin))
    st.plotly_chart(fig, width="stretch", key=key)


def latest_weather_snapshot(wwide: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    if wwide is None or wwide.empty:
        return pd.DataFrame()
    if "obs_iso8601" not in wwide.columns or "station_device_id" not in wwide.columns:
        return pd.DataFrame()
    cutoff = et_now() - timedelta(minutes=int(minutes))
    sub = wwide.dropna(subset=["obs_iso8601"]).copy()
    sub = sub[sub["obs_iso8601"] >= cutoff]
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values("obs_iso8601", ascending=False)
    sub = sub.drop_duplicates(subset=["station_device_id"], keep="first")
    sub["lat"] = pd.to_numeric(sub.get("lat"), errors="coerce")
    sub["lon"] = pd.to_numeric(sub.get("lon"), errors="coerce")
    return sub


def station_series(wwide: pd.DataFrame, station_id: str, metric: str, hours: int) -> pd.DataFrame:
    if wwide is None or wwide.empty:
        return pd.DataFrame()
    if "obs_iso8601" not in wwide.columns or "station_device_id" not in wwide.columns or metric not in wwide.columns:
        return pd.DataFrame()
    cutoff = et_now() - timedelta(hours=int(hours))
    sub = wwide[wwide["station_device_id"].astype(str) == str(station_id)].copy()
    sub = sub.dropna(subset=["obs_iso8601"])
    sub = sub[sub["obs_iso8601"] >= cutoff]
    sub["val"] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=["val"]).sort_values("obs_iso8601")
    return sub[["obs_iso8601", "val"]]


def network_trend(wwide: pd.DataFrame, metric: str, hours: int = 24, bucket: str = "30min") -> pd.DataFrame:
    if wwide is None or wwide.empty or "obs_iso8601" not in wwide.columns or metric not in wwide.columns:
        return pd.DataFrame()
    cutoff = et_now() - timedelta(hours=int(hours))
    sub = wwide.dropna(subset=["obs_iso8601"]).copy()
    sub = sub[sub["obs_iso8601"] >= cutoff]
    sub["val"] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=["val"])
    if sub.empty:
        return pd.DataFrame()
    sub["bucket"] = sub["obs_iso8601"].dt.floor(bucket)
    out = sub.groupby("bucket")["val"].mean().reset_index().sort_values("bucket")
    out["smooth"] = out["val"].rolling(3, min_periods=1).mean()
    return out


def events_trend(events_df: pd.DataFrame, hours: int = 24, bucket: str = "30min") -> pd.DataFrame:
    if events_df is None or events_df.empty or "update_time" not in events_df.columns:
        return pd.DataFrame()
    cutoff = et_now() - timedelta(hours=int(hours))
    tmp = events_df.dropna(subset=["update_time"]).copy()
    tmp = tmp[tmp["update_time"] >= cutoff]
    if tmp.empty:
        return pd.DataFrame()
    tmp["bucket"] = tmp["update_time"].dt.floor(bucket)
    out = tmp.groupby("bucket").size().reset_index(name="count").sort_values("bucket")
    out["smooth"] = out["count"].rolling(3, min_periods=1).mean()
    return out


try:
    ingest_if_due()
except Exception as e:
    st.error(str(e))

events_df = parse_loaded_events(load_csv(EVENTS_CSV))
wwide = parse_loaded_weather_wide(load_csv(WEATHER_WIDE_CSV))

events_updated = file_mtime_utc(EVENTS_CSV)
weather_updated = newest_timestamp([file_mtime_utc(WEATHER_LONG_CSV), file_mtime_utc(WEATHER_WIDE_CSV)])

weather_metrics = numeric_metric_columns(wwide)
default_metric = choose_default_metric(weather_metrics, DEFAULT_WEATHER_METRIC)
default_vis = choose_default_metric(weather_metrics, DEFAULT_VIS_METRIC)
default_wind = choose_default_metric(weather_metrics, DEFAULT_WIND_METRIC)
default_rh = choose_default_metric(weather_metrics, DEFAULT_RH_METRIC)

st.title("SmarterRoads Real-Time Monitoring Dashboard")

h1, h2, h3 = st.columns([1.2, 1.2, 2.2])
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

with tab_overview:
    snap = latest_weather_snapshot(wwide, minutes=30)

    total_records = int(len(events_df)) if events_df is not None and not events_df.empty else 0
    active_incidents = (
        int(events_df["status"].fillna("").str.contains("active", case=False, na=False).sum())
        if events_df is not None and not events_df.empty and "status" in events_df.columns
        else 0
    )
    new_last_hour = 0
    if events_df is not None and not events_df.empty and "update_time" in events_df.columns:
        cutoff = et_now() - timedelta(hours=1)
        new_last_hour = int((events_df["update_time"] >= cutoff).sum())

    avg_air = None
    avg_vis = None
    if snap is not None and not snap.empty:
        if default_metric in snap.columns:
            v = pd.to_numeric(snap[default_metric], errors="coerce")
            avg_air = float(v.dropna().mean()) if v.notna().any() else None
        if default_vis in snap.columns:
            v = pd.to_numeric(snap[default_vis], errors="coerce")
            avg_vis = float(v.dropna().mean()) if v.notna().any() else None

    stations_reporting_air = 0
    if snap is not None and not snap.empty and default_metric in snap.columns:
        stations_reporting_air = int(pd.to_numeric(snap[default_metric], errors="coerce").notna().sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total records loaded", f"{total_records:,}")
    k2.metric("Active incidents", f"{active_incidents:,}")
    k3.metric("New incidents (last 1h)", f"{new_last_hour:,}")
    k4.metric(metric_label(default_metric), "—" if avg_air is None else f"{avg_air:,.2f}")
    k5.metric(metric_label(default_vis), "—" if avg_vis is None else f"{avg_vis:,.2f}")

    st.caption(f"Stations reporting {METRIC_LABELS.get(default_metric, default_metric)} in last 30 min: {stations_reporting_air:,}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Incidents Trend (last 24h)")
        tr = events_trend(events_df, hours=24, bucket="30min")
        if not tr.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tr["bucket"], y=tr["count"], mode="lines", name="Count"))
            fig.add_trace(go.Scatter(x=tr["bucket"], y=tr["smooth"], mode="lines", name="Smoothed"))
            fig.update_layout(title="Incidents per 30 min (ET)", xaxis_title="Time (ET)", yaxis_title="Count")
            safe_plotly(fig, 360, key="ov_inc_trend_v2", bottom_margin=10)
        else:
            st.info("No events available yet for the last 24h window.")

    with c2:
        st.subheader("Weather Trend (network average)")
        met_choices = weather_metrics if weather_metrics else []
        if met_choices:
            with st.expander("Controls", expanded=True):
                met = st.selectbox(
                    "Metric",
                    met_choices,
                    index=met_choices.index(default_metric) if default_metric in met_choices else 0,
                    format_func=metric_label,
                    key="ov_net_metric",
                )
                hours = st.slider("Time window (hours)", 6, 72, 24, key="ov_net_hours")
            nt = network_trend(wwide, met, hours=hours, bucket="30min")
            if not nt.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=nt["bucket"], y=nt["val"], mode="lines", name="Avg"))
                fig.add_trace(go.Scatter(x=nt["bucket"], y=nt["smooth"], mode="lines", name="Smoothed"))
                fig.update_layout(title=f"{metric_label(met)} (network avg)", xaxis_title="Time (ET)", yaxis_title="Value")
                safe_plotly(fig, 360, key="ov_weather_net_v2", bottom_margin=10)
            else:
                st.info("No weather data available yet for this metric/window.")
        else:
            st.info("No weather metrics available yet.")

    st.divider()

    b1, b2 = st.columns(2)
    with b1:
        st.subheader("Incidents by Type (Top 12)")
        if events_df is not None and not events_df.empty and "event_type" in events_df.columns:
            top = events_df["event_type"].fillna("Unknown").value_counts().head(12).reset_index()
            top.columns = ["event_type", "count"]
            fig = px.bar(top, x="event_type", y="count", title="Top incident types")
            safe_plotly(fig, 360, key="ov_types_v2", bottom_margin=120)
        else:
            st.info("No incident type data available yet.")

    with b2:
        st.subheader("Incidents by District (Top 12)")
        if events_df is not None and not events_df.empty and "district" in events_df.columns:
            top = events_df["district"].fillna("Unknown").value_counts().head(12).reset_index()
            top.columns = ["district", "count"]
            fig = px.bar(top, x="district", y="count", title="Top districts")
            safe_plotly(fig, 360, key="ov_districts_v2", bottom_margin=120)
        else:
            st.info("No district data available yet.")

with tab_weather:
    st.subheader("Weather Monitoring")

    if wwide is None or wwide.empty or not weather_metrics:
        st.info("Weather data not available yet.")
    else:
        stations = sorted(wwide["station_device_id"].dropna().astype(str).unique().tolist())
        with st.expander("Controls", expanded=True):
            met = st.selectbox(
                "Metric",
                weather_metrics,
                index=weather_metrics.index(default_metric) if default_metric in weather_metrics else 0,
                format_func=metric_label,
                key="w_metric_v2",
            )
            stn = st.selectbox("Station", stations, index=0, key="w_station_v2")
            hours = st.slider("Trend window (hours)", 6, 72, 24, key="w_hours_v2")
            snap_minutes = st.slider("Current snapshot window (minutes)", 10, 60, 30, key="w_snap_mins_v2")

        snap = latest_weather_snapshot(wwide, minutes=int(snap_minutes))

        top = st.columns(4)

        latest_station_val = None
        if snap is not None and not snap.empty and met in snap.columns:
            ssub = snap[snap["station_device_id"].astype(str) == str(stn)].copy()
            if not ssub.empty:
                v = pd.to_numeric(ssub[met], errors="coerce").dropna()
                if not v.empty:
                    latest_station_val = float(v.iloc[0])

        map_df = pd.DataFrame()
        if snap is not None and not snap.empty and met in snap.columns:
            map_df = snap.dropna(subset=["lat", "lon"]).copy()
            map_df["val"] = pd.to_numeric(map_df[met], errors="coerce")
            map_df = map_df.dropna(subset=["val"])
            map_df = map_df[(map_df["lat"].between(-90, 90)) & (map_df["lon"].between(-180, 180))]

        stations_reporting = int(map_df["val"].notna().sum()) if not map_df.empty else 0
        avg_snapshot = float(map_df["val"].mean()) if not map_df.empty else None

        top[0].metric("Selected metric", metric_label(met))
        top[1].metric("Latest @ station", "—" if latest_station_val is None else f"{latest_station_val:,.2f}")
        top[2].metric("Avg (current snapshot)", "—" if avg_snapshot is None else f"{avg_snapshot:,.2f}")
        top[3].metric("Stations reporting", f"{stations_reporting:,}")

        c1, c2 = st.columns([1.2, 1.8])
        with c1:
            st.subheader("Trend (selected station)")
            ts = station_series(wwide, stn, met, hours)
            if not ts.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts["obs_iso8601"], y=ts["val"], mode="lines", name="Value"))
                smooth = ts["val"].rolling(5, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=ts["obs_iso8601"], y=smooth, mode="lines", name="Smoothed"))
                fig.update_layout(title=f"{metric_label(met)} @ {stn}", xaxis_title="Time (ET)", yaxis_title="Value")
                safe_plotly(fig, 420, key="w_station_trend_v2", bottom_margin=10)
            else:
                st.info("No usable numeric data for this station/metric in the selected window.")

            st.subheader("Trend (network average)")
            nt = network_trend(wwide, met, hours=hours, bucket="30min")
            if not nt.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=nt["bucket"], y=nt["val"], mode="lines", name="Avg"))
                fig.add_trace(go.Scatter(x=nt["bucket"], y=nt["smooth"], mode="lines", name="Smoothed"))
                fig.update_layout(title=f"{metric_label(met)} (network avg)", xaxis_title="Time (ET)", yaxis_title="Value")
                safe_plotly(fig, 420, key="w_network_trend_v2", bottom_margin=10)
            else:
                st.info("No network trend available for this metric/window.")

        with c2:
            st.subheader("Map (current snapshot)")
            if not map_df.empty:
                fig = px.scatter_mapbox(
                    map_df,
                    lat="lat",
                    lon="lon",
                    color="val",
                    hover_name="station_device_name" if "station_device_name" in map_df.columns else "station_device_id",
                    hover_data={"station_device_id": True, "val": True},
                    zoom=6,
                    height=520,
                )
                fig.update_layout(mapbox_style="open-street-map", margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch", key="w_map_v2")
            else:
                st.info("No mappable values for this metric in the current snapshot window.")

            st.subheader("Distribution (current snapshot)")
            if not map_df.empty:
                fig = px.histogram(map_df, x="val", nbins=30, title=f"{metric_label(met)} distribution")
                safe_plotly(fig, 420, key="w_hist_v2", bottom_margin=70)
            else:
                st.info("No values available for distribution.")

with tab_events:
    st.subheader("Events Monitoring")

    if events_df is None or events_df.empty:
        st.info("Events data not available yet.")
    else:
        with st.expander("Controls", expanded=True):
            hours = st.slider("Time window (hours)", 6, 72, 24, key="e_hours_v2")
            bucket = st.selectbox("Trend bucket", ["10min", "15min", "30min", "60min"], index=2, key="e_bucket_v2")

        tr = events_trend(events_df, hours=hours, bucket=bucket)

        ev_map = events_df.copy()
        ev_map["lat"] = pd.to_numeric(ev_map.get("lat"), errors="coerce")
        ev_map["lon"] = pd.to_numeric(ev_map.get("lon"), errors="coerce")
        if "update_time" in ev_map.columns:
            cutoff = et_now() - timedelta(hours=int(hours))
            ev_map = ev_map.dropna(subset=["update_time"])
            ev_map = ev_map[ev_map["update_time"] >= cutoff]
        ev_map = ev_map.dropna(subset=["lat", "lon"])
        ev_map = ev_map[(ev_map["lat"].between(-90, 90)) & (ev_map["lon"].between(-180, 180))]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Records in window", f"{int(len(events_df)):,}")
        k2.metric("Mappable events", f"{int(len(ev_map)):,}")
        missing_geo = int(len(events_df) - len(ev_map))
        k3.metric("Missing location", f"{missing_geo:,}")
        k4.metric("Bucket", bucket)

        left, right = st.columns([1.2, 1.8])

        with left:
            st.subheader("Incidents Trend")
            if not tr.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tr["bucket"], y=tr["count"], mode="lines", name="Count"))
                fig.add_trace(go.Scatter(x=tr["bucket"], y=tr["smooth"], mode="lines", name="Smoothed"))
                fig.update_layout(title=f"Incidents per {bucket} (ET)", xaxis_title="Time (ET)", yaxis_title="Count")
                safe_plotly(fig, 380, key="e_trend_v2", bottom_margin=10)
            else:
                st.info("No events available for this window.")

            st.subheader("Top Types")
            if "event_type" in events_df.columns:
                top = events_df["event_type"].fillna("Unknown").value_counts().head(12).reset_index()
                top.columns = ["event_type", "count"]
                fig = px.bar(top, x="event_type", y="count", title="Top incident types")
                safe_plotly(fig, 380, key="e_types_v2", bottom_margin=120)
            else:
                st.info("No event_type available.")

        with right:
            st.subheader("Events Map")
            if not ev_map.empty:
                hover_cols = []
                for c in ["incident_id", "event_type", "status", "district", "route", "location_name", "update_time"]:
                    if c in ev_map.columns:
                        hover_cols.append(c)
                fig = px.scatter_mapbox(
                    ev_map,
                    lat="lat",
                    lon="lon",
                    color="event_type" if "event_type" in ev_map.columns else None,
                    hover_data=hover_cols,
                    zoom=6,
                    height=780,
                )
                fig.update_layout(mapbox_style="open-street-map", margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch", key="e_map_v2")
            else:
                st.info("No events with valid coordinates in this window.")

