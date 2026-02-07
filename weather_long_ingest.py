import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timezone, timedelta
from urllib.parse import quote
from typing import Optional, Iterator

URL_BASE = "https://data.511-atis-ttrip-prod.iteriscloud.com/smarterRoads/weather/weatherTMDD/current/weather_tmdd.xml"
TOKEN = "$2b$10$SiJ.34LhxY/5Q84fvVYw5uB1mMENe1p/du9/bFIhL7BCv9VH4bkVy"

NS = {
    "orci": "http://www.openroadsconsulting.org/weather",
    "ess": "http://www.openroadsconsulting.org/orci_ess",
    "qfree": "http://www.qfree.com/common",
}

def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag

def find_first_by_local(elem: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if elem is None:
        return None
    for child in elem.iter():
        if local_name(child.tag) == name:
            return child
    return None

def text_by_local(elem: Optional[ET.Element], name: str) -> Optional[str]:
    node = find_first_by_local(elem, name)
    return node.text.strip() if node is not None and node.text else None

def iter_by_local(elem: Optional[ET.Element], name: str) -> Iterator[ET.Element]:
    if elem is None:
        return
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
        return dt.replace(tzinfo=tz).isoformat()
    except (ValueError, IndexError):
        return None

def to_float_degree(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num / 1e6 if abs(num) > 180 else num

def build_url_with_token(token: str) -> str:
    return f"{URL_BASE}?token={quote(token.strip(), safe='')}"

req_url = build_url_with_token(TOKEN)

resp = requests.get(
    req_url,
    headers={"accept": "application/xml", "user-agent": "environmental-sensor-ingestion"},
    timeout=90,
)

if resp.status_code in (401, 403):
    raise RuntimeError(f"Auth failed (HTTP {resp.status_code}).\n{(resp.text or '')[:800]}")

resp.raise_for_status()
root = ET.fromstring(resp.content)

rows_long = []

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
                bucket = next(
                    (child for child in obs_type if local_name(child.tag) == bucket_name),
                    None,
                )
                if bucket is None:
                    continue
                for metric in list(bucket):
                    if metric.text:
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

wide = (
    long_df.pivot_table(index=index_cols, columns="metric_full", values="value", aggfunc="first").reset_index()
    if not long_df.empty
    else pd.DataFrame(columns=index_cols)
)

for col in wide.columns:
    if col not in index_cols:
        wide[col] = pd.to_numeric(wide[col], errors="coerce")

long_df.to_csv("Weather_sensor_observations_long.csv", index=False)
wide.to_csv("Weather_sensor_observations_wide.csv", index=False)
