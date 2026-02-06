import requests
import pandas as pd
import xml.etree.ElementTree as ET

API_URL = "https://data.511-atis-ttrip-prod.iteriscloud.com/smarterRoads/weather/weatherTMDD/current/weather_tmdd.xml"
API_TOKEN = "$2b$10$SiJ.34LhxY/5Q84fvVYw5uB1mMENe1p/du9/bFIhL7BCv9VH4bkVy"

OUTPUT_CSV = "weather_tmdd_clean_latest.csv"


def local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def normalize_coord(v):
    try:
        x = float(v)
        return x / 1e6 if abs(x) > 180 else x
    except:
        return None


def flatten_leaf_text(elem: ET.Element):
    """
    Flattens an XML subtree into {local_tag: first_nonempty_text}.
    If a local_tag repeats, we keep the first non-empty (good enough for MVP).
    """
    d = {}
    for e in elem.iter():
        if len(list(e)) != 0:
            continue
        t = (e.text or "").strip()
        if not t:
            continue
        k = local(e.tag)
        d.setdefault(k, t)
    return d


# Fetch XML
resp = requests.get(
    API_URL,
    params={"token": API_TOKEN},
    headers={"Accept": "application/xml"},
    timeout=90,
)
resp.raise_for_status()

root = ET.fromstring(resp.content)

rows = []
ess_count = 0

for node in root.iter():
    if local(node.tag) != "ess-sensor":
        continue

    ess_count += 1
    flat = flatten_leaf_text(node)

    # Build row using flattened keys (works even when values are nested under weather-data/surface-data)
    row = {
        "device_id": flat.get("device-id"),
        "device_name": flat.get("device-name"),
        "latitude": normalize_coord(flat.get("latitude")),
        "longitude": normalize_coord(flat.get("longitude")),
        "iso_8601": flat.get("iso-8601") or flat.get("ess-observation-timestamp") or flat.get("update-time"),

        # Weather metrics (may appear under nested sections, but flattening captures them)
        "air_temperature": flat.get("air-temperature"),
        "surface_temperature": flat.get("surface-temperature"),
        "relative_humidity": flat.get("relative-humidity"),
        "avg_wind_speed": flat.get("avg-wind-speed"),
        "precipitation_rate": flat.get("precipitation-rate"),
        "visibility_data": flat.get("visibility-data"),

        "surface_status": flat.get("surface-status"),
        "sensor_alert": flat.get("sensor-alert"),
    }

    rows.append(row)

print(f"ess-sensor elements found: {ess_count}")

df = pd.DataFrame(rows)

# Clean blanks -> NA
df = df.replace(r"^\s*$", pd.NA, regex=True)

# Parse timestamp
df["timestamp"] = pd.to_datetime(df["iso_8601"], errors="coerce", utc=True)

# Require at least station id + time
df = df.dropna(subset=["device_id", "timestamp"])

# Convert numeric fields
for c in ["air_temperature", "surface_temperature", "relative_humidity", "avg_wind_speed", "precipitation_rate", "visibility_data"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Keep rows that have at least one real metric
metric_cols = ["air_temperature", "surface_temperature", "relative_humidity", "avg_wind_speed", "precipitation_rate", "visibility_data"]
df = df.dropna(subset=metric_cols, how="all")

# Latest per device
df = df.sort_values("timestamp").groupby("device_id", as_index=False).tail(1)

# Final output columns
out = df[
    ["device_id", "device_name", "latitude", "longitude", "timestamp",
     "air_temperature", "surface_temperature", "relative_humidity",
     "avg_wind_speed", "precipitation_rate", "visibility_data",
     "surface_status", "sensor_alert"]
].copy()

out.to_csv(OUTPUT_CSV, index=False)
print(f"Clean rows written: {len(out)}")
print(f"Saved cleaned CSV → {OUTPUT_CSV}")
