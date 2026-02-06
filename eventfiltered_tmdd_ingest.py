#!/usr/bin/env python3
import os
import re
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Optional, Iterable, List, Dict, Any

URL = "https://data.511-atis-ttrip-prod.iteriscloud.com/smarterRoads/eventFiltered/eventFilteredTMDD/current/eventFiltered_tmdd.xml"

TOKEN = os.getenv("ITERIS_TOKEN_EVENTFILTERED") or "$2b$10$JpqMLCGpVoDVj0FwnKSM4O/y4QeWzv6zNw/UhuolJu2UffULsZACO"

HEADERS = {"Accept": "application/xml", "User-Agent": "tmdd-eventfiltered-clean"}

def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag

def text_clean(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = x.strip()
    return s if s else None

def iter_children(elem: ET.Element) -> Iterable[ET.Element]:
    return list(elem)

def find_first_by_local(elem: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if elem is None:
        return None
    for n in elem.iter():
        if local_name(n.tag) == name:
            return n
    return None

def text_by_local(elem: Optional[ET.Element], name: str) -> Optional[str]:
    n = find_first_by_local(elem, name)
    return text_clean(n.text) if n is not None else None

def texts_by_local(elem: Optional[ET.Element], name: str) -> List[str]:
    if elem is None:
        return []
    out = []
    for n in elem.iter():
        if local_name(n.tag) == name:
            t = text_clean(n.text)
            if t is not None:
                out.append(t)
    return out

def uniq_join(values: List[str], sep: str = " | ") -> Optional[str]:
    seen = set()
    out = []
    for v in values:
        v2 = v.strip()
        if v2 and v2 not in seen:
            seen.add(v2)
            out.append(v2)
    return sep.join(out) if out else None

def to_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except:
        return None

def to_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(float(x))
    except:
        return None

def to_degree(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return v / 1e6 if abs(v) > 180 else v

def pick_first_nonempty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        v2 = text_clean(v)
        if v2:
            return v2
    return None

def is_event_record(elem: ET.Element) -> bool:
    needles = {"id", "updateTime", "description", "startTime", "clearTime", "latitude", "longitude", "typeEvent"}
    found = set()
    for n in elem.iter():
        ln = local_name(n.tag)
        if ln in needles:
            found.add(ln)
        if len(found) >= 3:
            return True
    return False

def extract_event_records(root: ET.Element) -> List[ET.Element]:
    container_names = {
        "events", "event", "incidents", "incident",
        "eventFiltered", "eventFilteredData", "eventFilteredTMDD",
        "eventrecord", "eventRecord",
    }

    candidates: List[ET.Element] = []
    for e in root.iter():
        if local_name(e.tag) in container_names:
            for c in iter_children(e):
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

def main():
    if not TOKEN or not TOKEN.strip():
        raise RuntimeError("Token missing. Set ITERIS_TOKEN_EVENTFILTERED or put token in TOKEN variable.")

    with requests.Session() as s:
        r = s.get(URL, headers=HEADERS, params={"token": TOKEN}, timeout=90)
        r.raise_for_status()

    root = ET.fromstring(r.content)
    records = extract_event_records(root)
    if not records:
        raise RuntimeError("No event records found (feed may be empty or XML structure changed).")

    rows: List[Dict[str, Any]] = []

    for rec in records:
        incident_id = text_by_local(rec, "id")
        update_time = pick_first_nonempty(text_by_local(rec, "updateTime"), text_by_local(rec, "modify-time"))
        start_time = text_by_local(rec, "startTime")
        clear_time = text_by_local(rec, "clearTime")

        type_event = pick_first_nonempty(text_by_local(rec, "typeEvent"), text_by_local(rec, "roadwork"), text_by_local(rec, "cause"))
        status = pick_first_nonempty(text_by_local(rec, "status"), text_by_local(rec, "condition"))

        county = text_by_local(rec, "county")
        city = text_by_local(rec, "city")
        region = text_by_local(rec, "region")
        district = text_by_local(rec, "district")

        location_name = pick_first_nonempty(text_by_local(rec, "locationName"), text_by_local(rec, "pointName"), text_by_local(rec, "locationID"))
        travel_direction = text_by_local(rec, "travelDirection")
        route_location = pick_first_nonempty(text_by_local(rec, "routeLocation"), text_by_local(rec, "route"), text_by_local(rec, "patrolRoute"))

        lat_raw = to_float(text_by_local(rec, "latitude"))
        lon_raw = to_float(text_by_local(rec, "longitude"))
        lat = to_degree(lat_raw)
        lon = to_degree(lon_raw)

        lane_cnt = to_int(pick_first_nonempty(text_by_local(rec, "laneCnt"), text_by_local(rec, "lanesAffected")))
        lanes_affected = uniq_join(texts_by_local(rec, "affectedLane") + texts_by_local(rec, "lanesAffected"))

        desc = uniq_join(texts_by_local(rec, "description"))
        msg = uniq_join(texts_by_local(rec, "five11Message"))
        add_text = uniq_join(texts_by_local(rec, "five11AdditionalText"))

        full_text = uniq_join([v for v in [desc, msg, add_text] if v], sep=" | ")

        sender_incident_id = text_by_local(rec, "senderIncidentID")

        rows.append({
            "incident_id": incident_id,
            "sender_incident_id": sender_incident_id,
            "update_time": update_time,
            "start_time": start_time,
            "clear_time": clear_time,
            "event_type": type_event,
            "status": status,
            "description": full_text,
            "county": county,
            "city": city,
            "region": region,
            "district": district,
            "route": route_location,
            "location_name": location_name,
            "travel_direction": travel_direction,
            "lanes_affected": lanes_affected,
            "lane_count": lane_cnt,
            "lat": lat,
            "lon": lon,
        })

    df = pd.DataFrame(rows)

    for c in ["update_time", "start_time", "clear_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    df = df.drop_duplicates(subset=["incident_id"], keep="last")

    df.to_csv("event_filtered_clean.csv", index=False)

    print(f"Parsed {len(records)} raw records")
    print(f"Output rows (unique incident_id): {len(df)}")
    print("Wrote: event_filtered_clean.csv")

if __name__ == "__main__":
    main()
