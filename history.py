"""
history.py: loads the PUBLIC DATASET into the database.

Dataset: ERA5 hourly reanalysis (Copernicus Climate Change Service / ECMWF),
accessed through the free Open-Meteo Historical Weather API.
ERA5 is published with a ~5-day delay, so we load the 30 days ending 6 days ago.

It runs automatically at the start of every collector job (about every 5.5 hours)
and can also be run by hand:  python history.py
Re-running is safe: rows are matched on their timestamp and updated, never duplicated.
"""

import os
from datetime import date, datetime, timedelta, timezone

import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAYS = 30
DELAY_DAYS = 6          # ERA5 is ~5 days behind real time
DATASET = "ERA5 reanalysis (Copernicus/ECMWF) via Open-Meteo"
HOURLY = {               # our column -> Open-Meteo variable
    "temperature": "temperature_2m",
    "humidity": "relative_humidity_2m",
    "pressure": "pressure_msl",
    "wind_speed": "wind_speed_10m",
    "wind_gust": "wind_gusts_10m",
    "cloud_cover": "cloud_cover",
    "precipitation": "precipitation",
}


def date_range(today=None, days=DAYS, delay=DELAY_DAYS):
    today = today or date.today()
    end = today - timedelta(days=delay)
    start = end - timedelta(days=days - 1)
    return start, end


def fetch_history(lat, lon, today=None):
    start, end = date_range(today)
    r = requests.get(ARCHIVE_URL, params={
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY.values()),
        "models": "era5",
        "wind_speed_unit": "ms", "timezone": "UTC",
    }, timeout=60)
    r.raise_for_status()
    return parse_history(r.json())


def parse_history(payload):
    """Turn Open-Meteo's column lists into one row per hour; skip hours with no data."""
    h = payload["hourly"]
    loaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def column(var):  # with a single model the name is plain; be safe if it gets a suffix
        return h.get(var) or h.get(f"{var}_era5") or [None] * len(h["time"])

    cols = {k: column(v) for k, v in HOURLY.items()}
    rows = []
    for i, t in enumerate(h["time"]):
        row = {k: cols[k][i] for k in HOURLY}
        if row["temperature"] is None:
            continue
        row["ts"] = t if t.endswith("Z") else f"{t}:00Z"   # "2026-10-01T05:00" -> UTC
        row["dataset"] = DATASET
        row["loaded_at"] = loaded_at
        rows.append(row)
    return rows


def refresh_history(sb, lat, lon):
    rows = fetch_history(lat, lon)
    for i in range(0, len(rows), 500):
        sb.table("history").upsert(rows[i:i + 500], on_conflict="ts").execute()
    return rows


if __name__ == "__main__":
    import collector   # reuses its database connection and location settings
    loaded = refresh_history(collector.sb, collector.LAT, collector.LON)
    if loaded:
        print(f"[history] {len(loaded)} hourly records, {loaded[0]['ts']} to {loaded[-1]['ts']}")
    else:
        print("[history] no records returned")
