"""
collector.py — the heart of the project. Runs every 10 minutes (GitHub Actions).

Each run:
  1. FETCH    current weather from 2 sources (OpenWeatherMap + Open-Meteo)
  2. STORE    the readings in Supabase
  3. CHECK    the prediction made 10 minutes ago against the real temperature
  4. LEARN    from that mistake (online learning with `river`)
  5. PREDICT  the temperature 10 minutes from now and save it
  6. SAVE     the updated model back to the database

The model predicts the CHANGE in temperature over the next 10 minutes,
then adds it to the current temperature. Its accuracy is compared with a
simple "no change" baseline to prove it is actually learning something.
"""

import base64
import math
import os
import pickle
import re
import sys
from datetime import datetime, timedelta, timezone

import requests
from river import compose, linear_model, optim, preprocessing
from supabase import create_client

try:  # lets you use a .env file when running on your own computer
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------- Settings ----------------
LAT = float(os.getenv("LAT", "49.2827"))     # default: Vancouver
LON = float(os.getenv("LON", "-123.1207"))
HORIZON_MIN = 10                              # predict this many minutes ahead
MAX_DELTA = 1.5                               # cap on predicted change (°C) for safety
FEATURES = ["temp", "humidity", "pressure", "wind", "cloud",
            "hour_sin", "hour_cos", "temp_trend", "pressure_trend"]
OWM_KEY = os.getenv("OWM_API_KEY")
OM_URL = "https://api.open-meteo.com/v1/forecast"
OWM_URL = "https://api.openweathermap.org/data/2.5/weather"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
AQ_FIELDS = ["pm2_5", "pm10", "us_aqi", "ozone", "nitrogen_dioxide", "carbon_monoxide"]
FIELDS = ["temperature", "humidity", "pressure", "wind_speed", "cloud_cover"]

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


# ---------------- Time helpers ----------------
def now_utc():
    return datetime.now(timezone.utc)


def iso(dt):
    """UTC timestamp string that is safe to put in a database query."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s):
    """Parse timestamps from Supabase/Open-Meteo on any Python 3.8+."""
    s = s.replace("Z", "+00:00")
    m = re.match(r"^(.*T\d\d:\d\d:\d\d)\.(\d+)(.*)$", s)
    if m:
        s = f"{m.group(1)}.{m.group(2)[:6].ljust(6, '0')}{m.group(3)}"
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------- 1. Fetch ----------------
def fetch_openweathermap():
    if not OWM_KEY:
        raise RuntimeError("OWM_API_KEY not set")
    r = requests.get(OWM_URL, params={"lat": LAT, "lon": LON, "appid": OWM_KEY,
                                      "units": "metric"}, timeout=15)
    r.raise_for_status()
    d = r.json()
    return {
        "source": "openweathermap",
        "temperature": d["main"]["temp"],
        "humidity": d["main"]["humidity"],
        "pressure": d["main"]["pressure"],
        "wind_speed": d.get("wind", {}).get("speed"),
        "wind_gust": d.get("wind", {}).get("gust"),
        "cloud_cover": d.get("clouds", {}).get("all"),
        "precipitation": d.get("rain", {}).get("1h", 0.0),   # mm in the last hour
    }


def fetch_open_meteo():
    r = requests.get(OM_URL, params={
        "latitude": LAT, "longitude": LON,
        "current": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,"
                   "wind_gusts_10m,cloud_cover,precipitation",
        "wind_speed_unit": "ms", "timezone": "UTC",
    }, timeout=15)
    r.raise_for_status()
    c = r.json()["current"]
    return {
        "source": "open-meteo",
        "temperature": c["temperature_2m"],
        "humidity": c["relative_humidity_2m"],
        "pressure": c["pressure_msl"],
        "wind_speed": c["wind_speed_10m"],
        "wind_gust": c.get("wind_gusts_10m"),
        "cloud_cover": c["cloud_cover"],
        "precipitation": c.get("precipitation"),
    }


def fetch_air_quality():
    """Third live source: air quality (Open-Meteo, based on the CAMS model)."""
    r = requests.get(AQ_URL, params={
        "latitude": LAT, "longitude": LON,
        "current": ",".join(AQ_FIELDS), "timezone": "UTC",
    }, timeout=15)
    r.raise_for_status()
    c = r.json()["current"]
    return {"source": "open-meteo-air-quality", **{f: c.get(f) for f in AQ_FIELDS}}


def safe(fn):
    """Run a fetcher; if one source is down, keep going with the other."""
    try:
        return fn()
    except Exception as e:
        print(f"[warn] {fn.__name__} failed: {e}")
        return None


def fuse(readings):
    """Combine sources into one value per field (simple average)."""
    out = {}
    for f in FIELDS:
        vals = [r[f] for r in readings if r.get(f) is not None]
        out[f] = sum(vals) / len(vals) if vals else 0.0
    return out


# ---------------- Features ----------------
def make_features(cur, prev, ts, readings):
    """
    cur:  fused weather now
    prev: (features dict from last run, minutes ago) or None
    """
    # Note: no "difference between sources" feature. The 30-day history has only one
    # source, so that feature was always 0 during warm-start; live values then got
    # scaled to huge numbers and made the model's weights explode.
    hour = ts.hour + ts.minute / 60
    f = {
        "temp": cur["temperature"],
        "humidity": cur["humidity"],
        "pressure": cur["pressure"],
        "wind": cur["wind_speed"],
        "cloud": cur["cloud_cover"],
        "hour_sin": math.sin(2 * math.pi * hour / 24),   # time of day as a circle
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "temp_trend": 0.0,       # how fast temp is changing (per 10 min)
        "pressure_trend": 0.0,
    }
    if prev:
        pf, mins = prev
        scale = HORIZON_MIN / mins
        f["temp_trend"] = (cur["temperature"] - pf["temp"]) * scale
        f["pressure_trend"] = (cur["pressure"] - pf["pressure"]) * scale
    return f


# ---------------- Model ----------------
def new_model():
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        # small learning rate + gradient clipping so one surprising reading
        # can't throw the model off
        linear_model.LinearRegression(optimizer=optim.SGD(0.005), l2=0.001, clip_gradient=1.0),
    )


LEARNING_ID, FROZEN_ID = 1, 2      # rows in the model_state table


def load_model(model_id=LEARNING_ID):
    res = sb.table("model_state").select("*").eq("id", model_id).execute()
    if res.data:
        row = res.data[0]
        return pickle.loads(base64.b64decode(row["model_b64"])), row["n_learned"]
    return None, 0


def save_model(model, n_learned, model_id=LEARNING_ID):
    sb.table("model_state").upsert({
        "id": model_id,
        "model_b64": base64.b64encode(pickle.dumps(model)).decode(),
        "n_learned": n_learned,
        "updated_at": iso(now_utc()),
    }).execute()


def warm_start(model):
    """
    First run only: pre-train on the last 30 days of hourly Open-Meteo data
    so the model starts with a sense of the daily temperature pattern.
    Live 10-minute learning then refines it.
    """
    try:
        r = requests.get(OM_URL, params={
            "latitude": LAT, "longitude": LON,
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,cloud_cover",
            "wind_speed_unit": "ms", "timezone": "UTC",
            "past_days": 30, "forecast_days": 1,
        }, timeout=30)
        r.raise_for_status()
        h = r.json()["hourly"]
    except Exception as e:
        print(f"[warn] warm start skipped: {e}")
        return model, 0

    keymap = {"temperature": "temperature_2m", "humidity": "relative_humidity_2m",
              "pressure": "pressure_msl", "wind_speed": "wind_speed_10m",
              "cloud_cover": "cloud_cover"}
    cutoff = now_utc()
    rows = []
    for i, t in enumerate(h["time"]):
        vals = {k: h[v][i] for k, v in keymap.items()}
        ts = parse_ts(t)
        if ts > cutoff:
            break  # only use the past, not forecasts
        rows.append(None if any(x is None for x in vals.values()) else (ts, vals))

    samples = 0
    for _ in range(2):  # two passes over the history
        for i in range(1, len(rows) - 1):
            if None in (rows[i - 1], rows[i], rows[i + 1]):
                continue
            (_, prev_v), (t1, cur_v), (_, nxt_v) = rows[i - 1], rows[i], rows[i + 1]
            prev = ({"temp": prev_v["temperature"], "pressure": prev_v["pressure"]}, 60)
            feats = make_features(cur_v, prev, t1, [])
            target = (nxt_v["temperature"] - cur_v["temperature"]) * HORIZON_MIN / 60
            model.learn_one(feats, target)
            samples += 1
    return model, samples


def predict_delta(model, feats):
    """Predicted temperature change over the next 10 minutes, capped for safety."""
    return max(-MAX_DELTA, min(MAX_DELTA, model.predict_one(feats)))


# ---------------- Frozen model (the "no learning" comparison) ----------------
# Trained exactly like the learning model (30 days of history) but NEVER updated
# with live data. Comparing the two shows how much the live learning helps.
_frozen = None


def get_frozen():
    """Load the frozen model; create it (and score past predictions) the first time."""
    global _frozen
    if _frozen is None:
        _frozen, _ = load_model(FROZEN_ID)
    if _frozen is None:
        model, n = warm_start(new_model())
        if n == 0:
            raise RuntimeError("history unavailable; frozen model not created")
        save_model(model, 0, FROZEN_ID)
        _frozen = model
        print(f"[frozen] created from {n} historical samples (it will never learn)")
        backfill_frozen(_frozen)
    return _frozen


def backfill_frozen(model):
    """Score the frozen model on every past prediction, using the inputs saved with each one."""
    last_id, done = 0, 0
    while True:
        rows = (sb.table("predictions")
                .select("id,baseline_temp,actual_temp,features")
                .is_("frozen_temp", "null").gt("id", last_id)
                .order("id").limit(500).execute().data)
        if not rows:
            break
        for p in rows:
            last_id = p["id"]
            feats = {k: v for k, v in (p["features"] or {}).items() if k in FEATURES}
            if len(feats) != len(FEATURES):
                continue
            frozen_temp = p["baseline_temp"] + predict_delta(model, feats)
            update = {"frozen_temp": round(frozen_temp, 3)}
            if p["actual_temp"] is not None:
                update["frozen_error"] = abs(p["actual_temp"] - frozen_temp)
            sb.table("predictions").update(update).eq("id", p["id"]).execute()
            done += 1
    if done:
        print(f"[frozen] scored {done} past predictions")
    return done


# ---------------- 3 + 4. Check old predictions and learn ----------------
def resolve_pending(model, actual, now):
    rows = (sb.table("predictions")
            .select("id,predicted_temp,baseline_temp,frozen_temp,features")
            .is_("actual_temp", "null")
            .lte("target_time", iso(now + timedelta(minutes=2)))
            .gte("target_time", iso(now - timedelta(minutes=30)))
            .execute().data)
    for p in rows:
        # Learn the TRUE change that happened, from the inputs we had back then
        feats = {k: v for k, v in p["features"].items() if k in FEATURES}
        model.learn_one(feats, actual - p["baseline_temp"])
        update = {
            "actual_temp": actual,
            "error": abs(actual - p["predicted_temp"]),
            "baseline_error": abs(actual - p["baseline_temp"]),
            "resolved_at": iso(now),
        }
        frozen_note = ""
        if p.get("frozen_temp") is not None:
            update["frozen_error"] = abs(actual - p["frozen_temp"])
            frozen_note = f", frozen error {update['frozen_error']:.2f}"
        sb.table("predictions").update(update).eq("id", p["id"]).execute()
        print(f"[check] predicted {p['predicted_temp']:.2f}°C, actual {actual:.2f}°C, "
              f"error {update['error']:.2f} "
              f"(baseline error {update['baseline_error']:.2f}{frozen_note})")
    return len(rows)


def get_prev(now):
    """Features from the previous run, used to compute trends."""
    res = (sb.table("predictions").select("made_at,features")
           .order("made_at", desc=True).limit(1).execute())
    if not res.data:
        return None
    row = res.data[0]
    mins = (now - parse_ts(row["made_at"])).total_seconds() / 60
    return (row["features"], mins) if 3 <= mins <= 40 else None


# ---------------- Main cycle ----------------
def main():
    now = now_utc()

    # 1. Fetch
    readings = [r for r in (safe(fetch_openweathermap), safe(fetch_open_meteo)) if r]
    if not readings:
        print("[error] no source returned data; stopping this run")
        sys.exit(1)
    for r in readings:
        r["ts"] = iso(now)

    # 2. Store
    sb.table("readings").insert(readings).execute()
    cur = fuse(readings)
    print(f"[fetch] {len(readings)} source(s), fused temp {cur['temperature']:.2f}°C")

    # 2b. Air quality (third source). A failure here never stops the weather cycle.
    aq = safe(fetch_air_quality)
    if aq:
        try:
            aq["ts"] = iso(now)
            sb.table("air_quality").insert(aq).execute()
            print(f"[air] PM2.5 {aq['pm2_5']} µg/m³, US AQI {aq['us_aqi']}")
        except Exception as e:
            print(f"[warn] could not save air quality: {e}")

    # Load model (or create + warm start on the very first run)
    model, n_learned = load_model()
    if model is None:
        model, n = warm_start(new_model())
        print(f"[model] new model, warm-started on {n} historical samples")

    # 3 + 4. Check and learn
    n_learned += resolve_pending(model, cur["temperature"], now)

    # 5. Predict
    feats = make_features(cur, get_prev(now), now, readings)
    delta = predict_delta(model, feats)
    pred = cur["temperature"] + delta
    row = {
        "made_at": iso(now),
        "target_time": iso(now + timedelta(minutes=HORIZON_MIN)),
        "predicted_temp": round(pred, 3),
        "baseline_temp": cur["temperature"],
        "features": feats,
    }
    frozen_note = ""
    try:  # the frozen comparison must never stop the main cycle
        frozen_pred = cur["temperature"] + predict_delta(get_frozen(), feats)
        row["frozen_temp"] = round(frozen_pred, 3)
        frozen_note = f" | frozen model: {frozen_pred:.2f}°C"
    except Exception as e:
        print(f"[warn] frozen model skipped: {e}")
    sb.table("predictions").insert(row).execute()
    print(f"[predict] in {HORIZON_MIN} min: {pred:.2f}°C (change {delta:+.2f}){frozen_note}")

    # 6. Save
    save_model(model, n_learned)
    print(f"[model] saved, live learning steps so far: {n_learned}")


if __name__ == "__main__":
    main()
