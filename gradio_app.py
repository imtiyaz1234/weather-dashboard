"""
Site Weather Risk Monitor: Vancouver (Gradio dashboard)

Audience: construction site managers in Vancouver.
Decisions it supports: pour concrete or not, run crane lifts or pause, do exterior
finishing or switch crews indoors, schedule cold-stress breaks, limit heavy outdoor
work when the air is bad.

Data (all stored in one Supabase database by the collector):
  - OpenWeatherMap: live weather (API)
  - Open-Meteo: live weather (API)
  - Open-Meteo Air Quality: live PM2.5 / US AQI (API)
  - ERA5 reanalysis: 30 days of hourly history (public dataset)

The collector predicts the temperature 10 minutes ahead, checks every prediction
when the real value arrives, and learns from the error. A "frozen" copy of the
model never learns, so the dashboard can show how much the learning helps.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client

# ---------------- Settings ----------------
TZ = ZoneInfo(os.getenv("TIMEZONE", "America/Vancouver"))
LOCATION = os.getenv("LOCATION_NAME", "Vancouver, BC")
REFRESH_SECONDS = 60
CACHE_SECONDS = 50
ROLL = 18                 # rolling error over the last 18 checks (about 3 hours)
HORIZON_MIN = 10
STALE_MINUTES = 30
WORK_HOURS = range(7, 17)  # 7 AM to 5 PM local time

WINDOWS = {"Last 6 hours": 6, "Last 24 hours": 24, "Last 3 days": 72,
           "Last 7 days": 168, "All data": None}
SOURCES = ["OpenWeatherMap", "Open-Meteo"]
SOURCE_LABELS = {"openweathermap": "OpenWeatherMap", "open-meteo": "Open-Meteo"}
VARIABLES = {  # label -> (table, column, unit)
    "Temperature (°C)": ("readings", "temperature", "°C"),
    "Wind gust (m/s)": ("readings", "wind_gust", "m/s"),
    "Wind speed (m/s)": ("readings", "wind_speed", "m/s"),
    "Rain (mm)": ("readings", "precipitation", "mm"),
    "Humidity (%)": ("readings", "humidity", "%"),
    "Pressure (hPa)": ("readings", "pressure", "hPa"),
    "Cloud cover (%)": ("readings", "cloud_cover", "%"),
    "Air quality: US AQI": ("air_quality", "us_aqi", "AQI"),
    "Air quality: PM2.5 (µg/m³)": ("air_quality", "pm2_5", "µg/m³"),
}
DEFAULT_LIMITS = {"concrete": 5.0, "crane": 15.0, "rain": 0.5, "cold": 0.0, "aqi": 100.0}

# Colors follow the entity. Validated colorblind-safe categorical steps.
COLORS = {"OpenWeatherMap": "#2a78d6", "Open-Meteo": "#eb6834", "Air quality": "#1baf7a",
          "ERA5": "#6250d6", "actual": "#1baf7a", "learning": "#2a78d6",
          "frozen": "#6250d6", "baseline": "#8a8985", "limit": "#e34948"}

READING_COLS = ["ts", "source", "temperature", "humidity", "pressure", "wind_speed",
                "wind_gust", "cloud_cover", "precipitation"]
PRED_COLS = ["made_at", "target_time", "predicted_temp", "baseline_temp", "frozen_temp",
             "actual_temp", "error", "baseline_error", "frozen_error", "resolved_at"]
AIR_COLS = ["ts", "pm2_5", "pm10", "us_aqi", "ozone", "nitrogen_dioxide"]
HIST_COLS = ["ts", "temperature", "wind_speed", "wind_gust", "precipitation", "humidity"]


# ---------------- Data loading ----------------
_client = None
_cache = {}


def client():
    global _client
    if _client is None:
        url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("Missing settings. In the Space, open Settings → Variables and secrets "
                               "and add SUPABASE_URL and SUPABASE_KEY (the publishable key).")
        _client = create_client(url, key)
    return _client


def fetch_all(table, columns, time_col, since_iso):
    """Supabase returns at most 1,000 rows per request, so page through."""
    rows, start, page = [], 0, 1000
    while True:
        q = client().table(table).select(",".join(columns))
        if since_iso:
            q = q.gte(time_col, since_iso)
        batch = q.order(time_col).range(start, start + page - 1).execute().data
        rows.extend(batch)
        if len(batch) < page:
            return rows
        start += page


def to_frame(rows, columns, time_cols, numeric_cols):
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    for c in columns:
        if c not in df:
            df[c] = None
    for c in time_cols:
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[columns]


def prep_readings(rows):
    df = to_frame(rows, READING_COLS, ["ts"], READING_COLS[2:])
    df["source"] = df["source"].map(lambda s: SOURCE_LABELS.get(s, s))
    return df.sort_values("ts").reset_index(drop=True)


def prep_predictions(rows):
    df = to_frame(rows, PRED_COLS, ["made_at", "target_time", "resolved_at"],
                  ["predicted_temp", "baseline_temp", "frozen_temp", "actual_temp",
                   "error", "baseline_error", "frozen_error"])
    return df.sort_values("made_at").reset_index(drop=True)


def prep_air(rows):
    return to_frame(rows, AIR_COLS, ["ts"], AIR_COLS[1:]).sort_values("ts").reset_index(drop=True)


def prep_history(rows):
    return to_frame(rows, HIST_COLS, ["ts"], HIST_COLS[1:]).sort_values("ts").reset_index(drop=True)


def load_data(hours):
    hit = _cache.get(hours)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    since = None
    if hours:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = {
        "readings": prep_readings(fetch_all("readings", READING_COLS, "ts", since)),
        "predictions": prep_predictions(fetch_all("predictions", PRED_COLS, "made_at", since)),
        "air": prep_air(fetch_all("air_quality", AIR_COLS, "ts", since)),
        "history": prep_history(fetch_all("history", HIST_COLS, "ts", None)),
    }
    _cache[hours] = (time.time(), data)
    return data


# ---------------- Calculations ----------------
def local(series):
    """UTC timestamps -> Vancouver wall-clock time (no timezone tag, so charts show it as is)."""
    return series.dt.tz_convert(TZ).dt.tz_localize(None)


def fmt_time(ts):
    return ts.tz_convert(TZ).strftime("%b %d, %I:%M %p").replace(" 0", " ")


def feels_like(temp, wind_ms):
    """Environment Canada wind chill (applies at or below 10 °C with some wind)."""
    if temp is None or pd.isna(temp):
        return None
    v = 0.0 if wind_ms is None or pd.isna(wind_ms) else wind_ms * 3.6
    if temp <= 10 and v >= 4.8:
        return 13.12 + 0.6215 * temp - 11.37 * v ** 0.16 + 0.3965 * temp * v ** 0.16
    return float(temp)


def first_valid(*values):
    for v in values:
        if v is not None and not pd.isna(v):
            return float(v)
    return None


def current_conditions(data):
    r, p, a = data["readings"], data["predictions"], data["air"]
    latest = r.loc[r.groupby("source")["ts"].idxmax()]
    temp = latest["temperature"].mean()
    wind = latest["wind_speed"].max()
    gust = first_valid(latest["wind_gust"].max(), wind)
    rain = first_valid(latest["precipitation"].max(), 0.0)
    nxt = None
    if not p.empty and pd.isna(p.iloc[-1]["actual_temp"]):
        nxt = p.iloc[-1]
    aq = a.iloc[-1] if not a.empty else None
    return {
        "ts": r["ts"].max(), "temp": temp, "wind": wind, "gust": gust, "rain": rain,
        "humidity": latest["humidity"].mean(), "feels": feels_like(temp, wind),
        "predicted": None if nxt is None else nxt["predicted_temp"],
        "target": None if nxt is None else nxt["target_time"],
        "aqi": None if aq is None else first_valid(aq["us_aqi"]),
        "pm25": None if aq is None else first_valid(aq["pm2_5"]),
        "sources": sorted(latest["source"].unique()),
    }


def assess(c, lim):
    """Turn current + predicted conditions into the five site decisions."""
    out = []
    t_now, t_pred = c["temp"], c["predicted"]
    worst = min(t_now, t_pred) if t_pred is not None else t_now
    temp_text = f"{t_now:.1f} °C now" + (f" → {t_pred:.1f} °C in 10 min" if t_pred is not None else "")
    if worst < lim["concrete"]:
        s, act = "stop", "Use heated enclosures or insulated blankets and warmed materials, or reschedule the pour."
    elif worst < lim["concrete"] + 1:
        s, act = "watch", "Close to the limit: have blankets and heaters ready before pouring."
    else:
        s, act = "ok", "Temperature is suitable for concrete pours."
    out.append({"decision": "Concrete pour", "status": s, "now": temp_text,
                "limit": f"min {lim['concrete']:.1f} °C", "action": act})

    g = c["gust"]
    if g >= lim["crane"]:
        s, act = "stop", "Pause crane lifts. Secure loads, formwork and loose materials."
    elif g >= 0.8 * lim["crane"]:
        s, act = "watch", "Gusts approaching the limit: brief the operator and plan short lifts."
    else:
        s, act = "ok", "Wind is within the crane limit."
    out.append({"decision": "Crane lifts", "status": s, "now": f"gusts {g:.1f} m/s",
                "limit": f"max {lim['crane']:.1f} m/s", "action": act})

    rain, hum = c["rain"], c["humidity"]
    if rain >= lim["rain"]:
        s, act = "stop", "Move crews to indoor tasks. Cover materials and open excavations."
    elif rain > 0 or (hum is not None and hum >= 90):
        s, act = "watch", "Light rain or very humid air: coatings and sealants may not cure; check product limits."
    else:
        s, act = "ok", "Dry: exterior finishing, roofing and painting can go ahead."
    out.append({"decision": "Roofing, painting & exterior finishes", "status": s,
                "now": f"rain {rain:.1f} mm, humidity {hum:.0f}%",
                "limit": f"max {lim['rain']:.1f} mm", "action": act})

    f = c["feels"]
    if f <= lim["cold"]:
        s, act = "stop", "Schedule warm-up breaks, rotate crews and provide a heated shelter."
    elif f <= lim["cold"] + 3:
        s, act = "watch", "Cold and windy: remind crews to layer up and check on workers more often."
    else:
        s, act = "ok", "No cold-stress measures needed."
    out.append({"decision": "Crew cold stress", "status": s, "now": f"feels like {f:.1f} °C",
                "limit": f"at or below {lim['cold']:.1f} °C", "action": act})

    aqi = c["aqi"]
    if aqi is None:
        s, act, now = "unknown", "Air-quality data unavailable right now.", "no data"
    else:
        now = f"US AQI {aqi:.0f}" + (f", PM2.5 {c['pm25']:.1f} µg/m³" if c["pm25"] is not None else "")
        if aqi >= lim["aqi"]:
            s, act = "stop", "Reduce strenuous outdoor work, provide N95 respirators, move work indoors where possible."
        elif aqi >= 0.75 * lim["aqi"]:
            s, act = "watch", "Air quality getting worse: watch for smoke and give workers with asthma extra breaks."
        else:
            s, act = "ok", "Air quality is fine for outdoor work."
    out.append({"decision": "Heavy outdoor work (air quality)", "status": s, "now": now,
                "limit": f"max AQI {lim['aqi']:.0f}", "action": act})
    return out


def history_risk(history, lim):
    """How many working hours in the past 30 days (ERA5) would each limit have stopped work?"""
    if history.empty:
        return None
    h = history.copy()
    h["local"] = local(h["ts"])
    h = h[h["local"].dt.hour.isin(list(WORK_HOURS))]
    if h.empty:
        return None
    gust = h["wind_gust"].fillna(h["wind_speed"])
    feels = [feels_like(t, w) for t, w in zip(h["temperature"], h["wind_speed"])]
    flags = pd.DataFrame({
        "Concrete pour": h["temperature"] < lim["concrete"],
        "Crane lifts": gust >= lim["crane"],
        "Exterior finishes": h["precipitation"].fillna(0) >= lim["rain"],
        "Crew cold stress": pd.Series([f is not None and f <= lim["cold"] for f in feels], index=h.index),
    })
    return {"hours": len(h), "start": h["local"].min(), "end": h["local"].max(),
            "counts": flags.sum().astype(int).to_dict(), "any": int(flags.any(axis=1).sum())}


def accuracy(preds):
    res = preds[preds["actual_temp"].notna()].sort_values("resolved_at")
    if res.empty:
        return None
    k = max(1, min(50, len(res) // 2))
    cols = {"Self-learning model": "error", "Frozen model (never learns)": "frozen_error",
            "No-change guess": "baseline_error"}
    table = {name: {"all": res[c].mean(), "early": res[c].head(k).mean(), "recent": res[c].tail(k).mean()}
             for name, c in cols.items()}
    both = res.dropna(subset=["error", "frozen_error"])
    vs_frozen = None
    if not both.empty and both["frozen_error"].mean() > 0:
        vs_frozen = (both["frozen_error"].mean() - both["error"].mean()) / both["frozen_error"].mean() * 100
    return {"n": len(res), "k": k, "table": table, "vs_frozen": vs_frozen,
            "n_both": len(both), "last": res.iloc[-1]}


def rolling_errors(preds, window=ROLL):
    res = preds[preds["actual_temp"].notna()].sort_values("resolved_at").copy()
    for c in ["error", "frozen_error", "baseline_error"]:
        res[c + "_roll"] = res[c].rolling(window, min_periods=3).mean()
    return res.dropna(subset=["error_roll"])


# ---------------- HTML pieces ----------------
CARD = ("border:1px solid var(--border-color-primary);border-radius:12px;padding:12px 16px;"
        "background:var(--block-background-fill);")
STATUS = {
    "stop": ("🔴", "Act now", "rgba(227,73,72,0.14)", "#e34948"),
    "watch": ("🟠", "Watch", "rgba(237,161,0,0.16)", "#eda100"),
    "ok": ("🟢", "OK", "rgba(0,131,0,0.10)", "#008300"),
    "unknown": ("⚪", "No data", "rgba(128,128,128,0.10)", "#8a8985"),
}


def header_html(c=None, error=None):
    title = (f"<div style='font-size:1.7em;font-weight:700;margin-bottom:2px'>🏗️ Site Weather Risk Monitor · "
             f"{LOCATION}</div><div style='opacity:.85;margin-bottom:6px'>For <b>construction site managers</b>: "
             "live weather, air quality and a self-correcting 10-minute forecast for safe go/no-go "
             "decisions on site.</div>")
    if error:
        return title + f"<div style='{CARD}border-color:#e34948'>⚠️ {error}</div>"
    age = (datetime.now(timezone.utc) - c["ts"].to_pydatetime()).total_seconds() / 60
    return title + (f"<div style='opacity:.7;font-size:.9em'>Last reading {fmt_time(c['ts'])} "
                    f"({age:.0f} min ago) · Sources: {', '.join(c['sources'])}, Open-Meteo Air Quality, "
                    f"ERA5 public dataset · Times in Vancouver time · Refreshes every {REFRESH_SECONDS} s</div>")


def banner_html(items, c):
    stops = [i["decision"] for i in items if i["status"] == "stop"]
    watches = [i["decision"] for i in items if i["status"] == "watch"]
    if stops:
        icon, bg, border = "🔴", "rgba(227,73,72,0.14)", "#e34948"
        text = f"<b>{len(stops)} risk{'s' if len(stops) > 1 else ''} need action now:</b> {', '.join(stops)}"
    elif watches:
        icon, bg, border = "🟠", "rgba(237,161,0,0.16)", "#eda100"
        text = f"<b>Watch closely:</b> {', '.join(watches)}"
    else:
        icon, bg, border = "🟢", "rgba(0,131,0,0.10)", "#008300"
        text = "<b>Good working conditions right now.</b> No site limits are exceeded."
    age = (datetime.now(timezone.utc) - c["ts"].to_pydatetime()).total_seconds() / 60
    stale = (f"<div style='margin-top:6px'>⚠️ No new data for {age:.0f} minutes; the collector may be "
             "delayed.</div>") if age > STALE_MINUTES else ""
    return (f"<div style='border:2px solid {border};background:{bg};border-radius:12px;padding:12px 16px;"
            f"font-size:1.1em'>{icon} {text}{stale}</div>")


def kpi_html(c):
    def card(label, value, sub=""):
        return (f"<div style='{CARD}flex:1;min-width:160px'><div style='opacity:.7;font-size:.85em'>{label}</div>"
                f"<div style='font-size:1.6em;font-weight:600'>{value}</div>"
                f"<div style='opacity:.75;font-size:.85em'>{sub}</div></div>")
    pred = (f"{c['predicted']:.1f} °C at {fmt_time(c['target']).split(', ')[1]}"
            if c["predicted"] is not None else "waiting for next run")
    aqi = "–" if c["aqi"] is None else f"{c['aqi']:.0f}"
    pm = "" if c["pm25"] is None else f"PM2.5 {c['pm25']:.1f} µg/m³"
    return ("<div style='display:flex;gap:12px;flex-wrap:wrap'>"
            + card("Temperature now", f"{c['temp']:.1f} °C", f"Predicted: {pred}")
            + card("Wind gusts", f"{c['gust']:.1f} m/s", f"Average wind {c['wind']:.1f} m/s")
            + card("Rain (latest)", f"{c['rain']:.1f} mm", f"Humidity {c['humidity']:.0f}%")
            + card("Feels like", f"{c['feels']:.1f} °C", "Wind chill")
            + card("Air quality (US AQI)", aqi, pm)
            + "</div>")


def actions_html(items):
    rows = ""
    for i in items:
        icon, label, bg, border = STATUS[i["status"]]
        rows += (f"<tr style='border-top:1px solid var(--border-color-primary)'>"
                 f"<td style='padding:10px;white-space:nowrap'><span style='background:{bg};border:1px solid "
                 f"{border};border-radius:999px;padding:2px 10px'>{icon} {label}</span></td>"
                 f"<td style='padding:10px'><b>{i['decision']}</b></td>"
                 f"<td style='padding:10px'>{i['now']}</td>"
                 f"<td style='padding:10px;opacity:.8'>{i['limit']}</td>"
                 f"<td style='padding:10px'>{i['action']}</td></tr>")
    return (f"<div style='{CARD}padding:4px 8px'><div style='font-weight:700;font-size:1.15em;padding:8px'>"
            "Alerts & recommended actions</div><div style='overflow-x:auto'><table style='width:100%;"
            "border-collapse:collapse'><tr style='text-align:left;opacity:.7;font-size:.85em'>"
            "<th style='padding:6px 10px'>Status</th><th style='padding:6px 10px'>Decision</th>"
            "<th style='padding:6px 10px'>Right now</th><th style='padding:6px 10px'>Your limit</th>"
            f"<th style='padding:6px 10px'>Recommended action</th></tr>{rows}</table></div>"
            "<div style='opacity:.7;font-size:.85em;padding:8px'>Temperature checks use the lower of the "
            "current and the predicted value, so you're warned before the limit is crossed. "
            "Adjust the limits under “Your site limits”.</div></div>")


def history_html(hr):
    if hr is None:
        return f"<div style='{CARD}'>The 30-day public dataset hasn't loaded yet.</div>"
    def pct(n):
        return f"{n} h ({n / hr['hours'] * 100:.0f}%)"
    items = "".join(f"<li><b>{k}:</b> {pct(v)}</li>" for k, v in hr["counts"].items())
    return (f"<div style='{CARD}'><div style='opacity:.75;font-size:.9em'>ERA5 reanalysis, "
            f"{hr['start']:%b %d} to {hr['end']:%b %d}, working hours 7 AM–5 PM: {hr['hours']} hours</div>"
            f"<div style='font-size:1.15em;margin:6px 0'><b>{pct(hr['any'])}</b> of working hours had at "
            "least one weather limit exceeded with your current settings.</div>"
            f"<ul style='margin:4px 0 4px 18px'>{items}</ul>"
            "<div style='opacity:.75;font-size:.9em'>Use this to plan schedules and budgets: these are paid "
            "crew hours that need a backup plan (indoor tasks, heating, covers) to avoid going idle.</div></div>")


def accuracy_html(acc):
    if acc is None:
        return f"<div style='{CARD}'>No predictions have been checked in this date range yet.</div>"
    t, last = acc["table"], acc["last"]

    def f(v):
        return "–" if v is None or pd.isna(v) else f"{v:.3f} °C"
    rows = "".join(f"<tr style='border-top:1px solid var(--border-color-primary)'><td style='padding:8px'>{name}"
                   f"</td><td style='padding:8px'>{f(v['all'])}</td><td style='padding:8px'>{f(v['early'])}</td>"
                   f"<td style='padding:8px'>{f(v['recent'])}</td></tr>" for name, v in t.items())
    if acc["vs_frozen"] is None:
        verdict = "The frozen-model comparison appears once its predictions have been checked."
    else:
        word = "lower" if acc["vs_frozen"] >= 0 else "higher"
        verdict = (f"Over {acc['n_both']:,} checked predictions, the self-learning model's average error is "
                   f"<b>{abs(acc['vs_frozen']):.0f}% {word}</b> than the frozen model that never learns.")
    frozen_last = ("" if pd.isna(last["frozen_temp"]) else
                   f" · frozen model said {last['frozen_temp']:.2f} °C (error {last['frozen_error']:.2f})")
    return (f"<div style='display:flex;gap:12px;flex-wrap:wrap'>"
            f"<div style='{CARD}flex:1;min-width:260px'><div style='opacity:.7;font-size:.85em'>Last check "
            f"({fmt_time(last['resolved_at'])})</div><div style='font-size:1.1em;margin-top:4px'>Predicted "
            f"<b>{last['predicted_temp']:.2f} °C</b> → actual <b>{last['actual_temp']:.2f} °C</b> → error "
            f"<b>{last['error']:.2f} °C</b> → model updated ✓</div><div style='opacity:.75;font-size:.85em'>"
            f"No-change guess error {last['baseline_error']:.2f}{frozen_last}</div></div>"
            f"<div style='{CARD}flex:1;min-width:260px'><div style='opacity:.7;font-size:.85em'>Predictions "
            f"checked in this range</div><div style='font-size:1.6em;font-weight:600'>{acc['n']:,}</div>"
            f"<div style='opacity:.85'>{verdict}</div></div></div>"
            f"<div style='{CARD}margin-top:12px'><div style='font-weight:700;margin-bottom:4px'>Average error "
            "(lower is better)</div><table style='width:100%;border-collapse:collapse'><tr style='text-align:"
            "left;opacity:.7;font-size:.85em'><th style='padding:6px 8px'>Model</th><th style='padding:6px 8px'>"
            f"All checks</th><th style='padding:6px 8px'>First {acc['k']} checks</th><th style='padding:6px 8px'>"
            f"Latest {acc['k']} checks</th></tr>{rows}</table></div>")


# ---------------- Charts ----------------
def style(fig, y_title, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#8a8985"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                  title_text=""),
                      yaxis_title=y_title, xaxis_title=None)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


def empty_fig(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                       font=dict(size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, "", 220)


def limit_for(column, lim):
    return {"temperature": ("Concrete limit", lim["concrete"]),
            "wind_gust": ("Crane limit", lim["crane"]),
            "wind_speed": ("Crane limit", lim["crane"]),
            "precipitation": ("Rain limit", lim["rain"]),
            "us_aqi": ("Air-quality limit", lim["aqi"])}.get(column)


def add_limit(fig, column, lim):
    lv = limit_for(column, lim)
    if lv:
        fig.add_hline(y=lv[1], line=dict(color=COLORS["limit"], width=1.5, dash="dash"),
                      annotation_text=f"{lv[0]} {lv[1]:g}", annotation_position="top left")


def explore_fig(data, sources, var_label, lim):
    table, col, unit = VARIABLES[var_label]
    fig = go.Figure()
    if table == "air_quality":
        a = data["air"].dropna(subset=[col])
        if a.empty:
            return empty_fig("No air-quality data in this date range yet")
        fig.add_trace(go.Scatter(x=local(a["ts"]), y=a[col], name="Open-Meteo Air Quality", mode="lines",
                                 line=dict(color=COLORS["Air quality"], width=2),
                                 hovertemplate=f"%{{y:.1f}} {unit}"))
    else:
        if not sources:
            return empty_fig("Select at least one weather source")
        r = data["readings"]
        r = r[r["source"].isin(sources)].dropna(subset=[col])
        if r.empty:
            return empty_fig(f"No {var_label.lower()} data for this selection yet")
        for src in SOURCES:
            g = r[r["source"] == src]
            if not g.empty:
                fig.add_trace(go.Scatter(x=local(g["ts"]), y=g[col], name=src, mode="lines",
                                         line=dict(color=COLORS[src], width=2),
                                         hovertemplate=f"%{{y:.1f}} {unit}"))
    add_limit(fig, col, lim)
    return style(fig, unit)


def explore_note(data, sources, var_label):
    table, col, unit = VARIABLES[var_label]
    if table == "air_quality":
        groups = [("Open-Meteo Air Quality", data["air"][col].dropna())]
    else:
        r = data["readings"]
        groups = [(s, r.loc[r["source"] == s, col].dropna()) for s in SOURCES if s in (sources or [])]
    lines = [f"- **{name}:** min {s.min():.1f}, average {s.mean():.1f}, max {s.max():.1f} {unit} "
             f"({len(s):,} readings)" for name, s in groups if not s.empty]
    return "\n".join(lines) if lines else "_No data for this selection._"


def history_fig(history, var_label, lim):
    table, col, unit = VARIABLES[var_label]
    if table == "air_quality" or col not in history or history[col].dropna().empty:
        return empty_fig("The ERA5 public dataset covers temperature, wind, rain and humidity. "
                         "Pick one of those to see the 30-day history.")
    h = history.dropna(subset=[col])
    fig = go.Figure(go.Scatter(x=local(h["ts"]), y=h[col], name="ERA5 hourly", mode="lines",
                               line=dict(color=COLORS["ERA5"], width=1.5),
                               hovertemplate=f"%{{y:.1f}} {unit}"))
    add_limit(fig, col, lim)
    fig = style(fig, unit, 300)
    fig.update_layout(showlegend=False)
    return fig


def prediction_fig(preds, c):
    res = preds[preds["actual_temp"].notna()]
    if res.empty:
        return empty_fig("Predicted vs. actual appears after the first prediction is checked")
    x = local(res["target_time"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=res["actual_temp"], name="Actual", mode="lines",
                             line=dict(color=COLORS["actual"], width=2.5), hovertemplate="%{y:.2f} °C"))
    fig.add_trace(go.Scatter(x=x, y=res["predicted_temp"], name="Self-learning model (predicted 10 min earlier)",
                             mode="lines", line=dict(color=COLORS["learning"], width=2, dash="dot"),
                             hovertemplate="%{y:.2f} °C"))
    fr = res.dropna(subset=["frozen_temp"])
    if not fr.empty:
        fig.add_trace(go.Scatter(x=local(fr["target_time"]), y=fr["frozen_temp"], name="Frozen model",
                                 mode="lines", line=dict(color=COLORS["frozen"], width=1.5, dash="dot"),
                                 hovertemplate="%{y:.2f} °C"))
    if c["predicted"] is not None:
        t0 = c["ts"].tz_convert(TZ).tz_localize(None)
        t1 = c["target"].tz_convert(TZ).tz_localize(None)
        fig.add_trace(go.Scatter(x=[t0, t1], y=[c["temp"], c["predicted"]], name="Next prediction",
                                 mode="lines+markers", line=dict(color=COLORS["learning"], width=2, dash="dot"),
                                 marker=dict(size=[0, 11], color=COLORS["learning"]),
                                 hovertemplate="%{y:.2f} °C"))
    return style(fig, "°C", 360)


def error_fig(preds):
    roll = rolling_errors(preds)
    if roll.empty:
        return empty_fig("The error trend appears after 3 predictions have been checked (about 30 minutes)")
    x = local(roll["resolved_at"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=roll["baseline_error_roll"], name="No-change guess", mode="lines",
                             line=dict(color=COLORS["baseline"], width=2, dash="dash"),
                             hovertemplate="%{y:.3f} °C"))
    if roll["frozen_error_roll"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=roll["frozen_error_roll"], name="Frozen model (never learns)",
                                 mode="lines", line=dict(color=COLORS["frozen"], width=2),
                                 hovertemplate="%{y:.3f} °C"))
    fig.add_trace(go.Scatter(x=x, y=roll["error_roll"], name="Self-learning model", mode="lines",
                             line=dict(color=COLORS["learning"], width=2.5), hovertemplate="%{y:.3f} °C"))
    fig = style(fig, f"Avg error, last {ROLL} checks (°C)")
    fig.update_yaxes(rangemode="tozero")
    return fig


# ---------------- Refresh (runs on load, every minute, and on every control change) ----------------
def refresh(window, sources, variable, concrete, crane, rain, cold, aqi):
    lim = {"concrete": concrete, "crane": crane, "rain": rain, "cold": cold, "aqi": aqi}
    try:
        data = load_data(WINDOWS.get(window, 24))
        if data["readings"].empty:
            raise RuntimeError("No readings in this date range yet. Pick a longer range, or check that "
                               "the collector is running on GitHub Actions.")
        c = current_conditions(data)
        items = assess(c, lim)
        acc = accuracy(data["predictions"])
        recent = data["readings"][data["readings"]["source"].isin(sources or [])].tail(60).iloc[::-1]
        table = pd.DataFrame({
            "Time": local(recent["ts"]).dt.strftime("%b %d %I:%M %p"), "Source": recent["source"],
            "Temp (°C)": recent["temperature"].round(1), "Wind (m/s)": recent["wind_speed"].round(1),
            "Gust (m/s)": recent["wind_gust"].round(1), "Rain (mm)": recent["precipitation"].round(1),
            "Humidity (%)": recent["humidity"].round(0)})
        return (header_html(c), banner_html(items, c), kpi_html(c), actions_html(items),
                explore_fig(data, sources, variable, lim), explore_note(data, sources, variable),
                history_html(history_risk(data["history"], lim)), history_fig(data["history"], variable, lim),
                table, accuracy_html(acc), prediction_fig(data["predictions"], c), error_fig(data["predictions"]))
    except Exception as e:
        blank = empty_fig("No data to show")
        return (header_html(error=str(e)), "", "", "", blank, "", "", blank,
                pd.DataFrame(), "", blank, blank)


PROBLEM_MD = """
**The problem.** On a construction site, weather decides what work is safe and what work will
last. Concrete poured below about 5 °C can freeze before it gains strength. Crane lifts must stop
when gusts exceed the crane's rated limit. Rain ruins roofing, paint and sealants. Cold wind
causes cold stress, and wildfire smoke makes heavy outdoor work unsafe. Calling these too late
means rework, safety incidents and paid crews standing idle.

**How this dashboard helps.** It combines three live sources with a 30-day public dataset, warns
you **10 minutes before** a temperature limit is crossed, and turns every reading into a clear
recommended action, using limits you can set to match your equipment and specs.
"""

ABOUT_MD = f"""
### Who it's for
**Construction site managers in {LOCATION}** making go/no-go calls during the work day.

### Decisions it supports
| Decision | Data used | Default limit (adjustable) |
|---|---|---|
| Pour concrete now? | Temperature + 10-minute prediction | {DEFAULT_LIMITS['concrete']:g} °C |
| Run crane lifts? | Wind gusts | {DEFAULT_LIMITS['crane']:g} m/s (use your crane's rated limit) |
| Roofing, painting, exterior finishes? | Rain, humidity | {DEFAULT_LIMITS['rain']:g} mm |
| Cold-stress breaks for crews? | Feels-like temperature (wind chill) | {DEFAULT_LIMITS['cold']:g} °C |
| Limit heavy outdoor work? | US Air Quality Index | {DEFAULT_LIMITS['aqi']:g} |

Defaults are starting points. Always follow your equipment manuals, product data sheets and
WorkSafeBC requirements.

### Data sources (all stored in one Supabase database)
- **OpenWeatherMap** (live API): temperature, wind, gusts, rain, humidity, pressure, cloud
- **Open-Meteo** (live API): the same variables from an independent forecast model
- **Open-Meteo Air Quality** (live API, CAMS model): PM2.5, PM10, US AQI, ozone, NO₂
- **ERA5 reanalysis** (public dataset, Copernicus/ECMWF): 30 days of hourly history

### How the forecast corrects itself
Every 10 minutes a collector on GitHub Actions saves new readings, checks the temperature
prediction it made 10 minutes earlier, learns from the error (online linear regression with
the `river` library), and predicts the next 10 minutes. A **frozen** copy of the model was
trained the same way but never learns; comparing the two shows how much the live learning helps.
A simple **no-change guess** (temperature stays the same) is the minimum bar to beat.
"""


# ---------------- Layout ----------------
with gr.Blocks() as demo:
    demo.title = "Site Weather Risk Monitor"
    header = gr.HTML()
    banner = gr.HTML()
    with gr.Row():
        window = gr.Dropdown(list(WINDOWS), value="Last 24 hours", label="Date range")
        sources = gr.CheckboxGroup(SOURCES, value=SOURCES, label="Weather sources")
        variable = gr.Dropdown(list(VARIABLES), value="Temperature (°C)", label="Variable to explore")
    with gr.Accordion("Your site limits (set these to match your equipment and specs)", open=False):
        with gr.Row():
            concrete = gr.Slider(0, 10, value=DEFAULT_LIMITS["concrete"], step=0.5,
                                 label="Concrete pour: minimum air temperature (°C)")
            crane = gr.Slider(5, 25, value=DEFAULT_LIMITS["crane"], step=0.5,
                              label="Crane lifts: maximum wind gust (m/s)")
            rain = gr.Slider(0.1, 5, value=DEFAULT_LIMITS["rain"], step=0.1,
                             label="Exterior finishes: maximum rain (mm)")
        with gr.Row():
            cold = gr.Slider(-15, 5, value=DEFAULT_LIMITS["cold"], step=0.5,
                             label="Crew cold stress: feels-like temperature (°C)")
            aqi = gr.Slider(50, 200, value=DEFAULT_LIMITS["aqi"], step=5,
                            label="Heavy outdoor work: maximum US AQI")
    with gr.Tabs():
        with gr.Tab("Site conditions & actions"):
            kpis = gr.HTML()
            actions = gr.HTML()
            with gr.Accordion("The problem this dashboard solves", open=True):
                gr.Markdown(PROBLEM_MD)
        with gr.Tab("Explore the data"):
            gr.Markdown("Use **Date range**, **Weather sources** and **Variable to explore** above. "
                        "Dashed red lines show your site limits.")
            explore_plot = gr.Plot(label="Live readings")
            explore_text = gr.Markdown()
            gr.Markdown("### Past 30 days at this site (ERA5 public dataset)")
            hist_summary = gr.HTML()
            hist_plot = gr.Plot(label="ERA5 hourly history")
            readings_table = gr.Dataframe(label="Latest readings", interactive=False)
        with gr.Tab("Forecast accuracy & learning"):
            gr.Markdown("Every 10 minutes the model predicts the temperature, then checks itself against the "
                        "real value and learns from the error. **Lower error is better.** When the blue line "
                        "sits below the purple one, learning is beating the model that never learns.")
            acc_summary = gr.HTML()
            pred_plot = gr.Plot(label="Predicted vs. actual temperature")
            err_plot = gr.Plot(label="Is the model learning? Rolling average error")
        with gr.Tab("About"):
            gr.Markdown(ABOUT_MD)

    inputs = [window, sources, variable, concrete, crane, rain, cold, aqi]
    outputs = [header, banner, kpis, actions, explore_plot, explore_text, hist_summary, hist_plot,
               readings_table, acc_summary, pred_plot, err_plot]
    demo.load(refresh, inputs, outputs)
    timer = gr.Timer(REFRESH_SECONDS)
    timer.tick(refresh, inputs, outputs, show_progress="hidden")
    for control in (window, sources, variable):
        control.change(refresh, inputs, outputs)
    for slider in (concrete, crane, rain, cold, aqi):
        slider.release(refresh, inputs, outputs)


if __name__ == "__main__":
    # Render (and most hosts) tell the app which port to use through $PORT
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
