"""
app.py: Live Weather Forecast dashboard (Day 2)

Reads the data the collector saves in Supabase every ~10 minutes and shows:
  - live readings from each weather source
  - the model's prediction for the next 10 minutes
  - predicted vs. actual temperature
  - the error trend, which is the proof that the model is learning

The page refreshes itself every minute, so new data appears automatically.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client

# ---------------- Settings ----------------
REFRESH_SECONDS = 60     # how often the page reloads data
ROLL_WINDOW = 12         # rolling error over the last 12 checked predictions (about 2 hours)
STALE_MINUTES = 30       # warn if no new data arrives for this long
HORIZON_MIN = 10         # the collector predicts this many minutes ahead

WINDOWS = {"Last 6 hours": 6, "Last 24 hours": 24, "Last 3 days": 72, "All data": None}
SOURCE_LABELS = {"openweathermap": "OpenWeatherMap", "open-meteo": "Open-Meteo"}

# Colors follow the entity, never its position: OpenWeatherMap is always blue,
# Open-Meteo is always orange. Validated colorblind-safe pair, light and dark steps.
PALETTE = {
    "light": {"s1": "#2a78d6", "s2": "#eb6834", "neutral": "#7a7974",
              "grid": "rgba(0,0,0,0.08)", "surface": "#ffffff"},
    "dark": {"s1": "#3987e5", "s2": "#d95926", "neutral": "#9a998f",
             "grid": "rgba(255,255,255,0.10)", "surface": "#0e1117"},
}
SOURCE_SLOT = {"OpenWeatherMap": "s1", "Open-Meteo": "s2"}

READING_COLS = ["ts", "source", "temperature", "humidity", "pressure", "wind_speed", "cloud_cover"]
PRED_COLS = ["made_at", "target_time", "predicted_temp", "baseline_temp",
             "actual_temp", "error", "baseline_error", "resolved_at"]


# ---------------- Helpers ----------------
def secret(name, default=None):
    """Read a value from Streamlit secrets without crashing if it's missing."""
    try:
        return st.secrets[name]
    except Exception:
        return default


def theme_colors():
    try:
        kind = st.context.theme.type or "light"
    except Exception:
        kind = "light"
    return PALETTE.get(kind, PALETTE["light"])


def fmt_time(ts, tz):
    return ts.tz_convert(tz).strftime("%b %d, %I:%M %p").replace(" 0", " ")


# ---------------- Data loading ----------------
@st.cache_resource
def get_client(url, key):
    return create_client(url, key)


def fetch_all(client, table, columns, time_col, since_iso):
    """Fetch every row (Supabase returns at most 1,000 per request, so page through)."""
    rows, start, page = [], 0, 1000
    while True:
        q = client.table(table).select(columns)
        if since_iso:
            q = q.gte(time_col, since_iso)
        batch = q.order(time_col).range(start, start + page - 1).execute().data
        rows.extend(batch)
        if len(batch) < page:
            return rows
        start += page


def prepare_readings(rows):
    df = pd.DataFrame(rows, columns=READING_COLS) if not rows else pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=READING_COLS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df["source"] = df["source"].map(SOURCE_LABELS).fillna(df["source"])
    for c in READING_COLS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)


def prepare_predictions(rows):
    df = pd.DataFrame(rows, columns=PRED_COLS) if not rows else pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=PRED_COLS)
    for c in ["made_at", "target_time", "resolved_at"]:
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    for c in ["predicted_temp", "baseline_temp", "actual_temp", "error", "baseline_error"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("made_at").reset_index(drop=True)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_data(url, key, hours):
    client = get_client(url, key)
    since = None
    if hours:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    readings = fetch_all(client, "readings", ",".join(READING_COLS), "ts", since)
    preds = fetch_all(client, "predictions", ",".join(PRED_COLS), "made_at", since)
    return prepare_readings(readings), prepare_predictions(preds)


# ---------------- Calculations ----------------
def latest_snapshot(readings):
    """Average temperature across sources for the latest run and the run before it."""
    runs = readings.groupby("ts")["temperature"].mean().sort_index()
    prev = runs.iloc[-2] if len(runs) > 1 else None
    return runs.index[-1], runs.iloc[-1], prev


def latest_by_source(readings):
    idx = readings.groupby("source")["ts"].idxmax()
    return readings.loc[idx].sort_values("source")


def split_predictions(preds):
    if preds.empty:
        return preds, None
    resolved = preds[preds["actual_temp"].notna()].sort_values("resolved_at")
    newest = preds.iloc[-1]
    nxt = newest if pd.isna(newest["actual_temp"]) else None
    return resolved, nxt


def error_summary(resolved):
    if resolved.empty:
        return None
    model = resolved["error"].mean()
    base = resolved["baseline_error"].mean()
    improvement = (base - model) / base * 100 if base > 0 else None
    return {"model": model, "base": base, "improvement": improvement, "n": len(resolved)}


def rolling_errors(resolved, window=ROLL_WINDOW):
    r = resolved[["resolved_at", "error", "baseline_error"]].copy()
    r["model_mae"] = r["error"].rolling(window, min_periods=3).mean()
    r["base_mae"] = r["baseline_error"].rolling(window, min_periods=3).mean()
    return r.dropna(subset=["model_mae"])


# ---------------- Charts ----------------
def style(fig, colors, height=340, y_title="°C"):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        yaxis_title=y_title,
        xaxis_title=None,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=colors["grid"], zeroline=False)
    return fig


def chart_by_source(readings, field, unit, colors, height=320, decimals=1):
    fig = go.Figure()
    for src, g in readings.groupby("source"):
        fig.add_trace(go.Scatter(
            x=g["ts_local"], y=g[field], name=src, mode="lines",
            line=dict(color=colors[SOURCE_SLOT.get(src, "neutral")], width=2),
            hovertemplate=f"%{{y:.{decimals}f}} {unit}<extra>{src}</extra>",
        ))
    return style(fig, colors, height, unit)


def chart_pred_vs_actual(resolved, nxt, now_ts, now_temp, colors, tz):
    fig = go.Figure()
    x = resolved["target_time"].dt.tz_convert(tz)
    fig.add_trace(go.Scatter(
        x=x, y=resolved["actual_temp"], name="Actual", mode="lines",
        line=dict(color=colors["s1"], width=2),
        hovertemplate="%{y:.2f} °C<extra>Actual</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=resolved["predicted_temp"], name="Predicted 10 min earlier", mode="lines",
        line=dict(color=colors["s2"], width=2, dash="dot"),
        hovertemplate="%{y:.2f} °C<extra>Predicted</extra>",
    ))
    if nxt is not None:
        fig.add_trace(go.Scatter(
            x=[now_ts.tz_convert(tz), nxt["target_time"].tz_convert(tz)],
            y=[now_temp, nxt["predicted_temp"]],
            name="Next prediction", mode="lines+markers",
            line=dict(color=colors["s2"], width=2, dash="dot"),
            marker=dict(size=[0, 11], color=colors["s2"],
                        line=dict(color=colors["surface"], width=2)),
            hovertemplate="%{y:.2f} °C<extra>Next prediction</extra>",
        ))
    return style(fig, colors, 360, "°C")


def chart_error_trend(roll, colors, tz):
    fig = go.Figure()
    x = roll["resolved_at"].dt.tz_convert(tz)
    fig.add_trace(go.Scatter(
        x=x, y=roll["base_mae"], name="No-change baseline", mode="lines",
        line=dict(color=colors["neutral"], width=2, dash="dash"),
        hovertemplate="%{y:.3f} °C<extra>Baseline</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=roll["model_mae"], name="Self-learning model", mode="lines",
        line=dict(color=colors["s1"], width=2),
        hovertemplate="%{y:.3f} °C<extra>Model</extra>",
    ))
    fig = style(fig, colors, 320, "Avg error (°C)")
    fig.update_yaxes(rangemode="tozero")
    return fig


PLOT_CONFIG = {"displayModeBar": False}


# ---------------- Page ----------------
@st.fragment(run_every=REFRESH_SECONDS)
def dashboard(url, key, hours, window_label, tz, location):
    colors = theme_colors()
    st.title(f"🌦️ Live Weather Forecast · {location}")

    try:
        readings, preds = load_data(url, key, hours)
    except Exception as e:
        st.error(f"Couldn't load data from Supabase. Check the app's secrets.\n\nDetails: {e}")
        return

    if readings.empty:
        st.info("No readings in this time window yet. Check that the collector is running "
                "in GitHub Actions, or pick a longer time window in the sidebar.")
        return

    readings["ts_local"] = readings["ts"].dt.tz_convert(tz)
    last_ts, now_temp, prev_temp = latest_snapshot(readings)
    resolved, nxt = split_predictions(preds)
    summary = error_summary(resolved)
    age_min = (datetime.now(timezone.utc) - last_ts.to_pydatetime()).total_seconds() / 60
    sources = ", ".join(sorted(readings["source"].unique()))

    st.caption(f"Sources: {sources} · Collected every ~10 minutes · "
               f"Last reading {fmt_time(last_ts, tz)} ({age_min:.0f} min ago) · "
               f"Times shown in {tz.key}")
    if age_min > STALE_MINUTES:
        st.warning(f"No new data for {age_min:.0f} minutes. Check the Actions tab on GitHub; "
                   "the collector may be delayed or failing.")

    # --- Headline numbers ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Temperature now", f"{now_temp:.1f} °C",
        None if prev_temp is None else f"{now_temp - prev_temp:+.2f} °C since last reading",
        delta_color="off", help="Average of all sources at the latest collection run",
    )
    if nxt is not None:
        c2.metric(
            f"Predicted in {HORIZON_MIN} min", f"{nxt['predicted_temp']:.1f} °C",
            f"{nxt['predicted_temp'] - nxt['baseline_temp']:+.2f} °C expected change",
            delta_color="off", help=f"Forecast for {fmt_time(nxt['target_time'], tz)}",
        )
    else:
        c2.metric(f"Predicted in {HORIZON_MIN} min", "–", help="Waiting for the next collector run")
    if summary:
        c3.metric(
            "Model error (MAE)", f"{summary['model']:.3f} °C",
            f"{summary['model'] - summary['base']:+.3f} °C vs no-change",
            delta_color="inverse",
            help=f"Average absolute error over the {window_label.lower()}. "
                 "Negative (green) means the model beats the simple 'no change' guess.",
        )
        c4.metric("Predictions checked", f"{summary['n']:,}",
                  help="Each checked prediction is one learning step for the model")
    else:
        c3.metric("Model error (MAE)", "–", help="Appears after the first prediction is checked")
        c4.metric("Predictions checked", "0")

    # --- Sources ---
    st.subheader("Live readings from each source")
    st.plotly_chart(chart_by_source(readings, "temperature", "°C", colors),
                    width="stretch", config=PLOT_CONFIG)
    latest = latest_by_source(readings)
    table = pd.DataFrame({
        "Source": latest["source"],
        "Time": latest["ts"].dt.tz_convert(tz).dt.strftime("%I:%M %p"),
        "Temp (°C)": latest["temperature"].round(1),
        "Humidity (%)": latest["humidity"].round(0),
        "Pressure (hPa)": latest["pressure"].round(0),
        "Wind (m/s)": latest["wind_speed"].round(1),
        "Cloud (%)": latest["cloud_cover"].round(0),
    })
    st.dataframe(table, hide_index=True, width="stretch")

    # --- Predictions ---
    st.subheader("Predicted vs. actual temperature")
    if resolved.empty and nxt is None:
        st.info("No predictions yet.")
    else:
        st.caption("Each dotted point is what the model predicted 10 minutes before; "
                   "the solid line is what actually happened. The last point is the "
                   "prediction for the next 10 minutes.")
        st.plotly_chart(chart_pred_vs_actual(resolved, nxt, last_ts, now_temp, colors, tz),
                        width="stretch", config=PLOT_CONFIG)

    # --- Learning ---
    st.subheader("Is the model learning?")
    roll = rolling_errors(resolved)
    if roll.empty:
        st.info("This chart appears after a few predictions have been checked (about 30 minutes).")
    else:
        st.caption(f"Rolling average error over the last {ROLL_WINDOW} checked predictions "
                   f"(about {ROLL_WINDOW * HORIZON_MIN // 60} hours). When the blue line sits "
                   "below the gray one, the model beats the 'temperature stays the same' guess.")
        st.plotly_chart(chart_error_trend(roll, colors, tz), width="stretch", config=PLOT_CONFIG)
        if summary and summary["improvement"] is not None:
            imp = summary["improvement"]
            word = "better" if imp >= 0 else "worse"
            st.markdown(f"Over the {window_label.lower()}, the model is "
                        f"**{abs(imp):.0f}% {word}** than the no-change baseline "
                        f"({summary['n']:,} predictions checked).")

    # --- Other conditions ---
    with st.expander("Other conditions (humidity, pressure, wind, cloud)"):
        left, right = st.columns(2)
        for col, (field, unit, title, dec) in zip(
            [left, right, left, right],
            [("humidity", "%", "Humidity", 0), ("pressure", "hPa", "Pressure", 1),
             ("wind_speed", "m/s", "Wind speed", 1), ("cloud_cover", "%", "Cloud cover", 0)],
        ):
            col.markdown(f"**{title}**")
            col.plotly_chart(chart_by_source(readings, field, unit, colors, height=240, decimals=dec),
                             width="stretch", config=PLOT_CONFIG)

    # --- Raw data ---
    with st.expander("Raw data"):
        st.markdown("**Readings**")
        st.dataframe(readings.drop(columns=["ts_local"]).sort_values("ts", ascending=False),
                     hide_index=True, width="stretch")
        st.download_button("Download readings (CSV)", readings.drop(columns=["ts_local"]).to_csv(index=False),
                           "readings.csv", "text/csv")
        st.markdown("**Predictions**")
        st.dataframe(preds.sort_values("made_at", ascending=False), hide_index=True, width="stretch")
        st.download_button("Download predictions (CSV)", preds.to_csv(index=False),
                           "predictions.csv", "text/csv")

    with st.expander("How it works"):
        st.markdown(
            f"""
1. **Collect.** Every ~10 minutes a GitHub Actions job fetches current weather from
   OpenWeatherMap and Open-Meteo and stores it in a Supabase (Postgres) database.
2. **Check.** It compares the prediction made {HORIZON_MIN} minutes earlier with the
   temperature that actually arrived, and records the error.
3. **Learn.** The online model (`river` linear regression) updates itself with that one
   new example (`learn_one`), so it corrects its mistakes every cycle without retraining.
4. **Predict.** It forecasts the temperature change for the next {HORIZON_MIN} minutes
   from current conditions, time of day, recent trends and the disagreement between sources.
5. **Display.** This dashboard reads the database and refreshes every {REFRESH_SECONDS} seconds.

Every prediction is also scored against a **no-change baseline** (next temperature =
current temperature), so the error chart shows whether the learning adds real value.
"""
        )


def main():
    st.set_page_config(page_title="Live Weather Forecast", page_icon="🌦️", layout="wide")

    url, key = secret("SUPABASE_URL"), secret("SUPABASE_KEY")
    if not url or not key:
        st.error("Missing Supabase settings. In Streamlit Cloud open **Settings → Secrets** and add "
                 "`SUPABASE_URL` and `SUPABASE_KEY` (your publishable key).")
        st.stop()

    tz = ZoneInfo(secret("TIMEZONE", "America/Vancouver"))
    location = secret("LOCATION_NAME", "Vancouver, BC")

    with st.sidebar:
        st.header("Settings")
        window_label = st.radio("Time window", list(WINDOWS), index=1)
        if st.button("Refresh now"):
            st.cache_data.clear()
        st.caption(f"The page refreshes itself every {REFRESH_SECONDS} seconds.")

    dashboard(url, key, WINDOWS[window_label], window_label, tz, location)


if __name__ == "__main__":
    main()
