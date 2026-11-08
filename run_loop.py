"""
run_loop.py: keeps the collector running every 10 minutes.

GitHub's scheduler is unreliable on free accounts (it can skip hours), so instead of
relying on it, each GitHub Actions job runs this loop for about 5.5 hours, calling
collector.main() every 10 minutes. When the loop ends, the workflow starts the next
job itself, so collection continues around the clock.
"""

import os
import time
import traceback
from datetime import datetime, timezone

import collector
import history

INTERVAL = int(os.getenv("INTERVAL_SECONDS", "600"))     # 10 minutes
LOOP_MINUTES = int(os.getenv("LOOP_MINUTES", "340"))     # stay under GitHub's 6-hour job limit


def load_public_dataset():
    """Refresh the 30-day ERA5 public dataset once per job. Never blocks collection."""
    try:
        rows = history.refresh_history(collector.sb, collector.LAT, collector.LON)
        span = f", {rows[0]['ts']} to {rows[-1]['ts']}" if rows else ""
        print(f"[history] public dataset: {len(rows)} hourly ERA5 records loaded{span}", flush=True)
    except Exception as e:
        print(f"[warn] public dataset not loaded: {e}", flush=True)


def main():
    load_public_dataset()
    end = time.time() + LOOP_MINUTES * 60
    cycle = 0
    while True:
        start = time.time()
        cycle += 1
        print(f"=== cycle {cycle} at {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC ===", flush=True)
        try:
            collector.main()
        except SystemExit as e:          # collector exits when no source responds
            print(f"[warn] cycle skipped (exit code {e.code})", flush=True)
        except Exception:                # never let one bad cycle stop the loop
            traceback.print_exc()
        next_start = start + INTERVAL
        if next_start > end:
            print(f"Loop finished after {cycle} cycles.", flush=True)
            return
        time.sleep(max(0, next_start - time.time()))


if __name__ == "__main__":
    main()
