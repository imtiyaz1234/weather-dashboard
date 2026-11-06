# Day 1 Setup Guide — Get Data Flowing

Goal for today: a new row in Supabase every 10 minutes, with predictions that get checked and learned from.

## Files in this folder
| File | What it does |
|---|---|
| `schema.sql` | Creates the 3 database tables (readings, predictions, model_state) |
| `collector.py` | Fetch → store → check last prediction → learn → predict next 10 min |
| `requirements.txt` | Python packages |
| `.github/workflows/collector.yml` | Runs `collector.py` every 10 minutes on GitHub Actions |
| `.env.example` | Template for your keys when testing on your own computer |

---

## Step 1 — OpenWeatherMap key (do this FIRST, ~5 min)
1. Sign up at https://openweathermap.org/api (free plan).
2. Go to your profile → **My API keys** and copy the key.
3. New keys can take **up to 2 hours** to start working. Continue with the other steps meanwhile.
   (The collector still runs with only Open-Meteo until the key activates.)

## Step 2 — Supabase database (~10 min)
1. Sign up at https://supabase.com → **New project** (pick any name and a strong database password; region: closest to you).
2. Open **SQL Editor** → **New query**, paste all of `schema.sql`, and click **Run**.
3. Check **Table Editor**: you should see `readings`, `predictions`, `model_state`.
4. Go to **Project Settings → API** and copy:
   - **Project URL** → this is `SUPABASE_URL`
   - **service_role** secret key → this is `SUPABASE_KEY`
     (keep it secret — it can write to your database. The dashboard tomorrow will use the public *anon* key instead.)

## Step 3 — Test on your computer (optional but recommended, ~10 min)
```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your real keys into .env
python collector.py
```
Expected first-run output:
```
[fetch] 2 source(s), fused temp 8.43°C
[model] new model, warm-started on ~1400 historical samples
[predict] in 10 min: 8.51°C (change +0.08)
[model] saved, live learning steps so far: 0
```
Run it again after ~10 minutes and you'll also see a `[check] predicted ... actual ...` line — that's the model learning from its mistake.

## Step 4 — Put the code on GitHub (~10 min)
1. Create a new repository on https://github.com (e.g. `weather-dashboard`). **Public** repos get unlimited free Actions minutes.
2. Upload all files from this folder, **including the `.github/workflows/collector.yml` file** (keep the folder path exactly). Do **not** upload `.env`.

## Step 5 — Add your secrets to GitHub (~5 min)
Repo → **Settings → Secrets and variables → Actions → New repository secret**. Add three:
- `OWM_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_KEY`

## Step 6 — Start the collector
1. Repo → **Actions** tab → enable workflows if asked.
2. Click **Weather collector → Run workflow** to test it now.
3. Open the run and check the log for the `[fetch]` and `[predict]` lines.
4. From now on it runs by itself every ~10 minutes (GitHub sometimes delays runs by 5–15 minutes; the code handles that).

## ✅ Day 1 done when
- Supabase `readings` table gets new rows every ~10 minutes (2 per run: one per source)
- `predictions` rows get `actual_temp` and `error` filled in on the following run
- `model_state.n_learned` keeps going up

## Troubleshooting
| Problem | Fix |
|---|---|
| `[warn] fetch_openweathermap failed: 401` | Key not active yet — wait up to 2 hours |
| `KeyError: 'SUPABASE_URL'` | Secret missing or misspelled in GitHub / `.env` |
| `new row violates row-level security` | You used the anon key; use the **service_role** key for the collector |
| Workflow doesn't run on schedule | Make sure the file is at `.github/workflows/collector.yml` and Actions are enabled |
| Scheduled runs stop after 60 days | GitHub pauses schedules on inactive repos; just push any small change |

## How the learning works (for your report)
- The model predicts the **change** in temperature over the next 10 minutes, using current conditions, time of day, recent temperature and pressure trends, and the disagreement between the two sources.
- Each prediction is stored with the exact inputs used. When the next run gets the real temperature, it computes the error and calls `learn_one()` with the true change — so the model corrects itself every cycle.
- Every prediction is also compared to a **"no change" baseline** (next temp = current temp). If the model's error is lower than the baseline's, it's genuinely learning something useful.
- On the very first run, the model is **warm-started** on 30 days of hourly history so it doesn't start from zero.
