-- ============================================================
-- Weather Dashboard — database setup (run once in Supabase SQL Editor)
-- ============================================================

-- 1. Raw readings: one row per source per collection cycle
create table if not exists readings (
  id           bigint generated always as identity primary key,
  ts           timestamptz not null default now(),
  source       text not null,              -- 'openweathermap' or 'open-meteo'
  temperature  double precision,           -- °C
  humidity     double precision,           -- %
  pressure     double precision,           -- hPa (sea level)
  wind_speed   double precision,           -- m/s
  cloud_cover  double precision            -- %
);
create index if not exists readings_ts_idx on readings (ts desc);

-- 2. Predictions: what the model said, and (10 min later) what really happened
create table if not exists predictions (
  id              bigint generated always as identity primary key,
  made_at         timestamptz not null default now(),
  target_time     timestamptz not null,     -- made_at + 10 minutes
  predicted_temp  double precision not null,
  baseline_temp   double precision not null, -- "no change" guess = temp at made_at
  actual_temp     double precision,          -- filled in by the next run
  error           double precision,          -- |actual - predicted|
  baseline_error  double precision,          -- |actual - baseline|
  resolved_at     timestamptz,
  features        jsonb not null             -- inputs used, so the model can learn later
);
create index if not exists predictions_target_idx on predictions (target_time);
create index if not exists predictions_made_idx on predictions (made_at desc);

-- 3. Model state: the trained model, saved between runs
create table if not exists model_state (
  id          int primary key,
  model_b64   text not null,
  n_learned   int not null default 0,       -- live learning steps so far
  updated_at  timestamptz not null default now()
);

-- ============================================================
-- Security: the dashboard (Day 2) may READ readings and predictions.
-- Only the collector (using the secret key) can write.
-- model_state has no public policy, so it stays private.
-- ============================================================
alter table readings    enable row level security;
alter table predictions enable row level security;
alter table model_state enable row level security;

create policy "public read readings"    on readings    for select using (true);
create policy "public read predictions" on predictions for select using (true);
