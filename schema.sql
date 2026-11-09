-- ============================================================
-- Site Weather Risk Monitor: complete database setup
-- Run once in the Supabase SQL Editor. Safe to run again.
-- ============================================================

-- 1. Live weather readings: one row per source per 10-minute cycle
create table if not exists readings (
  id             bigint generated always as identity primary key,
  ts             timestamptz not null default now(),
  source         text not null,               -- 'openweathermap' or 'open-meteo'
  temperature    double precision,            -- °C
  humidity       double precision,            -- %
  pressure       double precision,            -- hPa (sea level)
  wind_speed     double precision,            -- m/s
  cloud_cover    double precision             -- %
);
alter table readings add column if not exists precipitation double precision;  -- mm
alter table readings add column if not exists wind_gust     double precision;  -- m/s
create index if not exists readings_ts_idx on readings (ts desc);

-- 2. Live air quality (Open-Meteo Air Quality API, CAMS model)
create table if not exists air_quality (
  id                bigint generated always as identity primary key,
  ts                timestamptz not null default now(),
  source            text not null,
  pm2_5             double precision,  -- µg/m³
  pm10              double precision,  -- µg/m³
  us_aqi            double precision,  -- US Air Quality Index
  ozone             double precision,  -- µg/m³
  nitrogen_dioxide  double precision,  -- µg/m³
  carbon_monoxide   double precision   -- µg/m³
);
create index if not exists air_quality_ts_idx on air_quality (ts desc);

-- 3. Public dataset: hourly ERA5 reanalysis history (Copernicus / ECMWF)
create table if not exists history (
  ts             timestamptz primary key,
  temperature    double precision,
  humidity       double precision,
  pressure       double precision,
  wind_speed     double precision,
  wind_gust      double precision,
  cloud_cover    double precision,
  precipitation  double precision,
  dataset        text not null default 'ERA5 reanalysis',
  loaded_at      timestamptz not null default now()
);

-- 4. Predictions: each 10-minute forecast and, once checked, how far off it was
create table if not exists predictions (
  id              bigint generated always as identity primary key,
  made_at         timestamptz not null default now(),
  target_time     timestamptz not null,        -- made_at + 10 minutes
  predicted_temp  double precision not null,   -- self-learning model
  baseline_temp   double precision not null,   -- "no change" guess = temperature at made_at
  actual_temp     double precision,            -- filled in by the next cycle
  error           double precision,            -- |actual - predicted|
  baseline_error  double precision,            -- |actual - baseline|
  resolved_at     timestamptz,
  features        jsonb not null               -- model inputs, so the model can learn later
);
alter table predictions add column if not exists frozen_temp  double precision;  -- never-learning model
alter table predictions add column if not exists frozen_error double precision;
create index if not exists predictions_target_idx on predictions (target_time);
create index if not exists predictions_made_idx   on predictions (made_at desc);

-- 5. Saved models: id 1 = self-learning model, id 2 = frozen model
create table if not exists model_state (
  id          int primary key,
  model_b64   text not null,
  n_learned   int not null default 0,          -- live learning steps so far
  updated_at  timestamptz not null default now()
);

-- ============================================================
-- Security: the dashboard (publishable key) may only READ these tables.
-- Only the collector (secret key) can write. model_state stays private.
-- ============================================================
alter table readings    enable row level security;
alter table air_quality enable row level security;
alter table history     enable row level security;
alter table predictions enable row level security;
alter table model_state enable row level security;

drop policy if exists "public read readings"    on readings;
drop policy if exists "public read air_quality" on air_quality;
drop policy if exists "public read history"     on history;
drop policy if exists "public read predictions" on predictions;
create policy "public read readings"    on readings    for select using (true);
create policy "public read air_quality" on air_quality for select using (true);
create policy "public read history"     on history     for select using (true);
create policy "public read predictions" on predictions for select using (true);
