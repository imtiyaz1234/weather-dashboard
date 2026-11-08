-- ============================================================
-- Weather Dashboard: new data sources (run once in Supabase SQL Editor)
-- Safe to run more than once.
-- ============================================================

-- 1. Live readings: add rain and wind gusts (important for site decisions)
alter table readings add column if not exists precipitation double precision;  -- mm
alter table readings add column if not exists wind_gust     double precision;  -- m/s

-- 2. Public dataset: hourly ERA5 reanalysis history (Copernicus / ECMWF)
create table if not exists history (
  ts             timestamptz primary key,
  temperature    double precision,   -- °C
  humidity       double precision,   -- %
  pressure       double precision,   -- hPa (sea level)
  wind_speed     double precision,   -- m/s
  wind_gust      double precision,   -- m/s
  cloud_cover    double precision,   -- %
  precipitation  double precision,   -- mm
  dataset        text not null default 'ERA5 reanalysis',
  loaded_at      timestamptz not null default now()
);

-- 3. Third live source: air quality (Open-Meteo / CAMS)
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

-- 4. Security: the dashboard may read the new tables; only the collector writes
alter table history     enable row level security;
alter table air_quality enable row level security;

drop policy if exists "public read history"     on history;
drop policy if exists "public read air_quality" on air_quality;
create policy "public read history"     on history     for select using (true);
create policy "public read air_quality" on air_quality for select using (true);
