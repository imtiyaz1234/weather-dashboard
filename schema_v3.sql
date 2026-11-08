-- ============================================================
-- Weather Dashboard: frozen-model comparison (run once in Supabase SQL Editor)
-- Safe to run more than once.
-- ============================================================
-- frozen_temp:  what the never-learning model predicted
-- frozen_error: how far off it was, compared with the learning model's "error"
alter table predictions add column if not exists frozen_temp  double precision;
alter table predictions add column if not exists frozen_error double precision;
