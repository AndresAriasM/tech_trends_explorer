-- ============================================================================
-- 004 · Tablero de la matriz patentes/papers
--
-- Un taller evalúa varias tecnologías en vivo y quiere verlas JUNTAS en la
-- matriz (convención de Zartha: X = año de inflexión de papers, Y = de
-- patentes, corte en el año de referencia). Hoy cada análisis de
-- /patents-papers se pierde al recargar.
--
--   matrix_sessions  · el contenedor ("Taller de hoy"), con su año de
--                      referencia y el análisis de IA cuando se genera.
--   matrix_runs      · una tecnología corrida dentro de la sesión: ecuaciones,
--                      fuente de patentes, resúmenes de ambas curvas y su
--                      posición en la matriz.
--
-- Las inflexiones se guardan como MEDIANA de los modelos que convergieron, con
-- el rango entre modelos: un solo modelo que extrapola décadas no decide el
-- cuadrante (ver data/macro_test y backend/scripts/macro_consolidate.py).
--
-- RLS activa y sin políticas, igual que zartha_chunks: sólo el backend
-- (postgres / service_role) lee y escribe. Idempotente.
-- ============================================================================

begin;

create table if not exists matrix_sessions (
  id              uuid primary key default gen_random_uuid(),
  name            text        not null check (length(trim(name)) between 1 and 200),
  description     text        not null default '',
  reference_year  integer     not null default 2025 check (reference_year between 1990 and 2100),
  created_by      uuid        null,
  created_at      timestamptz not null default now(),
  -- Análisis de IA de la sesión completa. Siempre marcado is_ai_generated.
  ai_analysis     jsonb       null,
  ai_generated_at timestamptz null
);

create table if not exists matrix_runs (
  id                  uuid primary key default gen_random_uuid(),
  session_id          uuid        not null references matrix_sessions(id) on delete cascade,
  label               text        not null check (length(trim(label)) between 1 and 200),
  scopus_query        text        not null,
  patents_query       text        not null,
  patent_source       text        not null check (patent_source in ('epo', 'google')),
  year_from           integer     not null,
  year_to             integer     not null,
  -- Resúmenes (series anuales, tabla de modelos sin residuos, fase, totales).
  papers              jsonb       not null,
  patents             jsonb       not null,
  comparison          jsonb       null,
  papers_inflection   numeric     null,
  patents_inflection  numeric     null,
  papers_range        numeric[]   null,
  patents_range       numeric[]   null,
  quadrant            text        null check (quadrant in ('maturity', 'industrial', 'disruption', 'watch')),
  -- Todo el rango de modelos cae del mismo lado del corte en ambos ejes.
  robust              boolean     null,
  created_at          timestamptz not null default now()
);

create index if not exists matrix_runs_session_idx on matrix_runs (session_id, created_at);

alter table matrix_sessions enable row level security;
alter table matrix_runs     enable row level security;

commit;
