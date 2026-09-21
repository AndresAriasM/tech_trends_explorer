-- ============================================================================
-- 001 · Delphi — completar el esquema para la mecánica de docs/LOGICA_DELPHI.md
--
-- El esquema existente (delphi_studies/rounds/questions/participants/
-- submissions/answers) es correcto y se conserva. Esta migración añade lo que
-- la mecánica necesita y el esquema aún no tiene.
--
-- Todas las tablas delphi_* están vacías, así que no hay backfill ni riesgo
-- sobre datos. Es idempotente: se puede correr dos veces.
--
-- NO incluye políticas RLS — van en 002, porque dependen de decisiones de
-- permisos (quién ve qué) que son independientes del modelo.
-- ============================================================================

begin;

-- ── 1. Configuración del panel (§5.3, §9) ───────────────────────────────────
-- El umbral de dicotomía es el parámetro más consecuente del ejercicio y hoy
-- vive en un literal del notebook. El criterio de parada debe declararse antes
-- de abrir la ronda 1, no decidirse al ver los resultados.

alter table delphi_studies
  add column if not exists horizon_year        integer,
  add column if not exists scale_type          text    not null default 'likert_5',
  add column if not exists dichotomy_threshold integer not null default 4,
  add column if not exists max_rounds          integer not null default 3,
  -- Criterio de parada: {"stability_min": 0.85, "iqr_max": 1.0}
  add column if not exists stopping_rule       jsonb   not null
      default '{"stability_min": 0.85, "iqr_max": 1.0}'::jsonb;

alter table delphi_studies
  drop constraint if exists delphi_studies_horizon_year_check;
alter table delphi_studies
  add constraint delphi_studies_horizon_year_check
  check (horizon_year is null or horizon_year between 2000 and 2100);

alter table delphi_studies
  drop constraint if exists delphi_studies_max_rounds_check;
alter table delphi_studies
  add constraint delphi_studies_max_rounds_check check (max_rounds between 1 and 10);


-- ── 2. Variables del estudio (§8) ───────────────────────────────────────────
-- El ítem Delphi ES una Variable del estudio, no una pregunta de una ronda.
-- Sin esto, el juicio del panel sobre "Científico de datos" y la curva en S de
-- esa misma ocupación no pueden colgar de la misma entidad.
--
-- `family` da su sitio a la taxonomía que hoy está implícita en las columnas
-- marcadoras Opiniones* y que la ETL descarta.

create table if not exists delphi_variables (
  id          uuid primary key default gen_random_uuid(),
  study_id    uuid not null references delphi_studies(id) on delete cascade,
  code        text not null,                 -- estable y legible: "cientifico_datos"
  name        text not null,
  description text,
  family      text,                          -- 8 familias de empleos + competencias
  order_index integer not null default 0,
  created_at  timestamptz not null default now(),
  unique (study_id, code)
);

create index if not exists delphi_variables_study_idx  on delphi_variables (study_id);
create index if not exists delphi_variables_family_idx on delphi_variables (study_id, family);


-- ── 3. La pregunta es la aparición de una variable en una ronda ─────────────
-- Nullable a propósito: una ronda puede llevar preguntas de contexto que no
-- sean variables del estudio (p. ej. las de competencia experta).

alter table delphi_questions
  add column if not exists variable_id uuid references delphi_variables(id) on delete cascade;

create index if not exists delphi_questions_variable_idx on delphi_questions (variable_id);

-- Una variable no puede preguntarse dos veces en la misma ronda.
create unique index if not exists delphi_questions_round_variable_uniq
  on delphi_questions (round_id, variable_id) where variable_id is not null;


-- ── 4. Abstención y argumento (§3, §5.1, §8) ────────────────────────────────
-- La abstención NUNCA es un valor dentro de la escala. Tratar NS/NR como 6
-- —o agruparlo con el extremo bajo— es imputación fuerte y no declarada: hoy
-- una abstención cuenta como juicio negativo.
--
-- El argumento es la materia prima de la devolución entre rondas: hoy se
-- recoge en las columnas Opiniones* y la ETL lo tira.

alter table delphi_answers
  add column if not exists is_abstention boolean not null default false,
  add column if not exists argument      text,
  add column if not exists answered_at   timestamptz not null default now();

-- `value` nacía NOT NULL, lo que hacía imposible registrar una abstención:
-- había que meter un número —el 6 del notebook— y volvíamos al problema que
-- esta migración resuelve. El CHECK de abajo garantiza que sólo puede ser
-- null cuando hay abstención declarada.
alter table delphi_answers alter column value drop not null;

-- Si se abstiene no hay valor; si no se abstiene, tiene que haberlo.
alter table delphi_answers drop constraint if exists delphi_answers_abstention_check;
alter table delphi_answers
  add constraint delphi_answers_abstention_check
  check (
    (is_abstention and value is null)
    or (not is_abstention and value is not null)
  );


-- ── 5. Competencia experta K (§5.2) ─────────────────────────────────────────
-- El instrumento ya pregunta Kc y el análisis trata a los 55 panelistas como
-- intercambiables. K = (Kc + Ka) / 2 pondera toda agregación posterior.

alter table delphi_participants
  add column if not exists kc          numeric(3,2),
  add column if not exists ka          numeric(3,2),
  add column if not exists education   text,
  add column if not exists affiliation text,
  add column if not exists display_name text;

alter table delphi_participants drop constraint if exists delphi_participants_kc_check;
alter table delphi_participants
  add constraint delphi_participants_kc_check check (kc is null or kc between 0 and 1);
alter table delphi_participants drop constraint if exists delphi_participants_ka_check;
alter table delphi_participants
  add constraint delphi_participants_ka_check check (ka is null or ka between 0 and 1);

-- K es derivado: columna generada para que no pueda desincronizarse.
alter table delphi_participants
  add column if not exists k numeric(3,2)
  generated always as ((coalesce(kc, 0) + coalesce(ka, 0)) / 2.0) stored;


-- ── 6. Estado de transición entre rondas (§8) ───────────────────────────────
-- Un enum explícito, nunca una ausencia. Hoy la celda en blanco significa
-- cuatro cosas indistinguibles y dos de ellas son bugs de etiqueta.

do $$
begin
  if not exists (select 1 from pg_type where typname = 'transition_state') then
    create type transition_state as enum (
      -- comparables
      'mantiene_alto', 'mantiene_bajo', 'subio', 'bajo',
      -- no comparables, pero distinguibles entre sí
      'abstuvo', 'no_respondio', 'variable_ausente'
    );
  end if;
end$$;


-- ── 7. Respuestas por variable y ronda ──────────────────────────────────────
-- Vista que aplana el camino answer → submission → participant/round y expone
-- la variable. Es la base de todo el cálculo entre rondas.

create or replace view delphi_variable_responses as
select
  r.study_id,
  r.id            as round_id,
  r.round_number,
  q.variable_id,
  s.participant_id,
  p.k             as panelist_k,
  a.value,
  a.is_abstention,
  a.argument,
  s.status        as submission_status
from delphi_answers a
join delphi_submissions s on s.id = a.submission_id
join delphi_questions   q on q.id = a.question_id
join delphi_rounds      r on r.id = s.round_id
join delphi_participants p on p.id = s.participant_id
where q.variable_id is not null
  and s.status = 'submitted';


-- ── 8. Transiciones entre rondas consecutivas ───────────────────────────────
-- DERIVADO, no almacenado: una tabla de transiciones se desincroniza en cuanto
-- alguien corrige una respuesta. La dicotomización usa el umbral del estudio,
-- así que mover el umbral recalcula el mapa sin migrar nada (§5.3).
--
-- Sólo cubre la POBLACIÓN DE CAMBIO (§6): el inner join exige que el panelista
-- haya respondido ambas rondas. El agregado por ronda se calcula aparte, sobre
-- la población de agregado, y las dos cifras se reportan siempre separadas.

create or replace view delphi_transitions as
with numeric_responses as (
  select
    vr.*,
    st.dichotomy_threshold,
    case
      when vr.is_abstention then null
      when jsonb_typeof(vr.value) = 'number' then (vr.value #>> '{}')::numeric
      else null
    end as numeric_value
  from delphi_variable_responses vr
  join delphi_studies st on st.id = vr.study_id
)
select
  a.study_id,
  a.variable_id,
  a.participant_id,
  a.round_number as round_from,
  b.round_number as round_to,
  a.numeric_value as value_from,
  b.numeric_value as value_to,
  case
    when a.is_abstention or b.is_abstention               then 'abstuvo'::transition_state
    when a.numeric_value is null or b.numeric_value is null then 'no_respondio'::transition_state
    when a.numeric_value >= a.dichotomy_threshold
     and b.numeric_value >= b.dichotomy_threshold          then 'mantiene_alto'::transition_state
    when a.numeric_value <  a.dichotomy_threshold
     and b.numeric_value <  b.dichotomy_threshold          then 'mantiene_bajo'::transition_state
    when a.numeric_value <  a.dichotomy_threshold
     and b.numeric_value >= b.dichotomy_threshold          then 'subio'::transition_state
    else                                                        'bajo'::transition_state
  end as state
from numeric_responses a
join numeric_responses b
  on  b.study_id      = a.study_id
  and b.variable_id   = a.variable_id
  and b.participant_id = a.participant_id
  and b.round_number  = a.round_number + 1;


-- ── 9. Tabla de transición agregada por variable (§2, §9) ───────────────────
-- Las cuatro celdas a/b/c/d. El notebook colorea sólo a y d; b y c —quienes
-- cambiaron de opinión— son la respuesta a la pregunta del ejercicio y las dos
-- celdas discordantes de McNemar. El contraste estadístico se calcula en
-- Python (core/delphi.py); aquí se dejan los conteos listos.

create or replace view delphi_variable_transition_matrix as
select
  study_id,
  variable_id,
  round_from,
  round_to,
  count(*) filter (where state = 'mantiene_alto') as a_mantiene_alto,
  count(*) filter (where state = 'bajo')          as b_bajo,
  count(*) filter (where state = 'subio')         as c_subio,
  count(*) filter (where state = 'mantiene_bajo') as d_mantiene_bajo,
  count(*) filter (where state in ('abstuvo','no_respondio','variable_ausente'))
                                                  as no_comparables,
  count(*) filter (where state in
    ('mantiene_alto','bajo','subio','mantiene_bajo'))         as pareados,
  -- estabilidad = (a + d) / pareados ; direccion_neta = c − b
  round(
    (count(*) filter (where state in ('mantiene_alto','mantiene_bajo')))::numeric
    / nullif(count(*) filter (where state in
        ('mantiene_alto','bajo','subio','mantiene_bajo')), 0)
  , 4) as estabilidad,
  (count(*) filter (where state = 'subio'))
    - (count(*) filter (where state = 'bajo')) as direccion_neta
from delphi_transitions
group by study_id, variable_id, round_from, round_to;

commit;
