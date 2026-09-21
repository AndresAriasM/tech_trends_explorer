-- ============================================================================
-- 002 · Políticas RLS — quién ve qué en Prospecta
--
-- Prerrequisito: 001_delphi_prospecta.sql (crea delphi_variables y las tres
-- vistas de análisis; esta migración las asegura).
--
-- Punto de partida: RLS está ACTIVADO en las seis tablas delphi_* y en
-- zartha_chunks pero con CERO políticas. Sin política permisiva, RLS niega
-- todo: hoy sólo la service_role puede leer. Esto no es seguridad, es una
-- puerta tapiada. Esta migración pone las cerraduras.
--
-- Las tres reglas, en una línea cada una:
--
--   admin     → todo.
--   profesor  → los estudios de los que es owner, completos.
--   experto   → las rondas ABIERTAS de los estudios donde es participante,
--               y SÓLO SUS PROPIAS respuestas.
--
-- La tercera es la que manda: docs/LOGICA_DELPHI.md §5.1 y §6. Un panelista
-- nunca ve la respuesta individual de otro panelista. Ni la suya de otra
-- ronda que no esté abierta, ni la lista del panel. Lo que sí ve —la
-- distribución agregada y los argumentos anónimos de la devolución— NO sale
-- de aquí: lo sirve el backend con service_role, que es el único sitio donde
-- se puede anonimizar de verdad. Una política RLS puede ocultar filas; no
-- puede convertir filas en un agregado sin autor.
--
-- Convenciones heredadas de las políticas que ya existen en profiles y
-- role_assignments (`select * from pg_policies`):
--   · nombre `<tabla>_<sujeto>_<acción>`
--   · `to public` con el predicado sobre auth.uid(), no `to authenticated`
--   · helpers en SQL/plpgsql STABLE SECURITY DEFINER con `set search_path = ''`
--     y nombres siempre cualificados
--
-- Idempotente: se puede correr dos veces.
-- ============================================================================

begin;

-- ── 0. Helpers ──────────────────────────────────────────────────────────────
-- Todos SECURITY DEFINER, por dos razones:
--   1. Evitan la recursión: una política sobre delphi_answers que consultara
--      delphi_submissions dispararía la RLS de delphi_submissions, que a su
--      vez consulta delphi_participants… Dentro de un SECURITY DEFINER la
--      consulta corre como el dueño de las tablas y la RLS no se reevalúa.
--   2. Son el equivalente de `is_admin()`, que ya sigue exactamente este
--      patrón. Se mantiene la convención.
--
-- Ninguno recibe al usuario como argumento: todos parten de auth.uid(). Un
-- helper al que se le pudiera pasar «otro» usuario sería un oráculo.

create or replace function public.current_user_role()
returns public.user_role
language sql
stable
security definer
set search_path = ''
as $$
  select role from public.profiles where id = auth.uid();
$$;

comment on function public.current_user_role() is
  'Rol del usuario autenticado según profiles. La fuente de verdad es la tabla, '
  'no el claim user_role del JWT: el claim lo inyecta custom_access_token() al '
  'emitir el token y queda obsoleto si un admin cambia el rol a media sesión.';


create or replace function public.is_study_owner(p_study uuid)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1 from public.delphi_studies s
    where s.id = p_study and s.owner_id = auth.uid()
  );
$$;


create or replace function public.is_study_participant(p_study uuid)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1 from public.delphi_participants p
    where p.study_id = p_study
      and p.user_id  = auth.uid()
      and p.status <> 'dropped'
  );
$$;

comment on function public.is_study_participant(uuid) is
  'Participante vivo del estudio. Un panelista dado de baja (status=dropped) '
  'pierde el acceso pero conserva sus respuestas: la deserción es un dato del '
  'panel (LOGICA_DELPHI §6), no un borrado.';


create or replace function public.is_my_participant(p_participant uuid)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1 from public.delphi_participants p
    where p.id = p_participant and p.user_id = auth.uid()
  );
$$;


create or replace function public.participant_study_id(p_participant uuid)
returns uuid
language sql
stable
security definer
set search_path = ''
as $$
  select study_id from public.delphi_participants where id = p_participant;
$$;


create or replace function public.round_study_id(p_round uuid)
returns uuid
language sql
stable
security definer
set search_path = ''
as $$
  select study_id from public.delphi_rounds where id = p_round;
$$;


create or replace function public.is_round_open(p_round uuid)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1 from public.delphi_rounds r
    where r.id = p_round
      and r.status = 'open'
      and (r.opens_at  is null or now() >= r.opens_at)
      and (r.closes_at is null or now() <= r.closes_at)
  );
$$;

comment on function public.is_round_open(uuid) is
  'Abierta = estado open Y dentro de la ventana. El cierre por fecha deja de '
  'depender de que alguien ejecute el cambio de estado a tiempo.';


create or replace function public.question_round_id(p_question uuid)
returns uuid
language sql
stable
security definer
set search_path = ''
as $$
  select round_id from public.delphi_questions where id = p_question;
$$;


create or replace function public.submission_round_id(p_submission uuid)
returns uuid
language sql
stable
security definer
set search_path = ''
as $$
  select round_id from public.delphi_submissions where id = p_submission;
$$;


create or replace function public.is_my_submission(p_submission uuid)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1
    from public.delphi_submissions s
    join public.delphi_participants p on p.id = s.participant_id
    where s.id = p_submission and p.user_id = auth.uid()
  );
$$;


create or replace function public.is_submission_study_owner(p_submission uuid)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1
    from public.delphi_submissions s
    join public.delphi_rounds  r on r.id = s.round_id
    join public.delphi_studies st on st.id = r.study_id
    where s.id = p_submission and st.owner_id = auth.uid()
  );
$$;


create or replace function public.variable_in_open_round(p_variable uuid)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1
    from public.delphi_questions q
    join public.delphi_rounds    r on r.id = q.round_id
    where q.variable_id = p_variable
      and r.status = 'open'
      and (r.opens_at  is null or now() >= r.opens_at)
      and (r.closes_at is null or now() <= r.closes_at)
  );
$$;


-- ── 1. delphi_studies ───────────────────────────────────────────────────────
-- El experto ve la ficha del estudio —título, pregunta focal— porque sin ella
-- la ronda no tiene contexto. No ve los estudios en borrador: un estudio en
-- draft es trabajo en curso del profesor.

alter table public.delphi_studies enable row level security;

drop policy if exists delphi_studies_admin_all on public.delphi_studies;
create policy delphi_studies_admin_all on public.delphi_studies
  for all using (public.is_admin()) with check (public.is_admin());

drop policy if exists delphi_studies_owner_all on public.delphi_studies;
create policy delphi_studies_owner_all on public.delphi_studies
  for all
  using (owner_id = auth.uid())
  -- El WITH CHECK impide dos cosas distintas: crear un estudio a nombre de
  -- otro, y que un experto se convierta en profesor por la vía de crear uno.
  with check (
    owner_id = auth.uid()
    and public.current_user_role() in ('admin', 'profesor')
  );

drop policy if exists delphi_studies_participant_select on public.delphi_studies;
create policy delphi_studies_participant_select on public.delphi_studies
  for select
  using (status <> 'draft' and public.is_study_participant(id));


-- ── 2. delphi_variables (creada en 001) ─────────────────────────────────────
-- 001 la crea sin RLS y Supabase concede por defecto a anon/authenticated:
-- sin esto, cualquier usuario autenticado lee las variables de cualquier
-- estudio. El experto sólo ve las que se le están preguntando ahora.

alter table public.delphi_variables enable row level security;

drop policy if exists delphi_variables_admin_all on public.delphi_variables;
create policy delphi_variables_admin_all on public.delphi_variables
  for all using (public.is_admin()) with check (public.is_admin());

drop policy if exists delphi_variables_owner_all on public.delphi_variables;
create policy delphi_variables_owner_all on public.delphi_variables
  for all
  using (public.is_study_owner(study_id))
  with check (public.is_study_owner(study_id));

drop policy if exists delphi_variables_participant_select on public.delphi_variables;
create policy delphi_variables_participant_select on public.delphi_variables
  for select
  using (
    public.is_study_participant(study_id)
    and public.variable_in_open_round(id)
  );


-- ── 3. delphi_rounds ────────────────────────────────────────────────────────
-- La regla literal del enunciado: el experto ve las rondas ABIERTAS, y sólo
-- ésas. Las cerradas no las ve ni siquiera para consultar lo que contestó —
-- lo anterior le llega por la devolución (§5.1), que construye el backend.

alter table public.delphi_rounds enable row level security;

drop policy if exists delphi_rounds_admin_all on public.delphi_rounds;
create policy delphi_rounds_admin_all on public.delphi_rounds
  for all using (public.is_admin()) with check (public.is_admin());

drop policy if exists delphi_rounds_owner_all on public.delphi_rounds;
create policy delphi_rounds_owner_all on public.delphi_rounds
  for all
  using (public.is_study_owner(study_id))
  with check (public.is_study_owner(study_id));

drop policy if exists delphi_rounds_participant_select on public.delphi_rounds;
create policy delphi_rounds_participant_select on public.delphi_rounds
  for select
  using (public.is_study_participant(study_id) and public.is_round_open(id));


-- ── 4. delphi_questions ─────────────────────────────────────────────────────

alter table public.delphi_questions enable row level security;

drop policy if exists delphi_questions_admin_all on public.delphi_questions;
create policy delphi_questions_admin_all on public.delphi_questions
  for all using (public.is_admin()) with check (public.is_admin());

drop policy if exists delphi_questions_owner_all on public.delphi_questions;
create policy delphi_questions_owner_all on public.delphi_questions
  for all
  using (public.is_study_owner(public.round_study_id(round_id)))
  with check (public.is_study_owner(public.round_study_id(round_id)));

drop policy if exists delphi_questions_participant_select on public.delphi_questions;
create policy delphi_questions_participant_select on public.delphi_questions
  for select
  using (
    public.is_round_open(round_id)
    and public.is_study_participant(public.round_study_id(round_id))
  );


-- ── 5. delphi_participants ──────────────────────────────────────────────────
-- LA TABLA CRÍTICA PARA EL ANONIMATO. Un panelista sólo ve su propia fila.
-- No puede enumerar el panel, ni por correo ni por conteo: sin esta política
-- «respuestas anónimas» sería una promesa vacía, porque con 3 panelistas
-- visibles y 3 respuestas agregadas la reidentificación es aritmética.
--
-- No hay política de UPDATE para el participante. Es deliberado: un WITH
-- CHECK no puede comparar con el valor anterior de la fila, así que un
-- `update ... set study_id = <otro estudio>` con user_id intacto pasaría el
-- filtro y se auto-inscribiría en un estudio ajeno. El alta del panelista
-- (canje del token de invitación, Kc/Ka) la hace el backend con service_role,
-- que además es donde el token debe validarse.

alter table public.delphi_participants enable row level security;

drop policy if exists delphi_participants_admin_all on public.delphi_participants;
create policy delphi_participants_admin_all on public.delphi_participants
  for all using (public.is_admin()) with check (public.is_admin());

drop policy if exists delphi_participants_owner_all on public.delphi_participants;
create policy delphi_participants_owner_all on public.delphi_participants
  for all
  using (public.is_study_owner(study_id))
  with check (public.is_study_owner(study_id));

drop policy if exists delphi_participants_self_select on public.delphi_participants;
create policy delphi_participants_self_select on public.delphi_participants
  for select
  using (user_id = auth.uid());


-- ── 6. delphi_submissions ───────────────────────────────────────────────────
-- Asimetría deliberada entre LEER y ESCRIBIR lo propio:
--
--   · leer  → siempre, en cualquier ronda. Son sus datos. Y la devolución de
--             la §5.1 empieza por «su propia respuesta anterior»: negarle al
--             panelista lo que él mismo contestó en la ronda 1 sería una
--             restricción sin beneficiario.
--   · escribir → sólo con la ronda abierta. Cerrada la ronda, su respuesta
--             deja de ser suya y pasa a ser dato del panel: si se pudiera
--             retocar a posteriori, la matriz de transición de la §2 dejaría
--             de significar nada.
--
-- Lo que no cambia en ningún caso: no ve el envío de otro panelista.
--
-- El owner tiene SELECT, no ALL: el facilitador observa la participación pero
-- no escribe en nombre de un panelista. Si hiciera falta (importación del
-- histórico, §10), va por el backend con service_role y queda trazado.

alter table public.delphi_submissions enable row level security;

drop policy if exists delphi_submissions_admin_all on public.delphi_submissions;
create policy delphi_submissions_admin_all on public.delphi_submissions
  for all using (public.is_admin()) with check (public.is_admin());

drop policy if exists delphi_submissions_owner_select on public.delphi_submissions;
create policy delphi_submissions_owner_select on public.delphi_submissions
  for select
  using (public.is_study_owner(public.round_study_id(round_id)));

drop policy if exists delphi_submissions_self_select on public.delphi_submissions;
create policy delphi_submissions_self_select on public.delphi_submissions
  for select
  using (public.is_my_participant(participant_id));

drop policy if exists delphi_submissions_self_insert on public.delphi_submissions;
create policy delphi_submissions_self_insert on public.delphi_submissions
  for insert
  with check (
    public.is_my_participant(participant_id)
    and public.is_round_open(round_id)
    -- El participante y la ronda tienen que ser del mismo estudio: sin esto,
    -- ser panelista de un estudio daría pie a enviar a la ronda de otro.
    and public.participant_study_id(participant_id) = public.round_study_id(round_id)
  );

drop policy if exists delphi_submissions_self_update on public.delphi_submissions;
create policy delphi_submissions_self_update on public.delphi_submissions
  for update
  using (public.is_my_participant(participant_id) and public.is_round_open(round_id))
  with check (
    public.is_my_participant(participant_id)
    and public.is_round_open(round_id)
    and public.participant_study_id(participant_id) = public.round_study_id(round_id)
  );


-- ── 7. delphi_answers ───────────────────────────────────────────────────────
-- Aquí es donde la regla del panelista se vuelve concreta: la política de
-- SELECT del experto no menciona la ronda ni el estudio, sólo «el envío es
-- mío». No hay forma de redactar un select que devuelva la fila de otro.
--
-- Nótese que la vista delphi_variable_responses sí acaba filtrando por ronda
-- abierta aunque esta política no lo haga: la vista pasa por delphi_rounds,
-- que al panelista sólo le muestra las abiertas. La fila cruda propia se lee
-- siempre; la lectura analítica va por donde debe ir.

alter table public.delphi_answers enable row level security;

drop policy if exists delphi_answers_admin_all on public.delphi_answers;
create policy delphi_answers_admin_all on public.delphi_answers
  for all using (public.is_admin()) with check (public.is_admin());

drop policy if exists delphi_answers_owner_select on public.delphi_answers;
create policy delphi_answers_owner_select on public.delphi_answers
  for select
  using (public.is_submission_study_owner(submission_id));

drop policy if exists delphi_answers_self_select on public.delphi_answers;
create policy delphi_answers_self_select on public.delphi_answers
  for select
  using (public.is_my_submission(submission_id));

drop policy if exists delphi_answers_self_insert on public.delphi_answers;
create policy delphi_answers_self_insert on public.delphi_answers
  for insert
  with check (
    public.is_my_submission(submission_id)
    and public.is_round_open(public.submission_round_id(submission_id))
    -- La pregunta tiene que pertenecer a la ronda del envío.
    and public.question_round_id(question_id) = public.submission_round_id(submission_id)
  );

drop policy if exists delphi_answers_self_update on public.delphi_answers;
create policy delphi_answers_self_update on public.delphi_answers
  for update
  using (
    public.is_my_submission(submission_id)
    and public.is_round_open(public.submission_round_id(submission_id))
  )
  with check (
    public.is_my_submission(submission_id)
    and public.is_round_open(public.submission_round_id(submission_id))
    and public.question_round_id(question_id) = public.submission_round_id(submission_id)
  );

drop policy if exists delphi_answers_self_delete on public.delphi_answers;
create policy delphi_answers_self_delete on public.delphi_answers
  for delete
  using (
    public.is_my_submission(submission_id)
    and public.is_round_open(public.submission_round_id(submission_id))
  );


-- ── 8. zartha_chunks ────────────────────────────────────────────────────────
-- Corpus de referencia con embeddings. Lectura para cualquier usuario
-- autenticado —es bibliografía, no datos de panel—; escritura sólo admin,
-- porque la ingesta es un proceso, no una acción de usuario.
--
-- anon queda fuera: el corpus no es público.

alter table public.zartha_chunks enable row level security;

drop policy if exists zartha_chunks_authenticated_select on public.zartha_chunks;
create policy zartha_chunks_authenticated_select on public.zartha_chunks
  for select
  using (auth.uid() is not null);

drop policy if exists zartha_chunks_admin_all on public.zartha_chunks;
create policy zartha_chunks_admin_all on public.zartha_chunks
  for all using (public.is_admin()) with check (public.is_admin());

-- match_zartha_chunks() es SECURITY INVOKER, así que hereda la política de
-- arriba y no hay que tocarla. Se deja constancia por si alguien la convierte
-- en SECURITY DEFINER más adelante: eso abriría el corpus a anon.


-- ── 9. Las vistas de 001 tienen que respetar la RLS ─────────────────────────
-- CRÍTICO. Una vista en Postgres consulta como su DUEÑO (postgres), que es
-- dueño de las tablas y por tanto ignora la RLS. Las tres vistas de 001
-- —delphi_variable_responses en particular— devolverían las respuestas de
-- TODO el panel a cualquiera que las consultara con el token de un experto.
-- Todo el trabajo anterior quedaría anulado por un `select * from` de una
-- vista. security_invoker las hace correr como quien pregunta.

do $$
declare
  v text;
begin
  foreach v in array array[
    'delphi_variable_responses',
    'delphi_transitions',
    'delphi_variable_transition_matrix'
  ] loop
    if to_regclass('public.' || v) is not null then
      execute format('alter view public.%I set (security_invoker = true)', v);
    end if;
  end loop;
end$$;


-- ── 10. Defensa en profundidad sobre los GRANT ──────────────────────────────
-- Supabase concede por defecto INSERT/UPDATE/DELETE a anon en todo lo que se
-- crea en public. Con las políticas de arriba ninguna escritura de anon pasa
-- (todas exigen auth.uid()), así que esto es redundante — y precisamente por
-- eso es barato. Si algún día se añade por error una política permisiva sin
-- predicado, el GRANT ausente sigue frenando.

revoke insert, update, delete, truncate on
    public.delphi_studies,
    public.delphi_rounds,
    public.delphi_questions,
    public.delphi_participants,
    public.delphi_submissions,
    public.delphi_answers,
    public.zartha_chunks
  from anon;

do $$
begin
  if to_regclass('public.delphi_variables') is not null then
    revoke insert, update, delete, truncate on public.delphi_variables from anon;
    grant  select, insert, update, delete on public.delphi_variables to authenticated;
    grant  all on public.delphi_variables to service_role;
  end if;
end$$;

commit;

-- ============================================================================
-- Lo que esta migración deja fuera, a propósito
--
-- · El agregado anónimo de la devolución entre rondas (§5.1). No se puede
--   expresar como RLS: la RLS filtra filas, y aquí hace falta lo contrario
--   —fabricar una fila que no pertenece a nadie a partir de filas que sí—.
--   Va en el backend, con service_role, y con un mínimo de panelistas por
--   agregado (n ≥ 3) para que la media no delate a la minoría.
-- · El canje del token de invitación. Exige escribir en una fila que todavía
--   no es tuya: por definición no hay predicado RLS que lo autorice.
-- · profiles y role_assignments: sus políticas ya existen y no se tocan.
-- ============================================================================
