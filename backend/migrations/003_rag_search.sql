-- ============================================================================
-- 003 · RAG — el camino de búsqueda por similitud sobre el corpus
--
-- POR QUÉ EXISTE, si ya funciona en la base
--
-- `public.match_zartha_chunks(...)`, el índice HNSW y la extensión `vector`
-- están aplicados en Supabase, pero se aplicaron a mano: NADA en el repositorio
-- los reproduce. 001 crea el esquema Delphi y 002 pone la RLS —y da por hecho
-- que `zartha_chunks` ya existe—, así que hoy un entorno nuevo se queda sin
-- función de búsqueda y `core/rag.py` cae a su consulta en línea de respaldo.
--
-- Esta migración cierra ese hueco. Es idempotente y NO cambia el
-- comportamiento en la base actual: la firma y el cuerpo de la función son
-- exactamente los que ya están desplegados.
--
-- LO QUE NO HACE, a propósito
--
-- · No crea `zartha_chunks`. La tabla es el producto de la ingesta del corpus
--   (120 fragmentos con sus embeddings), no de una migración, y crearla aquí
--   invertiría el orden con 002, que ya la referencia.
-- · No toca la RLS de la tabla: eso es de 002 y está bien donde está.
--
-- VALIDACIÓN: probada con `begin; … rollback;`. No aplicada.
-- ============================================================================

begin;

-- ── 1. pgvector ─────────────────────────────────────────────────────────────
-- Instalada en `public`, que es donde está hoy y de donde el tipo `vector`
-- tiene que ser resoluble para la firma de la función.

create extension if not exists vector with schema public;


-- ── 2. Índices del corpus ───────────────────────────────────────────────────
-- Guardados con to_regclass: en un entorno nuevo la tabla puede no existir
-- todavía y esta migración no debe ser la que rompa el despliegue.
--
-- HNSW y no IVFFlat: el corpus es pequeño (120 filas) y crece por ingestas
-- esporádicas, así que no hay a quién entrenar las listas de IVFFlat, y HNSW
-- no necesita reentrenarse al insertar. `vector_cosine_ops` porque la
-- consulta ordena por `<=>`; con otra clase de operador el índice no se usa y
-- la búsqueda degrada a escaneo secuencial sin avisar.

do $$
begin
  if to_regclass('public.zartha_chunks') is null then
    raise notice 'public.zartha_chunks no existe todavía: se omiten índices.';
    return;
  end if;

  create index if not exists zartha_chunks_embedding_idx
    on public.zartha_chunks using hnsw (embedding vector_cosine_ops);

  -- `temas` filtra la recuperación por eje temático (curvas_s, delphi, …).
  create index if not exists zartha_chunks_temas_idx
    on public.zartha_chunks using gin (temas);
end$$;


-- ── 3. La función de búsqueda ───────────────────────────────────────────────
-- Firma IDÉNTICA a la desplegada: (vector, integer, text). Añadirle un cuarto
-- parámetro con valor por defecto —un umbral de similitud, por ejemplo— NO
-- sería un reemplazo sino una sobrecarga, y entonces toda llamada con tres
-- argumentos posicionales quedaría ambigua ("function is not unique"). El
-- umbral se aplica en `core/rag.py`, sobre el resultado.
--
-- SECURITY INVOKER (el de por defecto): la función hereda la RLS de quien
-- pregunta, que es justo lo que 002 asume. Convertirla en SECURITY DEFINER
-- abriría el corpus a anon.
--
-- `similarity` = 1 - distancia coseno, para que un número más alto sea mejor.
-- El ORDER BY usa `<=>` (la distancia) y no `1 - (…)`: la expresión negada no
-- es indexable y el planificador se saltaría el índice HNSW.

create or replace function public.match_zartha_chunks(
  query_embedding vector,
  match_count     integer default 6,
  filter_tema     text    default null
)
returns table (
  id          bigint,
  cite_key    text,
  title       text,
  year        integer,
  source      text,
  url         text,
  temas       text[],
  chunk_index integer,
  content     text,
  similarity  double precision
)
language sql
stable
as $function$
  select
    c.id, c.cite_key, c.title, c.year, c.source, c.url, c.temas,
    c.chunk_index, c.content,
    1 - (c.embedding <=> query_embedding) as similarity
  from public.zartha_chunks c
  where filter_tema is null or filter_tema = any(c.temas)
  order by c.embedding <=> query_embedding
  limit match_count;
$function$;

comment on function public.match_zartha_chunks(vector, integer, text) is
  'Búsqueda por similitud coseno sobre el corpus metodológico. La usa '
  'backend/core/rag.py. SECURITY INVOKER: hereda la RLS de zartha_chunks (002).';


-- ── 4. Quién puede ejecutarla ───────────────────────────────────────────────
-- Coherente con 002: el corpus es para usuarios autenticados y para el
-- backend; anon queda fuera. La RLS ya lo frenaría —la función es INVOKER y
-- no hay política para anon—, así que revocar el EXECUTE es redundante y por
-- eso mismo barato: si algún día se añade una política permisiva de más, el
-- permiso ausente sigue frenando.

revoke execute on function public.match_zartha_chunks(vector, integer, text) from public;
revoke execute on function public.match_zartha_chunks(vector, integer, text) from anon;
grant  execute on function public.match_zartha_chunks(vector, integer, text) to authenticated;
grant  execute on function public.match_zartha_chunks(vector, integer, text) to service_role;

commit;
