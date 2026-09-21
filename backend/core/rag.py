"""
backend/core/rag.py

Recuperación sobre el corpus metodológico (tabla `zartha_chunks` en Supabase).

Es la mitad "R" del RAG del asistente: convierte una consulta en texto a un
embedding de OpenAI, busca los fragmentos más próximos por similitud coseno y
los devuelve **con sus metadatos de cita**. Sin esos metadatos el generador no
puede citar, y una cita que no se puede comprobar es una cita inventada.

Decisiones:

  · El corpus vive en Postgres con pgvector (extensión `vector` 0.8.2, índice
    HNSW `zartha_chunks_embedding_idx` con `vector_cosine_ops`). La búsqueda se
    hace con la función SQL `match_zartha_chunks(query_embedding, match_count,
    filter_tema)`, que ya existe en la base. Si faltara, hay un fallback con la
    consulta equivalente en línea, para que el módulo no dependa de que la
    migración esté aplicada.

  · Se conecta por `DATABASE_URL` (psycopg). `zartha_chunks` tiene RLS activa y
    ninguna política, así que sólo los roles que la saltan —`postgres` y
    `service_role`— la leen. Eso es deliberado: el corpus se consulta desde el
    backend, nunca desde el navegador.

  · `min_similarity` es el mecanismo antialucinación en el lado de la
    recuperación: por debajo del umbral el fragmento no entra al contexto, y si
    no entra nada, el generador tiene que decir que el corpus no lo respalda.

Modelo de embeddings: el corpus está indexado con vectores de 1536 dimensiones
(`text-embedding-3-small`). Cambiarlo obliga a reindexar las 120 filas.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENAI_AVAILABLE = False
    logger.warning("openai no instalado — uv pip install openai")

try:
    import psycopg
    from psycopg.rows import dict_row
    PSYCOPG_AVAILABLE = True
except ImportError:  # pragma: no cover
    PSYCOPG_AVAILABLE = False
    logger.warning("psycopg no instalado — uv pip install 'psycopg[binary]'")


# ── Constantes ────────────────────────────────────────────────────────────────

TABLE = "zartha_chunks"
RPC_FUNCTION = "match_zartha_chunks"

# Dimensión con la que está indexado el corpus. No es configurable: cambiarla
# sin reindexar produce un error de pgvector, que es preferible a un resultado
# silenciosamente incorrecto.
EMBEDDING_DIMS = 1536
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Coseno sobre este corpus (español + inglés, fragmentos de ~1-2 párrafos):
# por debajo de ~0.25 lo recuperado ya no habla del tema de la consulta.
DEFAULT_MIN_SIMILARITY = 0.25
DEFAULT_MATCH_COUNT = 6

# Vocabulario controlado de `temas` tal como está poblado en la tabla.
TEMAS = (
    "ciclo_vida",
    "curvas_s",
    "delphi",
    "innovacion",
    "matriz",
    "prospectiva",
    "tecnologias_emergentes",
    "vigilancia_tecnologica",
)


class RagError(RuntimeError):
    """Fallo recuperable de la capa de recuperación."""


# ── Modelo de datos ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Chunk:
    """Un fragmento recuperado, con todo lo necesario para citarlo."""

    id: int
    cite_key: str
    title: str
    year: Optional[int]
    source: Optional[str]
    url: Optional[str]
    temas: tuple[str, ...] = field(default_factory=tuple)
    chunk_index: int = 0
    content: str = ""
    similarity: float = 0.0

    @property
    def citation(self) -> str:
        """Etiqueta tal como aparece en el informe: `[Zartha et al. 2016]`."""
        return f"[{self.cite_key}]"

    def reference(self) -> dict[str, Any]:
        """Entrada del bloque «Referencias del corpus»."""
        return {
            "cite_key": self.cite_key,
            "title": self.title,
            "year": self.year,
            "source": self.source,
            "url": self.url,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cite_key": self.cite_key,
            "title": self.title,
            "year": self.year,
            "source": self.source,
            "url": self.url,
            "temas": list(self.temas),
            "chunk_index": self.chunk_index,
            "content": self.content,
            "similarity": round(self.similarity, 4),
        }


# ── Cliente de embeddings ─────────────────────────────────────────────────────

def _openai_client() -> "OpenAI":
    if not OPENAI_AVAILABLE:
        raise RagError("openai no instalado — uv pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RagError("OPENAI_API_KEY no configurada")
    return OpenAI(api_key=api_key)


def embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embed(texts: Sequence[str]) -> list[list[float]]:
    """
    Embebe varios textos en UNA sola llamada a la API.

    Batchear importa: el informe lanza seis sondas de recuperación y hacerlas
    en una llamada en vez de seis es la diferencia entre 1 y 6 peticiones.
    """
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return []

    response = _openai_client().embeddings.create(
        model=embedding_model(),
        input=list(cleaned),
    )
    vectors = [item.embedding for item in response.data]

    for vec in vectors:
        if len(vec) != EMBEDDING_DIMS:
            raise RagError(
                f"El modelo de embeddings devolvió {len(vec)} dimensiones; el "
                f"corpus está indexado a {EMBEDDING_DIMS}. Reindexa "
                f"{TABLE} o fija OPENAI_EMBEDDING_MODEL={DEFAULT_EMBEDDING_MODEL}."
            )
    return vectors


def embed_one(text: str) -> list[float]:
    vectors = embed([text])
    if not vectors:
        raise RagError("No se puede embeber una consulta vacía")
    return vectors[0]


def _vector_literal(vector: Sequence[float]) -> str:
    """pgvector acepta el literal textual `[0.1,0.2,…]`; evita una dependencia."""
    return "[" + ",".join(f"{float(v):.8f}" for v in vector) + "]"


# ── Conexión ──────────────────────────────────────────────────────────────────

def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RagError("DATABASE_URL no configurada")
    return url


def _connect():
    if not PSYCOPG_AVAILABLE:
        raise RagError("psycopg no instalado — uv pip install 'psycopg[binary]'")
    return psycopg.connect(_database_url(), connect_timeout=20, row_factory=dict_row)


_RPC_SQL = f"""
    select id, cite_key, title, year, source, url, temas,
           chunk_index, content, similarity
    from public.{RPC_FUNCTION}(%(embedding)s::vector, %(k)s, %(tema)s)
"""

# Equivalente en línea, por si la función no existe todavía en el entorno.
_INLINE_SQL = f"""
    select c.id, c.cite_key, c.title, c.year, c.source, c.url, c.temas,
           c.chunk_index, c.content,
           1 - (c.embedding <=> %(embedding)s::vector) as similarity
    from public.{TABLE} c
    where %(tema)s::text is null or %(tema)s = any(c.temas)
    order by c.embedding <=> %(embedding)s::vector
    limit %(k)s
"""


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        id=int(row["id"]),
        cite_key=row["cite_key"] or "sin_clave",
        title=row["title"] or "",
        year=row["year"],
        source=row["source"],
        url=row["url"],
        temas=tuple(row.get("temas") or ()),
        chunk_index=int(row.get("chunk_index") or 0),
        content=row.get("content") or "",
        similarity=float(row.get("similarity") or 0.0),
    )


# ── Búsqueda ──────────────────────────────────────────────────────────────────

def search_by_vector(
    vector: Sequence[float],
    k: int = DEFAULT_MATCH_COUNT,
    tema: Optional[str] = None,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> list[Chunk]:
    """Búsqueda por similitud coseno con un embedding ya calculado."""
    if tema is not None and tema not in TEMAS:
        raise RagError(f"Tema desconocido «{tema}». Válidos: {', '.join(TEMAS)}")

    params = {"embedding": _vector_literal(vector), "k": max(1, int(k)), "tema": tema}

    with _connect() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(_RPC_SQL, params)
            except psycopg.errors.UndefinedFunction:
                # La migración de la función no está aplicada en este entorno.
                conn.rollback()
                logger.warning(
                    "%s no existe; usando la consulta en línea equivalente",
                    RPC_FUNCTION,
                )
                cur.execute(_INLINE_SQL, params)
            rows = cur.fetchall()

    chunks = [_row_to_chunk(r) for r in rows]
    return [c for c in chunks if c.similarity >= min_similarity]


def search(
    query: str,
    k: int = DEFAULT_MATCH_COUNT,
    tema: Optional[str] = None,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> list[Chunk]:
    """Embebe la consulta y recupera los `k` fragmentos más próximos."""
    if not query or not query.strip():
        return []
    return search_by_vector(embed_one(query), k=k, tema=tema, min_similarity=min_similarity)


def search_multi(
    queries: Sequence[str],
    k_per_query: int = 4,
    tema: Optional[str] = None,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    max_chunks: int = 14,
) -> list[Chunk]:
    """
    Varias sondas en una sola llamada de embeddings, resultados deduplicados.

    El informe necesita cubrir temas distintos (curvas en S, hype cycle,
    Delphi, limitaciones), y una única consulta larga los promedia a todos y no
    recupera bien ninguno. Se lanza una sonda por sección y se fusiona
    quedándose, para cada fragmento repetido, con su mejor similitud.
    """
    probes = [q.strip() for q in queries if q and q.strip()]
    if not probes:
        return []

    vectors = embed(probes)
    best: dict[int, Chunk] = {}
    for vector in vectors:
        for chunk in search_by_vector(
            vector, k=k_per_query, tema=tema, min_similarity=min_similarity
        ):
            previous = best.get(chunk.id)
            if previous is None or chunk.similarity > previous.similarity:
                best[chunk.id] = chunk

    ordered = sorted(best.values(), key=lambda c: c.similarity, reverse=True)
    return ordered[:max_chunks]


# ── Contexto y referencias ────────────────────────────────────────────────────

def build_context(chunks: Iterable[Chunk], max_chars_per_chunk: int = 1400) -> str:
    """
    Serializa los fragmentos para el prompt.

    Cada bloque lleva su `cite_key` delante: es lo único que el generador puede
    usar como etiqueta de cita, y la comprobación posterior se hace contra esta
    misma lista.
    """
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        body = c.content.strip().replace("\n", " ")
        if len(body) > max_chars_per_chunk:
            body = body[:max_chars_per_chunk].rsplit(" ", 1)[0] + "…"
        head = f"[{i}] cite_key: {c.cite_key} · {c.title}"
        if c.year:
            head += f" ({c.year})"
        head += f" · similitud {c.similarity:.3f}"
        blocks.append(f"{head}\n{body}")
    return "\n\n".join(blocks)


def collect_references(chunks: Iterable[Chunk]) -> list[dict[str, Any]]:
    """Una entrada por documento (no por fragmento), ordenada por cite_key."""
    by_key: dict[str, dict[str, Any]] = {}
    for c in chunks:
        by_key.setdefault(c.cite_key, c.reference())
    return sorted(by_key.values(), key=lambda r: r["cite_key"])


def available_cite_keys(chunks: Iterable[Chunk]) -> set[str]:
    """Las únicas claves que el generador tiene derecho a citar."""
    return {c.cite_key for c in chunks}


# ── Diagnóstico ───────────────────────────────────────────────────────────────

def corpus_status() -> dict[str, Any]:
    """Estado del corpus: filas, documentos, dimensión, índices y función RPC."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"select count(*) as n from public.{TABLE}")
            rows = cur.fetchone()["n"]

            cur.execute(
                f"select cite_key, title, year, source, url, count(*) as chunks "
                f"from public.{TABLE} group by 1,2,3,4,5 order by cite_key"
            )
            documents = cur.fetchall()

            cur.execute(
                f"select distinct vector_dims(embedding) as dims from public.{TABLE}"
            )
            dims = [r["dims"] for r in cur.fetchall()]

            cur.execute(
                "select indexname from pg_indexes "
                "where schemaname = 'public' and tablename = %s",
                (TABLE,),
            )
            indexes = [r["indexname"] for r in cur.fetchall()]

            cur.execute("select extname from pg_extension where extname = 'vector'")
            pgvector = cur.fetchone() is not None

            cur.execute(
                "select 1 from pg_proc p join pg_namespace n on n.oid = p.pronamespace "
                "where n.nspname = 'public' and p.proname = %s",
                (RPC_FUNCTION,),
            )
            rpc = cur.fetchone() is not None

    return {
        "table": TABLE,
        "chunks": rows,
        "documents": [dict(d) for d in documents],
        "embedding_dims": dims,
        "indexes": indexes,
        "pgvector": pgvector,
        "rpc_function": RPC_FUNCTION if rpc else None,
        "embedding_model": embedding_model(),
        "temas": list(TEMAS),
    }
