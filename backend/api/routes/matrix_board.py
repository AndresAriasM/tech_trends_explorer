"""
Tablero de la matriz patentes/papers — sesiones de taller.

GET    /api/matrix-board/sessions                    → sesiones con su número de tecnologías
POST   /api/matrix-board/sessions                    → crea una sesión ("Taller de hoy")
GET    /api/matrix-board/sessions/{id}               → sesión + tecnologías (para la gráfica)
DELETE /api/matrix-board/sessions/{id}               → borra la sesión y sus tecnologías
POST   /api/matrix-board/sessions/{id}/runs          → guarda una tecnología ya analizada
PATCH  /api/matrix-board/runs/{id}                   → renombra y/o mueve a otra sesión
DELETE /api/matrix-board/runs/{id}                   → quita una tecnología
POST   /api/matrix-board/sessions/{id}/analysis      → análisis de IA de la sesión (OpenAI)

Los análisis (Scopus, patentes, comparación) los hace la pantalla con los
endpoints de /api/s-curve; aquí sólo se guardan sus resultados, resumidos, y
se calcula la posición con la convención de años de inflexión (core/matrix_board.py).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterator, Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from api.routes.delphi import _connection
from core.matrix_board import QUADRANT_LABELS, QUADRANTS, analyze_session, place, summarize_series

logger = logging.getLogger(__name__)
router = APIRouter()


def get_conn() -> Iterator[Any]:
    with _connection() as conn:
        yield conn


class SessionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=2000)
    reference_year: int = Field(2025, ge=1990, le=2100)


class RunCreate(BaseModel):
    label: str = Field(..., min_length=1, max_length=200)
    scopus_query: str
    # Fuente de la curva científica: hoy Scopus; se deja abierta para OpenAlex,
    # Semantic Scholar o Lens sin cambiar el contrato.
    papers_source: Literal["scopus", "openalex", "semantic_scholar", "lens"] = "scopus"
    patents_query: str
    patent_source: Literal["epo", "google"] = "google"
    year_from: int
    year_to: int
    papers: dict[str, Any]
    patents: dict[str, Any]
    comparison: Optional[dict[str, Any]] = None


def _session(conn, sid: UUID) -> dict:
    row = conn.execute("select * from matrix_sessions where id = %s", (sid,)).fetchone()
    if not row:
        raise HTTPException(404, "Sesión no encontrada")
    return row


def _runs(conn, sid: UUID) -> list[dict]:
    rows = conn.execute(
        "select * from matrix_runs where session_id = %s order by created_at", (sid,)
    ).fetchall()
    for r in rows:
        for k in ("papers_inflection", "patents_inflection"):
            if r[k] is not None:
                r[k] = float(r[k])
        for k in ("papers_range", "patents_range"):
            if r[k] is not None:
                r[k] = [float(x) for x in r[k]]
        r["quadrant_label"] = QUADRANT_LABELS.get(r["quadrant"])
    return rows


@router.get("/sessions")
def list_sessions(conn=Depends(get_conn)):
    return conn.execute("""
        select s.id, s.name, s.description, s.reference_year, s.created_at,
               s.ai_generated_at, count(r.id)::int as n_runs
          from matrix_sessions s left join matrix_runs r on r.session_id = s.id
         group by s.id order by s.created_at desc
    """).fetchall()


@router.post("/sessions", status_code=201)
def create_session(body: SessionCreate, conn=Depends(get_conn)):
    row = conn.execute(
        "insert into matrix_sessions (name, description, reference_year) values (%s, %s, %s) returning *",
        (body.name.strip(), body.description, body.reference_year),
    ).fetchone()
    conn.commit()
    return row


@router.get("/sessions/{sid}")
def get_session(sid: UUID, conn=Depends(get_conn)):
    return {"session": _session(conn, sid), "runs": _runs(conn, sid)}


@router.delete("/sessions/{sid}", status_code=204)
def delete_session(sid: UUID, conn=Depends(get_conn)):
    _session(conn, sid)
    conn.execute("delete from matrix_sessions where id = %s", (sid,))
    conn.commit()


@router.post("/sessions/{sid}/runs", status_code=201)
def add_run(sid: UUID, body: RunCreate, conn=Depends(get_conn)):
    session = _session(conn, sid)
    papers = summarize_series(body.papers)
    patents = summarize_series(body.patents)
    pos = place(papers, patents, session["reference_year"])
    comparison = None
    if body.comparison:
        comparison = {k: body.comparison.get(k) for k in ("lag_years", "direction", "interpretation", "caveats")}
    row = conn.execute("""
        insert into matrix_runs (session_id, label, scopus_query, papers_source, patents_query,
            patent_source, year_from, year_to, papers, patents, comparison, papers_inflection,
            patents_inflection, papers_range, patents_range, quadrant, robust)
        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) returning id
    """, (
        sid, body.label.strip(), body.scopus_query, body.papers_source, body.patents_query,
        body.patent_source,
        body.year_from, body.year_to, Jsonb(papers), Jsonb(patents),
        Jsonb(comparison) if comparison else None,
        pos["papers_inflection"], pos["patents_inflection"], pos["papers_range"],
        pos["patents_range"], pos["quadrant"], pos["robust"],
    )).fetchone()
    conn.commit()
    return {"id": row["id"], **pos, "quadrant_label": QUADRANT_LABELS.get(pos["quadrant"])}


class RunUpdate(BaseModel):
    label: Optional[str] = Field(None, min_length=1, max_length=200)
    # Mover la tecnología a otra sesión del tablero.
    session_id: Optional[UUID] = None


def _quadrant_for(run: dict, ref_year: int) -> tuple[Optional[str], Optional[bool]]:
    """Cuadrante y robustez con el año de referencia de la sesión destino."""
    px, ty = run["papers_inflection"], run["patents_inflection"]
    if px is None or ty is None:
        return None, None
    px, ty = float(px), float(ty)
    quadrant = QUADRANTS[(px <= ref_year, ty <= ref_year)]
    ranges = [r for r in (run["papers_range"], run["patents_range"]) if r]
    robust = all((float(r[0]) > ref_year) == (float(r[1]) > ref_year) for r in ranges)
    return quadrant, robust


@router.patch("/runs/{rid}")
def update_run(rid: UUID, body: RunUpdate, conn=Depends(get_conn)):
    """
    Renombra la tecnología y/o la mueve a otra sesión. Al moverla, el
    cuadrante se recalcula con el año de referencia de la sesión destino: las
    inflexiones no cambian, pero el corte sí puede.
    """
    run = conn.execute("select * from matrix_runs where id = %s", (rid,)).fetchone()
    if not run:
        raise HTTPException(404, "Tecnología no encontrada")
    if body.label is None and body.session_id is None:
        raise HTTPException(400, "Nada que actualizar")

    if body.label is not None:
        conn.execute("update matrix_runs set label = %s where id = %s", (body.label.strip(), rid))

    if body.session_id is not None and body.session_id != run["session_id"]:
        target = _session(conn, body.session_id)
        quadrant, robust = _quadrant_for(run, target["reference_year"])
        conn.execute(
            "update matrix_runs set session_id = %s, quadrant = %s, robust = %s where id = %s",
            (body.session_id, quadrant, robust, rid),
        )

    conn.commit()
    row = conn.execute(
        "select id, label, session_id, quadrant, robust from matrix_runs where id = %s", (rid,)
    ).fetchone()
    row["quadrant_label"] = QUADRANT_LABELS.get(row["quadrant"])
    return row


@router.delete("/runs/{rid}", status_code=204)
def delete_run(rid: UUID, conn=Depends(get_conn)):
    conn.execute("delete from matrix_runs where id = %s", (rid,))
    conn.commit()


@router.post("/sessions/{sid}/analysis")
async def run_analysis(sid: UUID, conn=Depends(get_conn)):
    from core.assistant import AssistantError

    session = _session(conn, sid)
    runs = _runs(conn, sid)
    try:
        analysis = await asyncio.to_thread(analyze_session, session, runs)
    except AssistantError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("matrix_board: análisis IA falló: %s", exc)
        raise HTTPException(502, f"El proveedor de IA falló: {type(exc).__name__}: {str(exc)[:200]}")
    conn.execute(
        "update matrix_sessions set ai_analysis = %s, ai_generated_at = now() where id = %s",
        (Jsonb(analysis), sid),
    )
    conn.commit()
    return analysis
