"""
backend/api/routes/assistant.py

Endpoints de la capa transversal de IA.

POST /api/assistant/search-strategy  → ecuaciones Scopus/EPO/Hype + justificación
POST /api/assistant/report           → informe prospectivo con RAG sobre el corpus
POST /api/assistant/corpus/search    → búsqueda directa en el corpus (auditoría)
GET  /api/assistant/corpus/status    → estado del corpus: filas, docs, índices, RPC

NOTA DELIBERADA: este router NO está registrado en `backend/main.py`. El
asistente es una capa que cada módulo invoca, y el punto de montaje se decide
cuando se integre con Curvas en S y Hype Cycle. Para activarlo:

    from api.routes import assistant
    app.include_router(assistant.router, prefix="/api/assistant", tags=["Asistente IA"])

Todas las respuestas de generación vienen marcadas con `is_ai_generated=True`,
por la misma razón que los votos sintéticos llevan `is_ai_vote=True`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core import rag
from core.assistant import (
    AssistantError,
    generate_foresight_report,
    generate_search_strategy,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Esquemas ──────────────────────────────────────────────────────────────────

class SearchStrategyRequest(BaseModel):
    concept: str = Field(..., min_length=2, max_length=200,
                         description="El concepto núcleo, p. ej. «biodiesel»")
    year_from: Optional[int] = Field(None, ge=1900, le=2100)
    year_to: Optional[int] = Field(None, ge=1900, le=2100)
    notes: str = Field("", max_length=2000,
                       description="Contexto del estudio que acote la búsqueda")
    sources: list[str] = Field(default_factory=lambda: ["scopus", "epo"])


class ReportRequest(BaseModel):
    concept: str = Field(..., min_length=2, max_length=200)
    # Se aceptan tal cual los devuelve `core/s_curve.py`; el asistente es
    # permisivo con la forma para no romperse cuando el módulo evolucione.
    s_curve_papers: Optional[dict[str, Any]] = None
    s_curve_patents: Optional[dict[str, Any]] = None
    hype_cycle: Optional[dict[str, Any]] = None
    strategy: Optional[dict[str, Any]] = None
    extra_context: Optional[str] = Field(None, max_length=4000)
    k_per_probe: int = Field(4, ge=1, le=10)
    max_chunks: int = Field(14, ge=1, le=40)
    min_similarity: float = Field(rag.DEFAULT_MIN_SIMILARITY, ge=0.0, le=1.0)


class CorpusSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    k: int = Field(rag.DEFAULT_MATCH_COUNT, ge=1, le=20)
    tema: Optional[str] = Field(None, description=f"Uno de: {', '.join(rag.TEMAS)}")
    min_similarity: float = Field(rag.DEFAULT_MIN_SIMILARITY, ge=0.0, le=1.0)


# ── Rutas ─────────────────────────────────────────────────────────────────────

@router.post("/search-strategy")
async def search_strategy(payload: SearchStrategyRequest) -> dict[str, Any]:
    """
    Genera la estrategia de búsqueda de un concepto.

    Coste: 1 llamada a OpenAI. La estimación de peticiones a las APIs se
    calcula en Python, no la produce el modelo.
    """
    try:
        return await asyncio.to_thread(
            generate_search_strategy,
            concept=payload.concept,
            year_from=payload.year_from,
            year_to=payload.year_to,
            notes=payload.notes,
            sources=tuple(payload.sources) or ("scopus", "epo"),
        )
    except AssistantError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo generando la estrategia para «%s»", payload.concept)
        raise HTTPException(status_code=502, detail=f"Error del proveedor de IA: {exc}") from exc


@router.post("/report")
async def report(payload: ReportRequest) -> dict[str, Any]:
    """
    Genera el informe prospectivo anclado al corpus metodológico.

    Coste: 2 llamadas a OpenAI (embeddings de las sondas + generación).
    Devuelve las secciones fijas, las referencias realmente citadas y, si el
    modelo intentó inventarse alguna cita, la lista de las que se retiraron.
    """
    try:
        return await asyncio.to_thread(
            generate_foresight_report,
            concept=payload.concept,
            s_curve_papers=payload.s_curve_papers,
            s_curve_patents=payload.s_curve_patents,
            hype_cycle=payload.hype_cycle,
            strategy=payload.strategy,
            extra_context=payload.extra_context,
            k_per_probe=payload.k_per_probe,
            max_chunks=payload.max_chunks,
            min_similarity=payload.min_similarity,
        )
    except (AssistantError, rag.RagError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo generando el informe de «%s»", payload.concept)
        raise HTTPException(status_code=502, detail=f"Error del proveedor de IA: {exc}") from exc


@router.post("/corpus/search")
async def corpus_search(payload: CorpusSearchRequest) -> dict[str, Any]:
    """
    Búsqueda directa en el corpus. Existe para poder auditar qué recupera el
    RAG sin tener que generar un informe entero. Coste: 1 llamada de embeddings.
    """
    try:
        chunks = await asyncio.to_thread(
            rag.search,
            query=payload.query,
            k=payload.k,
            tema=payload.tema,
            min_similarity=payload.min_similarity,
        )
    except rag.RagError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo en la búsqueda del corpus")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "query": payload.query,
        "tema": payload.tema,
        "min_similarity": payload.min_similarity,
        "count": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
        "references": rag.collect_references(chunks),
    }


@router.get("/corpus/status")
async def corpus_status(
    verbose: bool = Query(False, description="Incluye el listado de documentos"),
) -> dict[str, Any]:
    """Diagnóstico del corpus: filas, documentos, dimensión, índices y RPC."""
    try:
        status = await asyncio.to_thread(rag.corpus_status)
    except rag.RagError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo consultando el estado del corpus")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not verbose:
        status["documents"] = len(status["documents"])
    return status
