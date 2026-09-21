"""
S-Curve routes.
POST /api/s-curve/analyze          → SSE: papers vía Scopus
POST /api/s-curve/analyze-local    → síncrono: serie ya agregada (CSV)
POST /api/s-curve/analyze-patents  → SSE: patentes vía EPO OPS
POST /api/s-curve/fit-models       → síncrono: tabla comparativa de 5 modelos
POST /api/s-curve/compare          → síncrono: desfase ciencia → tecnología
GET  /api/s-curve/epo/status       → diagnóstico de credenciales EPO
GET  /api/s-curve/history          → list saved analyses (placeholder)
"""

import json
from typing import Any, Optional, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.epo import check_credentials, credentials_configured
from core.s_curve import (
    analyze_from_yearly_data,
    analyze_patents_epo,
    analyze_s_curve,
    compare_papers_patents,
)
from core.s_curve_models import fit_all_models

router = APIRouter()


# ── Request models ────────────────────────────────────────────

class SCurveRequest(BaseModel):
    query: str
    api_key: str
    year_from: int = 1980
    year_to: Optional[int] = None
    max_results: Optional[int] = None


class PatentsEPORequest(BaseModel):
    """
    Consulta de patentes contra EPO OPS.

    No lleva api_key: a diferencia de Scopus —cuya clave viaja desde el
    navegador en cada request— las credenciales de OPS son del servidor
    (EPO_CONSUMER_KEY / EPO_API_KEY) y nunca salen de él.

    `query` admite un término suelto ("biodiesel") o CQL de OPS
    (`txt="biodiesel" and pa="basf"`); el cliente lo detecta y envuelve el
    término suelto en `txt="…"`.
    """
    query: str
    # OPS cobra una petición por año: arrancar en 1980 gasta 46 peticiones
    # para recoger casi sólo ceros. 2000 es un arranque razonable por defecto.
    year_from: int = 2000
    year_to: Optional[int] = None


class FitModelsRequest(BaseModel):
    yearly_counts: dict[str, int]
    cumulative: bool = False


class CompareRequest(BaseModel):
    """Los dos resultados completos tal como los devolvió cada análisis."""
    papers: dict[str, Any]
    patents: dict[str, Any]


# ── SSE helpers ───────────────────────────────────────────────

def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_analysis(payload: SCurveRequest) -> AsyncGenerator[str, None]:
    try:
        async for event in analyze_s_curve(
            query=payload.query,
            api_key=payload.api_key,
            year_from=payload.year_from,
            year_to=payload.year_to,
            max_results=payload.max_results,
        ):
            yield _sse_event(event)
    except Exception as exc:
        yield _sse_event({"status": "error", "message": str(exc)})


# ── Routes ────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze(payload: SCurveRequest):
    """
    POST body: { query, api_key, year_from?, year_to?, max_results? }
    Returns a text/event-stream of progress events.
    Final event: { status: "completed", result: {...} }
    """
    if not payload.query.strip():
        return {"error": "La consulta no puede estar vacía"}
    if not payload.api_key.strip():
        return {"error": "Se requiere una API key de Scopus"}

    return StreamingResponse(
        _stream_analysis(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/analyze-local")
async def analyze_local(payload: dict):
    """
    POST body: { yearly_counts: {year: count, ...}, title: str }
    Synchronous — runs sigmoid fit on pre-aggregated data.
    Returns the same result structure as the streaming analyze endpoint.
    """
    yearly_counts: dict = payload.get("yearly_counts", {})
    title: str = payload.get("title", "Análisis de patentes")

    if not yearly_counts:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="yearly_counts no puede estar vacío")

    # Convert keys to int (JSON keys are strings)
    counts = {int(k): int(v) for k, v in yearly_counts.items() if int(v) > 0}

    if not counts:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No hay datos con conteo > 0")

    # `source` permite que una serie cargada por CSV se etiquete como patentes
    # y entre en la comparación papers ↔ patentes igual que la de EPO OPS.
    source = str(payload.get("source") or "local")
    labels = {
        "patents": "patentes (CSV)",
        "scopus": "papers (Scopus)",
        "local": "serie local",
    }
    result = analyze_from_yearly_data(
        counts,
        title=title,
        source=source,
        source_label=payload.get("source_label") or labels.get(source),
    )
    return result


@router.post("/analyze-patents")
async def analyze_patents(payload: PatentsEPORequest):
    """
    Curva en S de patentes vía EPO OPS.

    Mismo stream de eventos que `/analyze`: `{status, progress, message}` y un
    `completed` final con `result`. El resultado tiene la misma forma que el
    de papers —incluida la tabla multi-modelo— para que la comparación entre
    ambos sea directa.
    """
    if not payload.query.strip():
        return {"error": "La consulta de patentes no puede estar vacía"}
    if not credentials_configured():
        return {
            "error": (
                "EPO OPS no está configurado en el servidor: faltan "
                "EPO_CONSUMER_KEY y/o EPO_API_KEY."
            )
        }

    async def _stream() -> AsyncGenerator[str, None]:
        try:
            async for event in analyze_patents_epo(
                query=payload.query.strip(),
                year_from=payload.year_from,
                year_to=payload.year_to,
            ):
                yield _sse_event(event)
        except Exception as exc:
            yield _sse_event({"status": "error", "message": str(exc)})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/epo/status")
async def epo_status():
    """
    ¿Está EPO OPS operativo? Autentica de verdad y lanza una búsqueda de
    prueba. Nunca devuelve fragmento alguno de las credenciales.
    """
    return await check_credentials()


@router.post("/fit-models")
async def fit_models(payload: FitModelsRequest):
    """
    Tabla comparativa de los cinco modelos sobre una serie ya agregada.

    Útil para reajustar sin volver a descargar. `cumulative: false` (por
    defecto) interpreta los valores como conteos anuales y acumula; `true`
    los toma ya acumulados.
    """
    counts = {int(k): int(v) for k, v in payload.yearly_counts.items()}
    if len(counts) < 5:
        raise HTTPException(
            status_code=400,
            detail="Se requieren al menos 5 años para comparar modelos de 3 parámetros",
        )

    years = list(range(min(counts), max(counts) + 1))
    values = [counts.get(y, 0) for y in years]
    if not payload.cumulative:
        running = 0
        acc = []
        for v in values:
            running += v
            acc.append(running)
        values = acc

    return fit_all_models(years, values)


@router.post("/compare")
async def compare(payload: CompareRequest):
    """
    Desfase ciencia → tecnología entre una curva de papers y una de patentes.

    Compara las inflexiones de los MEJORES modelos de cada serie, no las de
    las dos logísticas, y devuelve las salvedades (autocorrelación, rangos de
    inflexión solapados, coberturas temporales distintas) junto al número.
    """
    if not payload.papers or not payload.patents:
        raise HTTPException(
            status_code=400,
            detail="Se requieren los dos resultados: papers y patents",
        )
    return compare_papers_patents(payload.papers, payload.patents)


@router.get("/history")
async def get_history():
    """Return saved S-curve analyses (placeholder — wire to storage later)."""
    return []
