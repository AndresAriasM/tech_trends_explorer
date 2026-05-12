"""
S-Curve routes.
POST /api/s-curve/analyze  → SSE stream with real-time progress + final result
GET  /api/s-curve/history  → list saved analyses (placeholder)
"""

import json
from typing import Optional, AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.s_curve import analyze_s_curve, analyze_from_yearly_data

router = APIRouter()


# ── Request model ─────────────────────────────────────────────

class SCurveRequest(BaseModel):
    query: str
    api_key: str
    year_from: int = 1980
    year_to: Optional[int] = None
    max_results: Optional[int] = None


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

    result = analyze_from_yearly_data(counts, title=title)
    return result


@router.get("/history")
async def get_history():
    """Return saved S-curve analyses (placeholder — wire to storage later)."""
    return []
