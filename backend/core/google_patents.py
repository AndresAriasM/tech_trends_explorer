"""
backend/core/google_patents.py

Curva en S de patentes a partir de Google Patents (vía SerpAPI), como
alternativa a EPO OPS.

Por qué (prueba de octubre de 2026):

  · EPO OPS bloquea por "RobotDetected" en cuanto hay peticiones en paralelo,
    y su conteo anual tarda ~11 min por tecnología a ritmo seguro.
  · `engine=google_patents` devuelve `search_information.total_results` con
    filtros de fecha `after`/`before` en ~1 s por llamada (un crédito).

Lo que cambia frente a EPO y hay que decir al comparar:

  · Google Patents busca por defecto en el TEXTO COMPLETO (reivindicaciones y
    descripción), no sólo en título y resumen.
  · Cuenta más oficinas y tipos de documento: para `CPC=G06V40/16/low` en 2020
    dio 34 071 frente a 11 027 de EPO. Las dos series NO se mezclan.
  · Sintaxis: `CPC=<clase>/low` incluye subgrupos; sin `/low`, sólo la clase.

Para ahorrar créditos, los años antes de `split_year` se piden en UNA sola
consulta y entran como acumulado inicial: la curva acumulada queda completa
aunque esos años no tengan desglose anual.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search.json"


def total_results(
    query: str,
    date_from: str,
    date_to: str,
    api_key: Optional[str] = None,
    date_type: str = "publication",
    retries: int = 3,
) -> int:
    """Total de publicaciones de patente entre dos fechas YYYYMMDD. Un crédito."""
    params = {
        "engine": "google_patents",
        "q": query,
        "after": f"{date_type}:{date_from}",
        "before": f"{date_type}:{date_to}",
        "num": 10,
        "api_key": api_key or os.getenv("SERP_API_KEY", ""),
    }
    for attempt in range(retries):
        try:
            r = requests.get(SERPAPI_URL, params=params, timeout=90)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"SerpAPI Google Patents: {exc}") from exc
            time.sleep(2 ** attempt)
            continue
        if data.get("error"):
            # Sin resultados es un cero válido, no un fallo.
            if "hasn't returned any results" in data["error"]:
                return 0
            raise RuntimeError(f"SerpAPI Google Patents: {data['error']}")
        return int(data.get("search_information", {}).get("total_results") or 0)
    return 0


def yearly_counts(
    query: str,
    year_from: int = 1960,
    year_to: int = 2025,
    split_year: Optional[int] = 1985,
    api_key: Optional[str] = None,
    on_progress=None,
) -> dict:
    """
    Conteo anual. Devuelve {"base": acumulado antes de split_year,
    "base_range": [desde, hasta], "counts": {año: n}, "requests": k}.
    """
    counts: dict[int, int] = {}
    base = 0
    requests_made = 0
    first = year_from
    if split_year and split_year > year_from:
        base = total_results(query, f"{year_from}0101", f"{split_year - 1}1231", api_key)
        requests_made += 1
        first = split_year
        if on_progress:
            on_progress(f"{year_from}–{split_year - 1}: {base:,}")
    for year in range(first, year_to + 1):
        counts[year] = total_results(query, f"{year}0101", f"{year}1231", api_key)
        requests_made += 1
        if on_progress:
            on_progress(f"{year}: {counts[year]:,}")
    return {
        "base": base,
        "base_range": [year_from, first - 1] if base or first > year_from else None,
        "counts": counts,
        "requests": requests_made,
    }


async def analyze_patents_google(
    query: str,
    year_from: int = 2000,
    year_to: Optional[int] = None,
    split_year: Optional[int] = 1985,
):
    """
    Curva en S de patentes desde Google Patents con los mismos eventos SSE que
    `analyze_patents_epo`: `{status, progress, message}` y un `completed` final
    con `result`, para que la UI y la comparación con papers no cambien.
    """
    import asyncio
    from datetime import datetime

    import numpy as np

    from core.s_curve import _assemble_result

    if year_to is None:
        year_to = datetime.now().year
    if year_from > year_to:
        yield {"status": "error", "message": "El año inicial no puede ser posterior al final."}
        return
    if not os.getenv("SERP_API_KEY", "").strip():
        yield {"status": "error", "message": "El servidor no tiene configurada SERP_API_KEY."}
        return

    split = split_year if split_year and year_from < split_year <= year_to else None
    first = split or year_from
    years = list(range(first, year_to + 1))
    calls = len(years) + (1 if split else 0)
    yield {"status": "running", "progress": 3,
           "message": f"Google Patents: {calls} consultas ({calls} créditos de SerpAPI)…"}

    base = 0
    try:
        if split:
            base = await asyncio.to_thread(total_results, query, f"{year_from}0101", f"{split - 1}1231")
            yield {"status": "running", "progress": 6,
                   "message": f"Google Patents {year_from}–{split - 1}: {base:,} patentes (bloque inicial)"}
        counts: dict[int, int] = {}
        for i, y in enumerate(years, start=1):
            counts[y] = await asyncio.to_thread(total_results, query, f"{y}0101", f"{y}1231")
            yield {"status": "running", "progress": min(6 + int(i / len(years) * 82), 88),
                   "message": f"Google Patents {y}: {counts[y]:,} patentes ({i}/{len(years)} años)"}
    except RuntimeError as exc:
        yield {"status": "error", "message": str(exc)}
        return

    annual = [int(counts[y]) for y in years]
    if sum(annual) + base == 0:
        yield {"status": "error", "message": "Google Patents no encontró patentes con esta consulta."}
        return

    yield {"status": "running", "progress": 90, "message": "Ajustando los cinco modelos…"}
    total = sum(annual) + base
    result = _assemble_result(
        query=query,
        years=years,
        annual=annual,
        cumulative=[int(v) + base for v in np.cumsum(annual)],
        total_reported=total,
        total_retrieved=total,
        source="google_patents",
        source_label="patentes (Google Patents)",
        extra={"gpatents_requests": calls, "gpatents_base": base,
               "gpatents_base_range": [year_from, split - 1] if split else None},
    )
    yield {"status": "completed", "progress": 100,
           "message": f"{total:,} patentes en {len(years)} años ({calls} consultas a Google Patents).",
           "result": result}
