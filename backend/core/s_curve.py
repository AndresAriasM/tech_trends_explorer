"""
S-Curve Analysis Core
Fetches publications from Scopus API, aggregates by year,
fits a sigmoid curve, and detects the adoption phase.
No Streamlit dependencies — pure Python + scipy.
"""

import asyncio
import re
import time
from datetime import datetime
from typing import AsyncGenerator

import httpx
import numpy as np
from scipy.optimize import curve_fit


# ── Sigmoid ──────────────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray, L: float, x0: float, k: float) -> np.ndarray:
    """Logistic sigmoid: L / (1 + exp(-k*(x - x0)))"""
    return L / (1 + np.exp(-k * (x - x0)))


def fit_sigmoid(
    years: list[int], cumulative: list[int]
) -> dict | None:
    """
    Fit a sigmoid to cumulative publication data.
    Returns dict with L, x0, k and goodness-of-fit r2, or None on failure.
    """
    x = np.array(years, dtype=float)
    y = np.array(cumulative, dtype=float)

    if len(x) < 4:
        return None

    L_init = float(max(y)) * 1.3
    x0_init = float(np.median(x))
    k_init = 0.3

    try:
        popt, _ = curve_fit(
            sigmoid,
            x,
            y,
            p0=[L_init, x0_init, k_init],
            maxfev=20_000,
            bounds=([0, min(x) - 20, 0.001], [max(y) * 5, max(x) + 20, 5]),
        )
        L, x0, k = popt

        # R² goodness-of-fit
        y_pred = sigmoid(x, L, x0, k)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {"L": float(L), "x0": float(x0), "k": float(k), "r2": round(r2, 4)}
    except Exception:
        return None


def detect_phase(
    sigmoid_params: dict | None,
    cumulative: list[int],
    current_year: int,
) -> dict:
    """
    Classify the technology adoption phase based on the sigmoid fit.
    Returns phase name, color, description, and percentage of saturation.
    """
    if not sigmoid_params or not cumulative:
        return {
            "phase": "Sin datos",
            "color": "#6b7280",
            "description": "Datos insuficientes para determinar la fase",
            "saturation_pct": 0,
        }

    L = sigmoid_params["L"]
    x0 = sigmoid_params["x0"]
    last_cum = cumulative[-1]
    saturation_pct = round(last_cum / L * 100, 1) if L > 0 else 0

    if saturation_pct < 10:
        return {
            "phase": "Emergente",
            "color": "#6366f1",
            "description": "La tecnología está en etapas iniciales de adopción académica",
            "saturation_pct": saturation_pct,
        }
    elif saturation_pct < 40:
        return {
            "phase": "Crecimiento",
            "color": "#3b82f6",
            "description": "El interés académico está creciendo de forma sostenida",
            "saturation_pct": saturation_pct,
        }
    elif saturation_pct < 65:
        return {
            "phase": "Aceleración",
            "color": "#10b981",
            "description": "Punto de inflexión: la adopción se está acelerando al máximo",
            "saturation_pct": saturation_pct,
        }
    elif saturation_pct < 85:
        return {
            "phase": "Maduración",
            "color": "#f59e0b",
            "description": "El crecimiento se está desacelerando hacia la saturación",
            "saturation_pct": saturation_pct,
        }
    else:
        return {
            "phase": "Saturación",
            "color": "#ef4444",
            "description": "La tecnología ha alcanzado madurez en la literatura científica",
            "saturation_pct": saturation_pct,
        }


# ── Local (CSV) analysis ─────────────────────────────────────────────────────

def analyze_from_yearly_data(
    yearly_counts: dict[int, int],
    title: str = "Análisis",
) -> dict:
    """
    Run sigmoid fit + phase detection on pre-aggregated yearly data.
    Returns the same result dict structure as analyze_s_curve.
    """
    # Fill every year in the range (years with no publications → 0 annual, but
    # the cumulative still carries forward — critical for a correct sigmoid fit)
    min_yr = min(yearly_counts.keys())
    max_yr = max(yearly_counts.keys())
    years_sorted = list(range(min_yr, max_yr + 1))
    annual = [int(yearly_counts.get(y, 0)) for y in years_sorted]
    cumulative = [int(v) for v in np.cumsum(annual)]

    sig_params = fit_sigmoid(years_sorted, cumulative)

    sigmoid_curve: list[dict] = []
    if sig_params:
        x_fine = np.linspace(min(years_sorted), max(years_sorted) + 10, 200)
        y_fine = sigmoid(x_fine, sig_params["L"], sig_params["x0"], sig_params["k"])
        sigmoid_curve = [
            {"year": round(float(x), 2), "value": round(float(y), 2)}
            for x, y in zip(x_fine, y_fine)
        ]

    phase_info = detect_phase(sig_params, cumulative, datetime.now().year)

    yearly_data = [
        {"year": int(y), "annual": int(a), "cumulative": int(c)}
        for y, a, c in zip(years_sorted, annual, cumulative)
    ]

    return {
        "query": title,
        "total_publications": int(sum(annual)),
        "year_from": int(min(years_sorted)),
        "year_to": int(max(years_sorted)),
        "yearly_data": yearly_data,
        "sigmoid_params": sig_params,
        "sigmoid_curve": sigmoid_curve,
        "phase": phase_info,
        "inflection_year": int(round(sig_params["x0"])) if sig_params else None,
        "saturation_level": int(round(sig_params["L"])) if sig_params else None,
        "r2": sig_params["r2"] if sig_params else None,
        "analyzed_at": datetime.now().isoformat(),
    }


# ── Scopus fetcher ────────────────────────────────────────────────────────────

SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"

PAGE_SIZE = 25


async def _fetch_page(
    client: httpx.AsyncClient,
    headers: dict,
    params: dict,
    retries: int = 3,
) -> dict | None:
    """Fetch a single Scopus results page with retry logic."""
    for attempt in range(retries):
        try:
            r = await client.get(SCOPUS_URL, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:  # rate limit
                await asyncio.sleep(2 ** attempt)
            else:
                return None
        except Exception:
            await asyncio.sleep(1)
    return None


def _extract_year(entry: dict) -> int | None:
    """Try multiple Scopus fields to extract publication year."""
    for field in ("prism:coverDate", "prism:publicationDate"):
        raw = entry.get(field)
        if raw:
            try:
                return datetime.strptime(raw[:10], "%Y-%m-%d").year
            except Exception:
                pass

    raw = entry.get("prism:coverDisplayDate", "")
    m = re.search(r"\b(19|20)\d{2}\b", raw)
    if m:
        return int(m.group(0))

    return None


async def analyze_s_curve(
    query: str,
    api_key: str,
    year_from: int = 1980,
    year_to: int | None = None,
    max_results: int | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that:
    1. Validates the Scopus query
    2. Fetches publications page by page
    3. Yields progress events  {"status", "progress", "message"}
    4. Yields final event      {"status": "completed", "result": {...}}
    5. Or error event          {"status": "error", "message": "..."}
    """
    if year_to is None:
        year_to = datetime.now().year

    # Exclude patents, apply year range
    full_query = f"({query}) AND NOT DOCTYPE(pt) AND PUBYEAR > {year_from - 1} AND PUBYEAR < {year_to + 1}"

    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json",
    }

    async with httpx.AsyncClient() as client:
        # ── Step 1: Verify query ──────────────────────────────
        yield {"status": "running", "progress": 5, "message": "Verificando consulta con Scopus…"}

        probe = await _fetch_page(client, headers, {"query": full_query, "count": 1})
        if probe is None:
            yield {"status": "error", "message": "No se pudo conectar con la API de Scopus. Verifica tu API key."}
            return

        search_results = probe.get("search-results", {})
        total = int(search_results.get("opensearch:totalResults", 0))

        if total == 0:
            yield {"status": "error", "message": "No se encontraron publicaciones con esta consulta."}
            return

        yield {
            "status": "running",
            "progress": 10,
            "message": f"Encontradas {total:,} publicaciones. Descargando datos…",
        }

        # ── Step 2: Paginate ──────────────────────────────────
        limit = total if max_results is None else min(total, max_results)
        all_entries: list[dict] = []
        start = 0

        while start < limit:
            batch_params = {
                "query": full_query,
                "count": PAGE_SIZE,
                "start": start,
                "sort": "coverDate",
            }
            page_data = await _fetch_page(client, headers, batch_params)
            if page_data is None:
                break

            entries = page_data.get("search-results", {}).get("entry", [])
            if not entries:
                break

            all_entries.extend(entries)
            start += PAGE_SIZE

            progress = 10 + int((len(all_entries) / limit) * 70)
            yield {
                "status": "running",
                "progress": min(progress, 80),
                "message": f"Descargadas {len(all_entries):,} de {limit:,} publicaciones…",
            }

            if start < limit:
                await asyncio.sleep(0.3)  # polite rate limiting

    # ── Step 3: Aggregate by year ─────────────────────────────
    yield {"status": "running", "progress": 82, "message": "Agregando publicaciones por año…"}

    pubs_by_year: dict[int, int] = {}
    for entry in all_entries:
        yr = _extract_year(entry)
        if yr:
            pubs_by_year[yr] = pubs_by_year.get(yr, 0) + 1

    if not pubs_by_year:
        yield {"status": "error", "message": "No se pudieron extraer fechas de las publicaciones."}
        return

    # Fill the full year range so gaps (years with 0 pubs) are explicit —
    # the cumulative must be monotonically non-decreasing for a valid sigmoid fit
    min_yr = min(pubs_by_year.keys())
    max_yr = max(pubs_by_year.keys())
    years_sorted = list(range(min_yr, max_yr + 1))
    annual = [int(pubs_by_year.get(y, 0)) for y in years_sorted]
    cumulative = [int(v) for v in np.cumsum(annual)]

    # ── Step 4: Sigmoid fit ───────────────────────────────────
    yield {"status": "running", "progress": 90, "message": "Ajustando curva sigmoidal…"}

    sig_params = fit_sigmoid(years_sorted, cumulative)

    # Build sigmoid curve for smooth rendering
    sigmoid_curve: list[dict] = []
    if sig_params:
        x_fine = np.linspace(min(years_sorted), max(years_sorted) + 10, 200)
        y_fine = sigmoid(x_fine, sig_params["L"], sig_params["x0"], sig_params["k"])
        sigmoid_curve = [
            {"year": round(float(x), 2), "value": round(float(y), 2)}
            for x, y in zip(x_fine, y_fine)
        ]

    # ── Step 5: Phase detection ───────────────────────────────
    phase_info = detect_phase(sig_params, cumulative, datetime.now().year)

    yield {"status": "running", "progress": 96, "message": "Calculando métricas finales…"}

    # Yearly table — all native Python int/float for JSON serialization
    yearly_data = [
        {"year": int(y), "annual": int(a), "cumulative": int(c)}
        for y, a, c in zip(years_sorted, annual, cumulative)
    ]

    result = {
        "query": query,
        "total_publications": int(len(all_entries)),
        "year_from": int(min(years_sorted)),
        "year_to": int(max(years_sorted)),
        "yearly_data": yearly_data,
        "sigmoid_params": sig_params,
        "sigmoid_curve": sigmoid_curve,
        "phase": phase_info,
        "inflection_year": int(round(sig_params["x0"])) if sig_params else None,
        "saturation_level": int(round(sig_params["L"])) if sig_params else None,
        "r2": sig_params["r2"] if sig_params else None,
        "analyzed_at": datetime.now().isoformat(),
    }

    yield {"status": "completed", "progress": 100, "message": "Análisis completado.", "result": result}
