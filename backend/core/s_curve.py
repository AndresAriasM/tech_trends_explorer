"""
S-Curve Analysis Core
Fetches publications from Scopus API, aggregates by year,
fits a sigmoid curve, and detects the adoption phase.
No Streamlit dependencies — pure Python + scipy.

Ver docs/LOGICA_CURVA_S.md para el linaje de esta lógica y el porqué de
cada decisión. Lo recuperado del notebook original (G1) va marcado con §.
"""

import asyncio
import re
import time
from datetime import datetime
from typing import AsyncGenerator

import httpx
import numpy as np
from scipy.optimize import curve_fit

# Scopus no pagina más allá de este tope por consulta (§8.3).
SCOPUS_MAX_RESULTS = 5000

# Un valor t por encima de 2 ≈ significativo al 95 %.
T_SIGNIFICANCE = 2.0

# Error relativo máximo de L para considerarlo utilizable como techo de
# saturación. Por encima, el porcentaje de saturación no es reportable.
L_MAX_REL_ERROR = 0.20


# ── Sigmoid ──────────────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray, L: float, x0: float, k: float) -> np.ndarray:
    """Logistic sigmoid: L / (1 + exp(-k*(x - x0)))"""
    return L / (1 + np.exp(-k * (x - x0)))


def sigmoid_derivative(x: np.ndarray, L: float, x0: float, k: float) -> np.ndarray:
    """
    Velocidad de adopción: derivada analítica de la logística (§4.5).
    Su máximo está exactamente en x0 y vale k·L/4.
    """
    e = np.exp(-k * (x - x0))
    return (k * L * e) / (1 + e) ** 2


def fit_sigmoid(years: list[int], cumulative: list[int]) -> dict | None:
    """
    Ajusta una sigmoide al acumulado.

    Devuelve L, x0, k con sus errores estándar y valores t, además de R² y
    RMSE. Sin la banda de error de L la fase es una cifra inventada (§4.4):
    en una tecnología pre-inflexión L no es identificable aunque R² dé 0.99.
    """
    x = np.array(years, dtype=float)
    y = np.array(cumulative, dtype=float)

    if len(x) < 4:
        return None

    L_init = float(max(y)) * 1.3
    x0_init = float(np.median(x))
    k_init = 0.3

    try:
        popt, pcov = curve_fit(
            sigmoid,
            x,
            y,
            p0=[L_init, x0_init, k_init],
            maxfev=20_000,
            bounds=([0, min(x) - 20, 0.001], [max(y) * 5, max(x) + 20, 5]),
        )
        L, x0, k = popt

        y_pred = sigmoid(x, L, x0, k)
        residuals = y - y_pred

        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # RMSE en unidades de publicaciones. Sobre una serie acumulada el R²
        # es engañosamente alto —una recta le saca 0.98 a casi cualquier
        # monótona—, así que el error absoluto dice bastante más.
        rmse = float(np.sqrt(np.mean(residuals**2)))

        # Errores estándar desde la matriz de covarianza.
        with np.errstate(invalid="ignore"):
            perr = np.sqrt(np.diag(pcov))
        L_se, x0_se, k_se = (float(v) for v in perr)

        def t_of(value: float, err: float) -> float | None:
            if not np.isfinite(err) or err == 0:
                return None
            return float(value / err)

        # Ojo: para x0 el valor t NO es informativo — contrasta contra
        # "la inflexión ocurrió en el año 0" y siempre sale significativo.
        # Para x0 lo que vale es el intervalo de confianza (§4.4).
        t_L = t_of(L, L_se)
        t_k = t_of(k, k_se)

        L_significant = bool(t_L is not None and abs(t_L) > T_SIGNIFICANCE)
        k_significant = bool(t_k is not None and abs(t_k) > T_SIGNIFICANCE)

        x0_ci = None
        if np.isfinite(x0_se):
            x0_ci = [round(float(x0 - 1.96 * x0_se), 2), round(float(x0 + 1.96 * x0_se), 2)]

        # El valor t de L contrasta contra "L = 0", que con cualquier serie no
        # vacía es trivialmente falso: sale significativo casi siempre, incluso
        # con L = 2835 ± 643. Para decidir si el techo es utilizable lo que
        # importa es su PRECISIÓN, no que difiera de cero. Tres condiciones:
        #   · el t-test clásico (lo mínimo),
        #   · error relativo acotado,
        #   · inflexión dentro del rango observado — si x0 cae más allá del
        #     último dato, L se está extrapolando y no está medido.
        x0_extrapolated = bool(x0 < min(x) or x0 > max(x))
        L_rel_error = float(L_se / L) if (np.isfinite(L_se) and L > 0) else None
        L_identifiable = bool(
            L_significant
            and L_rel_error is not None
            and L_rel_error <= L_MAX_REL_ERROR
            and not x0_extrapolated
        )

        return {
            "L": float(L),
            "x0": float(x0),
            "k": float(k),
            "r2": round(r2, 4),
            "rmse": round(rmse, 3),
            "L_stderr": round(L_se, 3) if np.isfinite(L_se) else None,
            "x0_stderr": round(x0_se, 3) if np.isfinite(x0_se) else None,
            "k_stderr": round(k_se, 4) if np.isfinite(k_se) else None,
            "t_L": round(t_L, 2) if t_L is not None else None,
            "t_k": round(t_k, 2) if t_k is not None else None,
            "L_significant": L_significant,
            "k_significant": k_significant,
            "L_rel_error": round(L_rel_error, 4) if L_rel_error is not None else None,
            # Criterio que gobierna la fase: L_significant por sí solo no basta.
            "L_identifiable": L_identifiable,
            "x0_ci95": x0_ci,
            # x0 fuera del rango observado es extrapolación, no ajuste.
            "x0_extrapolated": x0_extrapolated,
        }
    except Exception:
        return None


def detect_phase(
    sigmoid_params: dict | None,
    cumulative: list[int],
    years: list[int] | None = None,
    current_year: int | None = None,
) -> dict:
    """
    Clasifica la fase de adopción en DOS ejes (§5):

      · saturación — cuánto del techo L se ha recorrido
      · posición temporal — si el último dato va antes o después de x0

    El segundo eje importa porque el primero depende enteramente de L: si L
    no es identificable, el porcentaje de saturación no significa nada y la
    fase se determina sólo por la posición frente a la inflexión.
    """
    if not sigmoid_params or not cumulative:
        return {
            "phase": "Sin datos",
            "color": "#6b7280",
            "description": "Datos insuficientes para determinar la fase",
            "saturation_pct": None,
            "provisional": True,
            "basis": "none",
        }

    L = sigmoid_params["L"]
    x0 = sigmoid_params["x0"]
    last_cum = cumulative[-1]
    last_year = years[-1] if years else (current_year or datetime.now().year)
    before_inflection = last_year < x0

    # ── L no identificable: sólo el eje temporal ──────────────
    if not sigmoid_params.get("L_identifiable", True):
        if before_inflection:
            return {
                "phase": "Pre-inflexión",
                "color": "#6366f1",
                "description": (
                    "La adopción aún no alcanza su punto de máxima velocidad. "
                    "El techo de saturación no es estimable con los datos "
                    "disponibles, así que no se reporta porcentaje."
                ),
                "saturation_pct": None,
                "provisional": True,
                "basis": "temporal",
            }
        return {
            "phase": "Post-inflexión",
            "color": "#f59e0b",
            "description": (
                "La adopción ya pasó su punto de máxima velocidad, pero el "
                "techo de saturación no es estimable con los datos disponibles."
            ),
            "saturation_pct": None,
            "provisional": True,
            "basis": "temporal",
        }

    # ── L identificable: bandas de saturación ─────────────────
    saturation_pct = round(last_cum / L * 100, 1) if L > 0 else 0.0

    if saturation_pct < 10:
        phase, color, desc = (
            "Emergente",
            "#6366f1",
            "La tecnología está en etapas iniciales de adopción académica",
        )
    elif saturation_pct < 40:
        phase, color, desc = (
            "Crecimiento",
            "#3b82f6",
            "El interés académico está creciendo de forma sostenida",
        )
    elif saturation_pct < 65:
        phase, color, desc = (
            "Aceleración",
            "#10b981",
            "Punto de inflexión: la adopción se está acelerando al máximo",
        )
    elif saturation_pct < 85:
        phase, color, desc = (
            "Maduración",
            "#f59e0b",
            "El crecimiento se está desacelerando hacia la saturación",
        )
    else:
        phase, color, desc = (
            "Saturación",
            "#ef4444",
            "La tecnología ha alcanzado madurez en la literatura científica",
        )

    return {
        "phase": phase,
        "color": color,
        "description": desc,
        "saturation_pct": saturation_pct,
        "provisional": False,
        "basis": "saturation+temporal",
        "before_inflection": bool(before_inflection),
    }


# ── Derivadas del ajuste para la UI ──────────────────────────────────────────

def _build_curves(years: list[int], cumulative: list[int], params: dict | None) -> dict:
    """
    Construye las series derivadas del ajuste: curva suave, velocidad de
    adopción, residuos y proyecciones tabuladas.
    """
    out: dict = {
        "sigmoid_curve": [],
        "growth_curve": [],
        "residuals": [],
        "projections": [],
    }
    if not params:
        return out

    L, x0, k = params["L"], params["x0"], params["k"]
    y_last = max(years)

    x_fine = np.linspace(min(years), y_last + 10, 200)
    y_fine = sigmoid(x_fine, L, x0, k)
    out["sigmoid_curve"] = [
        {"year": round(float(a), 2), "value": round(float(b), 2)}
        for a, b in zip(x_fine, y_fine)
    ]

    # Velocidad de adopción: responde "¿cuándo se adopta más rápido?", que es
    # distinto de "¿cuánto se ha adoptado?" (§4.5).
    d_fine = sigmoid_derivative(x_fine, L, x0, k)
    out["growth_curve"] = [
        {"year": round(float(a), 2), "rate": round(float(b), 3)}
        for a, b in zip(x_fine, d_fine)
    ]
    out["peak_growth_year"] = round(float(x0), 1)
    out["peak_growth_rate"] = round(float(k * L / 4), 2)

    # Residuos por año: en un acumulado mal ajustado muestran estructura
    # sistemática que el R² oculta.
    y_obs = np.array(cumulative, dtype=float)
    y_hat = sigmoid(np.array(years, dtype=float), L, x0, k)
    out["residuals"] = [
        {"year": int(a), "residual": round(float(b), 2)}
        for a, b in zip(years, y_obs - y_hat)
    ]

    # Proyecciones: predicción, no ajuste. La UI debe dibujarlas punteadas.
    out["projections"] = [
        {"year": int(y_last + i), "value": round(float(sigmoid(np.array([y_last + i]), L, x0, k)[0]), 1)}
        for i in range(1, 6)
    ]
    return out


def _annual_growth_rate(annual: list[int], years: list[int]) -> list[dict]:
    """
    Variación interanual del FLUJO (publicaciones por año), no del stock.
    G2 la calculaba sobre el acumulado, que es otra cosa y menos informativa.
    """
    out: list[dict] = []
    for i in range(1, len(annual)):
        prev = annual[i - 1]
        if prev > 0:
            out.append(
                {"year": int(years[i]), "pct": round((annual[i] - prev) / prev * 100, 1)}
            )
        else:
            out.append({"year": int(years[i]), "pct": None})
    return out


def _assemble_result(
    *,
    query: str,
    years: list[int],
    annual: list[int],
    cumulative: list[int],
    total_reported: int | None,
    total_retrieved: int,
    entries_without_date: int = 0,
    truncated_windows: list[dict] | None = None,
) -> dict:
    """Arma el contrato de salida, común a la ruta Scopus y a la local."""
    params = fit_sigmoid(years, cumulative)
    curves = _build_curves(years, cumulative, params)
    phase_info = detect_phase(params, cumulative, years, datetime.now().year)

    return {
        "query": query,
        # total_publications se conserva por compatibilidad con el frontend;
        # total_reported / total_retrieved desambiguan (§8.5).
        "total_publications": int(total_retrieved),
        "total_reported": int(total_reported) if total_reported is not None else None,
        "total_retrieved": int(total_retrieved),
        "entries_without_date": int(entries_without_date),
        "truncated": bool(truncated_windows),
        "truncated_windows": truncated_windows or [],
        "year_from": int(min(years)),
        "year_to": int(max(years)),
        "yearly_data": [
            {"year": int(y), "annual": int(a), "cumulative": int(c)}
            for y, a, c in zip(years, annual, cumulative)
        ],
        "annual_growth_rate": _annual_growth_rate(annual, years),
        "sigmoid_params": params,
        "sigmoid_curve": curves["sigmoid_curve"],
        "growth_curve": curves["growth_curve"],
        "residuals": curves["residuals"],
        "projections": curves["projections"],
        "peak_growth_year": curves.get("peak_growth_year"),
        "peak_growth_rate": curves.get("peak_growth_rate"),
        "phase": phase_info,
        "inflection_year": int(round(params["x0"])) if params else None,
        "saturation_level": int(round(params["L"])) if params else None,
        "r2": params["r2"] if params else None,
        "analyzed_at": datetime.now().isoformat(),
    }


# ── Local (CSV) analysis ─────────────────────────────────────────────────────

def analyze_from_yearly_data(
    yearly_counts: dict[int, int],
    title: str = "Análisis",
) -> dict:
    """Ajuste sobre datos ya agregados por año (CSV de patentes)."""
    # Rellenar todos los años del rango: los huecos deben ser 0 explícito o el
    # eje X no es uniforme y el ajuste de k se distorsiona.
    min_yr, max_yr = min(yearly_counts), max(yearly_counts)
    years = list(range(min_yr, max_yr + 1))
    annual = [int(yearly_counts.get(y, 0)) for y in years]
    cumulative = [int(v) for v in np.cumsum(annual)]

    return _assemble_result(
        query=title,
        years=years,
        annual=annual,
        cumulative=cumulative,
        total_reported=sum(annual),
        total_retrieved=sum(annual),
    )


# ── Scopus fetcher ────────────────────────────────────────────────────────────

SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"

PAGE_SIZE = 25


async def _fetch_page(
    client: httpx.AsyncClient,
    headers: dict,
    params: dict,
    retries: int = 3,
) -> dict | None:
    """Una página de resultados, con reintentos y backoff exponencial."""
    for attempt in range(retries):
        try:
            r = await client.get(SCOPUS_URL, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:  # rate limit
                await asyncio.sleep(2**attempt)
            else:
                return None
        except Exception:
            await asyncio.sleep(1)
    return None


async def _probe_total(
    client: httpx.AsyncClient, headers: dict, query: str
) -> int | None:
    """Sondeo con count=1 para leer opensearch:totalResults."""
    data = await _fetch_page(client, headers, {"query": query, "count": 1})
    if data is None:
        return None
    try:
        return int(data.get("search-results", {}).get("opensearch:totalResults", 0))
    except (TypeError, ValueError):
        return None


def _window_query(base_query: str, y_from: int, y_to: int) -> str:
    """
    Ventana temporal con BEF/AFT, que son exclusivos — de ahí los ±1.
    Más limpio que `PUBYEAR > a-1 AND PUBYEAR < b+1`.
    """
    return (
        f"({base_query}) AND NOT DOCTYPE(pt) "
        f"AND PUBYEAR AFT {y_from - 1} AND PUBYEAR BEF {y_to + 1}"
    )


async def _plan_windows(
    client: httpx.AsyncClient,
    headers: dict,
    base_query: str,
    y_from: int,
    y_to: int,
    max_probes: int = 40,
) -> tuple[list[dict], int]:
    """
    Escalera ADAPTATIVA de ventanas temporales (§2, §9).

    Parte el rango por la mitad allí donde el conteo supera el tope de Scopus,
    en vez de usar una tabla de épocas fija. Devuelve las ventanas y el total
    reportado por Scopus para el rango completo.

    Una ventana de un solo año que aún supere el tope no puede partirse más:
    se marca como truncada para que el resultado lo declare.
    """
    windows: list[dict] = []
    probes = 0
    pending: list[tuple[int, int]] = [(y_from, y_to)]
    grand_total = 0

    while pending:
        a, b = pending.pop(0)
        if probes >= max_probes:
            # Sin presupuesto para seguir sondeando: se acepta la ventana tal cual.
            windows.append({"from": a, "to": b, "total": None, "truncated": False})
            continue

        total = await _probe_total(client, headers, _window_query(base_query, a, b))
        probes += 1

        if total is None:
            windows.append({"from": a, "to": b, "total": None, "truncated": False})
            continue
        if total == 0:
            continue

        if total <= SCOPUS_MAX_RESULTS:
            windows.append({"from": a, "to": b, "total": total, "truncated": False})
            grand_total += total
        elif a < b:
            mid = (a + b) // 2
            pending.append((a, mid))
            pending.append((mid + 1, b))
        else:
            # Un único año por encima del tope: no hay dónde partir.
            windows.append({"from": a, "to": b, "total": total, "truncated": True})
            grand_total += total

    windows.sort(key=lambda w: w["from"])
    return windows, grand_total


async def _extract_year(entry: dict) -> int | None:  # pragma: no cover - shim
    return _extract_year_sync(entry)


def _extract_year_sync(entry: dict) -> int | None:
    """Cascada de tres intentos para sacar el año de publicación."""
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
    Generador async que:
      1. Planifica ventanas temporales que respeten el tope de Scopus
      2. Descarga cada ventana paginando
      3. Emite progreso {"status", "progress", "message"}
      4. Emite {"status": "completed", "result": {...}}
      5. O {"status": "error", "message": "..."}
    """
    if year_to is None:
        year_to = datetime.now().year

    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}

    async with httpx.AsyncClient() as client:
        # ── 1. Sondeo del rango completo ──────────────────────
        yield {"status": "running", "progress": 3, "message": "Verificando consulta con Scopus…"}

        full_query = _window_query(query, year_from, year_to)
        total = await _probe_total(client, headers, full_query)

        if total is None:
            yield {
                "status": "error",
                "message": "No se pudo conectar con la API de Scopus. Verifica tu API key.",
            }
            return
        if total == 0:
            yield {"status": "error", "message": "No se encontraron publicaciones con esta consulta."}
            return

        # ── 2. Planificación de ventanas ──────────────────────
        if total > SCOPUS_MAX_RESULTS:
            yield {
                "status": "running",
                "progress": 6,
                "message": (
                    f"{total:,} publicaciones superan el tope de {SCOPUS_MAX_RESULTS:,} "
                    "por consulta. Dividiendo por periodos…"
                ),
            }
            windows, planned_total = await _plan_windows(
                client, headers, query, year_from, year_to
            )
            total_reported = max(total, planned_total)
        else:
            windows = [{"from": year_from, "to": year_to, "total": total, "truncated": False}]
            total_reported = total

        if not windows:
            yield {"status": "error", "message": "No se encontraron publicaciones con esta consulta."}
            return

        truncated_windows = [w for w in windows if w["truncated"]]
        expected = sum(w["total"] or 0 for w in windows) or total_reported
        if max_results is not None:
            expected = min(expected, max_results)

        yield {
            "status": "running",
            "progress": 10,
            "message": (
                f"Encontradas {total_reported:,} publicaciones en {len(windows)} "
                f"periodo(s). Descargando…"
            ),
        }

        # ── 3. Descarga ventana por ventana ───────────────────
        all_entries: list[dict] = []

        for w in windows:
            if max_results is not None and len(all_entries) >= max_results:
                break

            w_query = _window_query(query, w["from"], w["to"])
            w_limit = w["total"] if w["total"] is not None else SCOPUS_MAX_RESULTS
            w_limit = min(w_limit, SCOPUS_MAX_RESULTS)
            start = 0

            while start < w_limit:
                if max_results is not None and len(all_entries) >= max_results:
                    break

                page = await _fetch_page(
                    client,
                    headers,
                    {
                        "query": w_query,
                        "count": PAGE_SIZE,
                        "start": start,
                        "sort": "coverDate",
                    },
                )
                if page is None:
                    # Fin de la ventana por error o por tope de la API.
                    if start >= SCOPUS_MAX_RESULTS - PAGE_SIZE:
                        w["truncated"] = True
                        if w not in truncated_windows:
                            truncated_windows.append(w)
                    break

                entries = page.get("search-results", {}).get("entry", [])
                if not entries:
                    break

                all_entries.extend(entries)
                start += PAGE_SIZE

                progress = 10 + int((len(all_entries) / max(expected, 1)) * 70)
                yield {
                    "status": "running",
                    "progress": min(progress, 80),
                    "message": (
                        f"Descargadas {len(all_entries):,} de ~{expected:,} "
                        f"({w['from']}–{w['to']})…"
                    ),
                }
                await asyncio.sleep(0.3)  # rate limiting cortés

            await asyncio.sleep(0.5)  # pausa entre ventanas

    # ── 4. Agregación por año ─────────────────────────────────
    yield {"status": "running", "progress": 82, "message": "Agregando publicaciones por año…"}

    pubs_by_year: dict[int, int] = {}
    without_date = 0
    for entry in all_entries:
        yr = _extract_year_sync(entry)
        if yr:
            pubs_by_year[yr] = pubs_by_year.get(yr, 0) + 1
        else:
            without_date += 1

    if not pubs_by_year:
        yield {"status": "error", "message": "No se pudieron extraer fechas de las publicaciones."}
        return

    min_yr, max_yr = min(pubs_by_year), max(pubs_by_year)
    years = list(range(min_yr, max_yr + 1))
    annual = [int(pubs_by_year.get(y, 0)) for y in years]
    cumulative = [int(v) for v in np.cumsum(annual)]

    # ── 5. Ajuste y métricas ──────────────────────────────────
    yield {"status": "running", "progress": 90, "message": "Ajustando curva sigmoidal…"}

    result = _assemble_result(
        query=query,
        years=years,
        annual=annual,
        cumulative=cumulative,
        total_reported=total_reported,
        total_retrieved=len(all_entries),
        entries_without_date=without_date,
        truncated_windows=[
            {"from": w["from"], "to": w["to"], "total": w["total"]}
            for w in truncated_windows
        ],
    )

    yield {"status": "running", "progress": 96, "message": "Calculando métricas finales…"}

    if result["truncated"]:
        periods = ", ".join(
            f"{w['from']}–{w['to']}" if w["from"] != w["to"] else str(w["from"])
            for w in result["truncated_windows"]
        )
        yield {
            "status": "running",
            "progress": 98,
            "message": (
                f"Aviso: Scopus truncó los resultados en {periods}. "
                "La curva puede subestimar esos años."
            ),
        }

    yield {"status": "completed", "progress": 100, "message": "Análisis completado.", "result": result}
