"""
backend/core/hype_trends.py

Hype Cycle a partir de Google Trends (vía SerpAPI), en lugar de contar
noticias recuperadas.

Por qué (prueba macro de octubre de 2026, ver data/macro_test):

  · Contar noticias de Google News mide cómo ordena el buscador, no el
    interés. Con las cuatro ventanas de `NewsAnalyzer._search_complex_query`
    los picos caían SIEMPRE en el último año de cada ventana (2019, 2022,
    2025, 2026), fuera cual fuera la tecnología.
  · Google dejó de respetar `num=100`: cada llamada trae ~10 noticias, así que
    ese método gasta ~40 créditos por tecnología.
  · Una llamada a `engine=google_trends` devuelve el interés MENSUAL desde 2004,
    normalizado 0–100. Con `gprop=news` se limita a búsquedas de noticias, que
    es la señal de expectativa mediática que usa la literatura del Hype Cycle.

Modelo (Sasaki 2015; Ren et al. 2015, citados en la presentación de Zartha):
la curva del Hype Cycle es la suma de una campana —la expectativa, que sube y
se desinfla— y una logística —la adopción, que crece y se estabiliza:

    h(t) = a · exp(−½((t − tp)/w)²) + L / (1 + exp(−k(t − t0)))

La fase sale de dónde cae HOY respecto a los parámetros ajustados, no de
umbrales sobre conteos. Las reglas de `phase_from_fit` son un primer criterio
explícito y revisable, no un estándar: se documentan para poder discutirlas.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import requests
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search.json"
_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)}

# Por debajo de este interés medio la serie es casi toda ceros: Trends
# redondea a enteros y una tecnología de nicho no deja curva que ajustar.
MIN_MEAN_INTEREST = 3.0
# Ruptura de medición de Google Trends: en agosto de 2025 las series de cinco
# términos sin relación entre sí (facial recognition, desalination, machine to
# machine, point of care, synthetic biology) saltaron a la vez por un factor
# ~2,3 y siguieron altas. Eso no es interés: es un cambio en cómo Trends mide.
# Los ajustes terminan antes de la ruptura; los meses posteriores se muestran
# pero no se ajustan.
TRENDS_BREAK = 2025 + 7 / 12  # agosto de 2025

# Una campana más ancha que esto ya no es un pico de expectativa, es
# crecimiento sostenido: se trata como "sin pico".
MAX_BELL_WIDTH = 5.0


def fetch_trends(
    terms: list[str],
    api_key: Optional[str] = None,
    gprop: Optional[str] = "news",
    date: str = "all",
    timeout: int = 120,
) -> dict:
    """Una llamada (un crédito) para hasta 5 términos. Devuelve la respuesta cruda."""
    if not 1 <= len(terms) <= 5:
        raise ValueError("Google Trends admite de 1 a 5 términos por llamada")
    params = {
        "engine": "google_trends",
        "q": ",".join(terms),
        "data_type": "TIMESERIES",
        "date": date,
        "api_key": api_key or os.getenv("SERP_API_KEY", ""),
    }
    if gprop:
        params["gprop"] = gprop
    r = requests.get(SERPAPI_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"SerpAPI: {data['error']}")
    return data


def monthly_series(data: dict) -> dict[str, list[tuple[float, float]]]:
    """{término: [(año decimal, interés 0–100), …]} desde la respuesta cruda."""
    out: dict[str, list[tuple[float, float]]] = {}
    for row in data.get("interest_over_time", {}).get("timeline_data", []):
        raw = row["date"]
        # "Jan 2004" en meses; semanas ("Jan 1 – 7, 2024") no se esperan con date=all.
        t = int(raw[-4:]) + _MONTHS.get(raw[:3], 0) / 12
        for v in row.get("values", []):
            out.setdefault(v["query"], []).append((t, float(v.get("extracted_value", 0))))
    return out


def hype_curve(t, a, tp, w, L, t0, k):
    return a * np.exp(-0.5 * ((t - tp) / w) ** 2) + L / (1 + np.exp(-k * (t - t0)))


def fit_hype(
    series: list[tuple[float, float]],
    start: float = 2008.0,
    end: Optional[float] = None,
    smooth_months: int = 12,
) -> dict:
    """
    Ajusta la curva sobre la serie mensual suavizada (media móvil de 12 meses:
    se busca la tendencia, no el pico de una sola noticia).

    `end` excluye por defecto el año en curso y todo lo posterior a
    `TRENDS_BREAK`.
    """
    if end is None:
        end = min(float(int(series[-1][0])), TRENDS_BREAK)
    t = np.array([p[0] for p in series if start <= p[0] < end])
    y = np.array([p[1] for p in series if start <= p[0] < end])
    if t.size < 36:
        return {"ok": False, "reason": "Menos de 3 años de datos mensuales."}

    mean = float(y.mean())
    if mean < MIN_MEAN_INTEREST:
        return {
            "ok": False,
            "reason": (
                f"Interés medio {mean:.1f}/100: la serie es casi toda ceros y "
                "no hay curva que ajustar."
            ),
            "mean_interest": round(mean, 2),
        }

    kernel = np.ones(smooth_months) / smooth_months
    ys = np.convolve(y, kernel, mode="same")
    # Los bordes de la convolución quedan sesgados a la baja; se recortan.
    edge = smooth_months // 2
    t_fit, y_fit = t[edge:-edge], ys[edge:-edge]

    lo = [0, start, 0.3, 0, start, 0.05]
    hi = [300, t[-1] + 4, 8, 300, t[-1] + 20, 10]
    best = None
    for tp0 in np.linspace(start + 2, t[-1], 5):
        for t00 in (t[-1] - 6, t[-1], t[-1] + 4):
            p0 = [max(y_fit.max() / 2, 1), tp0, 2, max(y_fit[-12:].mean(), 1), t00, 0.8]
            p0 = [min(max(v, l + 1e-6), h - 1e-6) for v, l, h in zip(p0, lo, hi)]
            try:
                p, _ = curve_fit(hype_curve, t_fit, y_fit, p0=p0, bounds=(lo, hi), maxfev=40000)
            except Exception:
                continue
            sse = float(((y_fit - hype_curve(t_fit, *p)) ** 2).sum())
            if best is None or sse < best[1]:
                best = (p, sse)

    if best is None:
        return {"ok": False, "reason": "El ajuste no convergió."}

    p, sse = best
    a, tp, w, L, t0, k = (float(v) for v in p)
    sst = float(((y_fit - y_fit.mean()) ** 2).sum()) or 1.0
    grid = np.linspace(t_fit[0], t_fit[-1], 4000)
    h = hype_curve(grid, *p)
    after = grid > tp
    trough = float(grid[after][np.argmin(h[after])]) if after.any() else None

    return {
        "ok": True,
        "params": {"a": a, "tp": tp, "w": w, "L": L, "t0": t0, "k": k},
        "r2": round(1 - sse / sst, 3),
        "mean_interest": round(mean, 2),
        "hype_peak_year": round(tp, 1),
        "bell_width_years": round(w, 1),
        "trough_year": round(trough, 1) if trough is not None else None,
        "adoption_midpoint": round(t0, 1),
        "last_observed": round(float(t[-1]), 2),
        "fitted": [
            {"t": round(float(tt), 3), "y": round(float(v), 2)}
            for tt, v in zip(t_fit[::3], hype_curve(t_fit[::3], *p))
        ],
    }


def phase_from_fit(fit: dict, now: Optional[float] = None) -> dict:
    """
    Fase del Hype Cycle a partir de los parámetros. Criterio:

      sin pico (campana pequeña frente a la adopción, o más ancha que 5 años):
          se lee sólo la logística → Trigger (<10 % de L) · Slope (<90 %) · Plateau
      con pico:
          antes de tp − 1.5w                 → Innovation Trigger
          entre tp − 1.5w y tp + 0.75w       → Peak of Inflated Expectations
          entre tp + 0.75w y el valle + 1    → Trough of Disillusionment
          después del valle, adopción < 90 % → Slope of Enlightenment
          después del valle, adopción ≥ 90 % → Plateau of Productivity
    """
    if not fit.get("ok"):
        return {"phase": "Insufficient signal", "reason": fit.get("reason")}

    now = now or fit["last_observed"]
    p = fit["params"]
    adoption = 1 / (1 + np.exp(-p["k"] * (now - p["t0"])))
    has_peak = p["a"] >= 0.25 * max(p["L"], 1e-9) and p["w"] <= MAX_BELL_WIDTH

    if not has_peak:
        phase = (
            "Innovation Trigger" if adoption < 0.10
            else "Slope of Enlightenment" if adoption < 0.90
            else "Plateau of Productivity"
        )
        basis = "adoption only (no distinct expectation peak)"
    else:
        tp, w, tr = p["tp"], p["w"], fit.get("trough_year") or (p["tp"] + 2 * p["w"])
        if now < tp - 1.5 * w:
            phase = "Innovation Trigger"
        elif now <= tp + 0.75 * w:
            phase = "Peak of Inflated Expectations"
        elif now <= tr + 1:
            phase = "Trough of Disillusionment"
        elif adoption < 0.90:
            phase = "Slope of Enlightenment"
        else:
            phase = "Plateau of Productivity"
        basis = "expectation peak + adoption"

    return {
        "phase": phase,
        "basis": basis,
        "has_peak": bool(has_peak),
        "adoption_share": round(float(adoption), 3),
        "evaluated_at": round(float(now), 2),
    }


def analyze(terms: list[str], api_key: Optional[str] = None, gprop: Optional[str] = None) -> dict:
    """
    Una llamada POR TÉRMINO (un crédito cada una) + ajuste + fase.

    Varios términos en una sola llamada se normalizan entre sí: el dominante
    se queda la escala y los demás caen a 0. Y las búsquedas de noticias
    (`gprop="news"`) de términos de nicho son casi todo ceros con picos
    sueltos de 100; por eso el valor por defecto es la búsqueda web.
    """
    series: dict[str, list[tuple[float, float]]] = {}
    raw: dict[str, list] = {}
    for term in terms:
        data = fetch_trends([term], api_key=api_key, gprop=gprop)
        series.update(monthly_series(data))
        raw[term] = data.get("interest_over_time", {}).get("timeline_data", [])
    results = {}
    for term, s in series.items():
        fit = fit_hype(s)
        results[term] = {
            "fit": fit,
            "phase": phase_from_fit(fit),
            "annual_mean": _annual_mean(s),
        }
    return {
        "source": f"google_trends gprop={gprop or 'web'}",
        "terms": terms,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
        "raw": raw,
    }


def _annual_mean(s: list[tuple[float, float]]) -> dict[int, float]:
    acc: dict[int, list[float]] = {}
    for t, v in s:
        acc.setdefault(int(t), []).append(v)
    return {y: round(float(np.mean(v)), 1) for y, v in sorted(acc.items())}
