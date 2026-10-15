"""
backend/core/matrix_board.py

Tablero de la matriz patentes/papers: ubica cada tecnología corrida en una
sesión de taller y genera el análisis prospectivo de la sesión con IA.

Convención de la matriz (presentación de Zartha, "Toma de decisiones en
Agroindustria apoyada en Cuadrantes Matriz Patentes – Papers"):

    X = año de inflexión de papers · Y = año de inflexión de patentes
    corte de ambos ejes en el año de referencia de la sesión

    papers ≤ ref y patentes ≤ ref  → maturity    (Madurez y Optimización)
    papers > ref y patentes ≤ ref  → industrial  (Explotación Industrial y Ciencia Aplicada)
    papers ≤ ref y patentes > ref  → disruption  (Oportunidad de Disrupción)
    papers > ref y patentes > ref  → watch       (Vigilancia Tecnológica y Apuestas a Futuro)

La inflexión de cada serie es la MEDIANA de los modelos que convergieron, no la
del mejor por AICc: en la prueba macro de octubre de 2026 el mejor modelo llegó
a poner inflexiones en 2077 con los otros cuatro alrededor de 2030.

El análisis de IA usa la integración OpenAI del asistente (`core/assistant.py`)
y, si el corpus metodológico responde, lo ancla con RAG. Toda salida va
marcada `is_ai_generated` y sólo puede citar fragmentos recuperados.
"""

from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

QUADRANTS = {
    (True, True): "maturity",
    (False, True): "industrial",
    (True, False): "disruption",
    (False, False): "watch",
}

QUADRANT_LABELS = {
    "maturity": "Madurez y Optimización",
    "industrial": "Explotación Industrial y Ciencia Aplicada",
    "disruption": "Oportunidad de Disrupción",
    "watch": "Vigilancia Tecnológica y Apuestas a Futuro",
}


def summarize_series(result: dict) -> dict:
    """
    Lo que se guarda de un SCurveResult: series anuales, tabla de modelos sin
    residuos ni curvas, fase y totales. Lo suficiente para redibujar la curva
    y la matriz sin volver a gastar peticiones.
    """
    comp = result.get("model_comparison") or {}
    models = []
    for m in comp.get("models") or []:
        models.append({k: m.get(k) for k in (
            "key", "label", "converged", "params", "r2_adj", "se", "durbin_watson",
            "aicc", "inflection_year", "inflection_passed", "saturation_L",
            "saturation_pct", "is_best",
        )})
    return {
        "source": result.get("source"),
        "source_label": result.get("source_label"),
        "query": result.get("query"),
        "year_from": result.get("year_from"),
        "year_to": result.get("year_to"),
        "total": result.get("total_retrieved") or result.get("total_publications"),
        "yearly_data": result.get("yearly_data") or [],
        "peak_annual": result.get("peak_annual"),
        "peak_annual_year": result.get("peak_annual_year"),
        "phase": result.get("phase"),
        "model_comparison": {
            "best": comp.get("best"),
            "best_label": comp.get("best_label"),
            "inflection_range": comp.get("inflection_range"),
            "saturation_range": comp.get("saturation_range"),
            "warning": comp.get("warning"),
            "models": models,
        },
    }


def median_inflection(summary: dict) -> tuple[Optional[float], Optional[list[float]]]:
    comp = summary.get("model_comparison") or {}
    infl = [
        float(m["inflection_year"]) for m in comp.get("models") or []
        if m.get("converged") and m.get("inflection_year") is not None
    ]
    if not infl:
        return None, None
    rng = comp.get("inflection_range") or [min(infl), max(infl)]
    return round(statistics.median(infl), 1), [round(float(rng[0]), 1), round(float(rng[1]), 1)]


def place(papers: dict, patents: dict, ref_year: int) -> dict:
    """Posición en la matriz a partir de los dos resúmenes."""
    px, prange = median_inflection(papers)
    ty, trange = median_inflection(patents)
    if px is None or ty is None:
        return {"papers_inflection": px, "patents_inflection": ty, "papers_range": prange,
                "patents_range": trange, "quadrant": None, "robust": None}
    robust = all((r[0] > ref_year) == (r[1] > ref_year) for r in (prange, trange) if r)
    return {
        "papers_inflection": px,
        "patents_inflection": ty,
        "papers_range": prange,
        "patents_range": trange,
        "quadrant": QUADRANTS[(px <= ref_year, ty <= ref_year)],
        "robust": robust,
    }


# ── Análisis de IA ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres el analista de prospectiva tecnológica de la plataforma Prospecta.
Recibes los resultados de una sesión de taller: varias tecnologías ubicadas en la
matriz patentes/papers (X = año de inflexión de la curva en S de artículos científicos,
Y = año de inflexión de la curva en S de patentes, corte en el año de referencia).

Cuadrantes (Zartha et al.):
- maturity (ambas inflexiones pasadas): Madurez y Optimización — optimización, eficiencia de costos, innovación incremental y de proceso.
- industrial (sólo patentes pasada): Explotación Industrial y Ciencia Aplicada — comercializar y escalar lo patentado, investigación aplicada con la industria.
- disruption (sólo papers pasada): Oportunidad de Disrupción — transferencia tecnológica, prototipos, protección agresiva de PI.
- watch (ninguna pasada): Vigilancia Tecnológica y Apuestas a Futuro — vigilancia, investigación exploratoria, pilotos pequeños.

Reglas:
1. Usa SOLO las cifras que te entregan. No inventes datos, años ni volúmenes.
2. Las inflexiones son la mediana de cinco modelos; si "robust" es false, el cuadrante depende del modelo y debes decirlo.
3. Una inflexión muy posterior al año de referencia es una extrapolación, no un pronóstico.
4. Si una tecnología parece mal medida (muy pocos registros, una ecuación ambigua, un evento que no es una tecnología), dilo.
5. Sólo puedes citar fragmentos del corpus que aparezcan en CONTEXTO, por su cite_key entre corchetes. Si no hay contexto, no cites.
6. Escribe en español, claro y directo, para un público de taller.

Responde SOLO con un objeto JSON con estas claves:
{
  "resumen": "3 a 5 frases con la lectura de conjunto",
  "por_tecnologia": [{"label": "...", "cuadrante": "...", "lectura": "2-3 frases", "alerta": "advertencia de calidad o null"}],
  "por_cuadrante": [{"cuadrante": "...", "tecnologias": ["..."], "recomendacion": "1-2 frases"}],
  "preguntas_para_el_panel": ["3 a 5 preguntas para discutir en el taller"],
  "citas": ["cite_key usados"]
}"""


def _run_brief(run: dict) -> dict:
    p, t = run.get("papers") or {}, run.get("patents") or {}

    def recent(s):
        yd = s.get("yearly_data") or []
        return {str(d["year"]): d["annual"] for d in yd[-6:]}

    return {
        "label": run["label"],
        "cuadrante": run.get("quadrant"),
        "robust": run.get("robust"),
        "inflexion_papers": float(run["papers_inflection"]) if run.get("papers_inflection") is not None else None,
        "rango_papers": run.get("papers_range"),
        "inflexion_patentes": float(run["patents_inflection"]) if run.get("patents_inflection") is not None else None,
        "rango_patentes": run.get("patents_range"),
        "papers_total": p.get("total"),
        "patentes_total": t.get("total"),
        "fuente_patentes": run.get("patent_source"),
        "fase_papers": (p.get("phase") or {}).get("phase"),
        "fase_patentes": (t.get("phase") or {}).get("phase"),
        "papers_ultimos_anios": recent(p),
        "patentes_ultimos_anios": recent(t),
        "ecuacion_scopus": run.get("scopus_query"),
        "ecuacion_patentes": run.get("patents_query"),
    }


def analyze_session(session: dict, runs: list[dict]) -> dict:
    """Llama al modelo con los resultados de la sesión. Lanza AssistantError si falla."""
    from core.assistant import AssistantError, _ai_marker, _call_json, default_model

    if not runs:
        raise AssistantError("La sesión no tiene tecnologías corridas todavía.")

    chunks = []
    try:
        from core import rag
        chunks = rag.search_multi([
            "curvas en S punto de inflexión ciclo de vida de la tecnología",
            "matriz patentes papers cuadrantes decisiones",
            "vigilancia tecnológica prospectiva recomendaciones estratégicas",
        ], k_per_query=3, max_chunks=8)
    except Exception as exc:  # noqa: BLE001 — sin RAG el análisis sigue, sin citas
        logger.warning("matrix_board: RAG no disponible (%s)", type(exc).__name__)

    contexto = "\n\n".join(
        f"[{c.cite_key}] {c.title} ({c.year or 's.f.'})\n{c.content[:900]}" for c in chunks
    ) or "(sin contexto: no cites nada)"

    user = (
        f"SESIÓN: {session['name']}\nAÑO DE REFERENCIA: {session['reference_year']}\n\n"
        f"TECNOLOGÍAS:\n{json.dumps([_run_brief(r) for r in runs], ensure_ascii=False, indent=1)}\n\n"
        f"CONTEXTO:\n{contexto}"
    )
    model = default_model()
    data = _call_json(SYSTEM_PROMPT, user, model=model, max_tokens=2500, temperature=0.3)

    # Ninguna cita sin respaldo: lo que no vino del RAG se descarta y se reporta.
    valid = {c.cite_key for c in chunks}
    citas = [c for c in (data.get("citas") or []) if c in valid]
    descartadas = [c for c in (data.get("citas") or []) if c not in valid]

    return {
        **data,
        "citas": citas,
        "citas_descartadas": descartadas,
        "fuentes": [{"cite_key": c.cite_key, "title": c.title, "year": c.year, "url": c.url} for c in chunks if c.cite_key in citas],
        "n_tecnologias": len(runs),
        **_ai_marker(model, generated_at=datetime.now().isoformat(timespec="seconds")),
    }
