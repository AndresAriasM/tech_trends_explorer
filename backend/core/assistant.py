"""
backend/core/assistant.py

La capa transversal de IA: no es un módulo con pantalla propia, es una
capacidad que los demás módulos invocan (ver docs/ARQUITECTURA_MODULOS.md).

Dos capacidades, las dos de la versión perdida:

  1. `generate_search_strategy` — de un concepto ("biodiesel") a la estrategia
     de búsqueda completa: ecuación Scopus, ecuación EPO OPS, términos de Hype
     Cycle, rango de años, estimación de peticiones y la justificación en prosa
     de por qué esos términos y no otros. Es el asistente para «elaborar las
     ecuaciones» de Curvas en S y Hype Cycle.

  2. `generate_foresight_report` — el informe prospectivo de secciones fijas,
     construido sobre la evidencia numérica que producen los módulos analíticos
     y **anclado al corpus metodológico** (RAG sobre `zartha_chunks`).

Dos invariantes que no se negocian:

  · **Todo lo generado va marcado.** Igual que `ai_voter.py` marca sus votos
    con `is_ai_vote=True`, aquí cada salida lleva `is_ai_generated=True` con
    proveedor, modelo e instante. Un informe de IA no puede confundirse nunca
    con uno redactado por el equipo.

  · **Ninguna cita se inventa.** El generador sólo puede citar las `cite_key`
    que la recuperación trajo de verdad, y lo que escribe se comprueba contra
    esa lista *después* de generarlo: una etiqueta que no esté respaldada se
    sustituye por `[cita no verificada]` y se reporta. Si el corpus no cubre
    una afirmación metodológica, el informe debe decirlo en vez de citar algo
    parecido.

Proveedor: OpenAI (decisión de septiembre de 2026). Modelo por `OPENAI_MODEL`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from core import rag

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENAI_AVAILABLE = False
    logger.warning("openai no instalado — uv pip install openai")


# ── Identidad de la IA ────────────────────────────────────────────────────────

AI_GENERATOR_ID = "PROSPECTA_ASSISTANT"
AI_GENERATOR_NAME = "Asistente IA Prospecta"
AI_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o"

CITATION_UNVERIFIED = "[cita no verificada]"


class AssistantError(RuntimeError):
    """Fallo recuperable del asistente."""


def default_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def _ai_marker(model: str, **extra: Any) -> dict[str, Any]:
    """
    El sello que distingue lo generado por IA de lo humano.

    Mismo papel que `is_ai_vote=True` en ai_voter.py; se incrusta en todas las
    salidas de este módulo para que el frontend y la persistencia nunca tengan
    que adivinar el origen de un texto.
    """
    marker = {
        "is_ai_generated": True,
        "ai_generator": AI_GENERATOR_ID,
        "ai_generator_name": AI_GENERATOR_NAME,
        "ai_provider": AI_PROVIDER,
        "ai_model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    marker.update(extra)
    return marker


# ── Cliente OpenAI ────────────────────────────────────────────────────────────

def _client() -> "OpenAI":
    if not OPENAI_AVAILABLE:
        raise AssistantError("openai no instalado — uv pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AssistantError("OPENAI_API_KEY no configurada")
    return OpenAI(api_key=api_key)


def _call_json(
    system: str,
    user: str,
    model: str,
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Una llamada a chat.completions con salida JSON obligatoria."""
    response = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssistantError(f"El modelo no devolvió JSON válido: {exc}") from exc
    if not isinstance(data, dict):
        raise AssistantError("El modelo devolvió un JSON que no es un objeto")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 1 · Generador de estrategia de búsqueda
# ══════════════════════════════════════════════════════════════════════════════

# Una petición por año y por fuente. Es el plan de ventanas anuales que usa
# `core/s_curve.py` para no chocar con el tope de 5.000 resultados por consulta
# de Scopus. Es una COTA INFERIOR: si un año supera el tope, se subdivide, y la
# paginación de 25 en 25 añade las suyas. Se calcula aquí, en Python, y no se
# le pide al modelo: una cifra operativa no se alucina.
SOURCE_LABELS = {
    "scopus": "Scopus (papers)",
    "epo": "EPO OPS (patentes)",
    "hype": "Hype Cycle (noticias)",
}
DEFAULT_SOURCES = ("scopus", "epo")

MIN_YEAR = 1900

STRATEGY_SYSTEM_PROMPT = """Eres un especialista en vigilancia tecnológica y prospectiva. \
Construyes las ecuaciones de búsqueda de un estudio de curvas en S y Hype Cycle.

Tienes que producir, a partir de un concepto, una estrategia de búsqueda coherente en tres \
fuentes distintas:

1. SCOPUS (literatura científica). Sintaxis obligatoria: TITLE-ABS-KEY(...). Puedes combinar \
con OR / AND / AND NOT y comillas dobles para frases exactas.
   Ejemplo: TITLE-ABS-KEY("solid state battery" OR "all-solid-state battery")

2. EPO OPS (patentes). Sintaxis CQL obligatoria del campo de texto libre: txt = "..." \
Puedes combinar con or / and. Ejemplo: txt = "solid state battery" or txt = "solid-state battery"

3. HYPE CYCLE (noticias y web). Una lista corta de 2 a 4 términos de búsqueda en lenguaje \
natural, sin operadores.

Criterios:
- La PRECISIÓN manda sobre la exhaustividad: un término ambiguo mete ruido que arruina la
  curva acumulada. Es preferible una ecuación estrecha y defendible.
- Scopus y EPO pueden ser MÁS ESTRICTAS que los términos de Hype Cycle: las noticias necesitan
  vocabulario divulgativo, los papers y las patentes no.
- El año inicial debe capturar el despegue real de la tecnología, sin retroceder décadas donde
  no hay señal. El año final es el año en curso.
- Justifica en prosa (3-5 oraciones, en español) por qué ese concepto núcleo, por qué ese año
  inicial y qué términos relacionados decidiste DEJAR FUERA y por qué.

Responde SIEMPRE con este JSON y nada más:
{
  "core_term": "<el concepto núcleo elegido>",
  "synonyms": ["<sinónimos o variantes incluidas>"],
  "excluded_terms": ["<términos relacionados que dejaste fuera>"],
  "scopus_query": "TITLE-ABS-KEY(...)",
  "epo_query": "txt = \\"...\\"",
  "hype_cycle_terms": ["<término 1>", "<término 2>"],
  "year_from": <entero>,
  "year_to": <entero>,
  "rationale": "<justificación en prosa, en español>"
}"""


def estimate_requests(
    year_from: int,
    year_to: int,
    sources: Sequence[str] = DEFAULT_SOURCES,
) -> dict[str, Any]:
    """
    Cota inferior de peticiones a las APIs, calculada, no generada.

    Una ventana anual por fuente. El desglose se devuelve para que la interfaz
    pueda explicar la cifra en vez de mostrar un número sin origen.
    """
    span = max(0, int(year_to) - int(year_from)) + 1
    breakdown = {s: span for s in sources}
    return {
        "years": span,
        "per_source": breakdown,
        "total": span * len(breakdown),
        "is_lower_bound": True,
        "basis": (
            "Una ventana anual por fuente. No incluye las subdivisiones "
            "por el tope de 5.000 resultados de Scopus ni la paginación."
        ),
    }


def _clean_terms(value: Any, limit: int = 6) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        term = str(item).strip().strip('"')
        if term and term.lower() not in {t.lower() for t in out}:
            out.append(term)
    return out[:limit]


def _normalize_strategy(data: dict[str, Any], concept: str) -> dict[str, Any]:
    """
    Repara la salida del modelo hasta que sea ejecutable.

    Una ecuación con la sintaxis rota no falla en el asistente, falla más tarde
    contra la API y con un error que no dice nada. Se corrige aquí.
    """
    current_year = datetime.now().year
    core = str(data.get("core_term") or concept).strip() or concept

    scopus = str(data.get("scopus_query") or "").strip()
    if "TITLE-ABS-KEY" not in scopus.upper():
        scopus = f'TITLE-ABS-KEY("{core}")'

    epo = str(data.get("epo_query") or "").strip()
    if not re.search(r"\btxt\s*=", epo, flags=re.IGNORECASE):
        epo = f'txt = "{core}"'

    hype = _clean_terms(data.get("hype_cycle_terms"), limit=4) or [core]

    def _year(key: str, fallback: int) -> int:
        try:
            value = int(data.get(key))
        except (TypeError, ValueError):
            return fallback
        return max(MIN_YEAR, min(current_year, value))

    year_to = _year("year_to", current_year)
    year_from = _year("year_from", max(MIN_YEAR, year_to - 25))
    if year_from > year_to:
        year_from, year_to = year_to, year_from

    return {
        "concept": concept,
        "core_term": core,
        "synonyms": _clean_terms(data.get("synonyms"), limit=8),
        "excluded_terms": _clean_terms(data.get("excluded_terms"), limit=8),
        "scopus_query": scopus,
        "epo_query": epo,
        "hype_cycle_terms": hype,
        "year_from": year_from,
        "year_to": year_to,
        "rationale": str(data.get("rationale") or "").strip(),
    }


def generate_search_strategy(
    concept: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    notes: str = "",
    sources: Sequence[str] = DEFAULT_SOURCES,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """
    Una llamada a OpenAI. Devuelve la estrategia marcada como generada por IA.

    `year_from` / `year_to` son sugerencias del usuario: si vienen, el modelo
    las respeta; si no, las propone él y se justifican en `rationale`.
    """
    concept = (concept or "").strip()
    if not concept:
        raise AssistantError("El concepto no puede estar vacío")

    model = model or default_model()
    current_year = datetime.now().year

    user = f"CONCEPTO: {concept}\nAÑO EN CURSO: {current_year}\n"
    if year_from is not None:
        user += f"AÑO INICIAL IMPUESTO POR EL USUARIO: {year_from} (respétalo)\n"
    if year_to is not None:
        user += f"AÑO FINAL IMPUESTO POR EL USUARIO: {year_to} (respétalo)\n"
    if notes.strip():
        user += f"\nCONTEXTO ADICIONAL DEL ESTUDIO:\n{notes.strip()}\n"
    user += "\nGenera la estrategia siguiendo estrictamente el formato JSON indicado."

    data = _call_json(STRATEGY_SYSTEM_PROMPT, user, model=model, max_tokens=800)
    strategy = _normalize_strategy(data, concept)

    # Los años impuestos por el usuario mandan sobre lo que devuelva el modelo.
    if year_from is not None:
        strategy["year_from"] = int(year_from)
    if year_to is not None:
        strategy["year_to"] = int(year_to)
    if strategy["year_from"] > strategy["year_to"]:
        strategy["year_from"], strategy["year_to"] = (
            strategy["year_to"],
            strategy["year_from"],
        )

    strategy["estimated_requests"] = estimate_requests(
        strategy["year_from"], strategy["year_to"], sources
    )
    strategy["queries"] = [
        {"source": "scopus", "label": SOURCE_LABELS["scopus"], "query": strategy["scopus_query"]},
        {"source": "epo", "label": SOURCE_LABELS["epo"], "query": strategy["epo_query"]},
        {
            "source": "hype",
            "label": SOURCE_LABELS["hype"],
            "query": " · ".join(strategy["hype_cycle_terms"]),
        },
    ]
    strategy.update(_ai_marker(model, openai_calls=1))
    return strategy


# ══════════════════════════════════════════════════════════════════════════════
# 2 · Informe prospectivo con RAG
# ══════════════════════════════════════════════════════════════════════════════

# Las secciones son fijas: un informe prospectivo comparable entre estudios no
# puede tener un índice distinto cada vez.
#   id · encabezado · forma · qué debe contener
SECTIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "resumen_ejecutivo", "Resumen ejecutivo", "prose",
        "Qué se analizó, con qué instrumentos, y las 2-3 conclusiones que "
        "aguantan. Un párrafo de 4-6 oraciones.",
    ),
    (
        "dinamica_cientifica", "Dinámica científica (papers)", "prose",
        "Volumen de publicaciones, punto de inflexión, saturación y fase del "
        "ciclo de vida según la curva en S de papers. Un párrafo.",
    ),
    (
        "dinamica_tecnologica", "Dinámica tecnológica (patentes)", "prose",
        "Lo mismo para patentes, incluyendo si el ajuste es fiable "
        "(autocorrelación de residuos, error de L). Un párrafo.",
    ),
    (
        "triangulacion", "Triangulación y fase del ciclo de vida", "prose",
        "Cruce ciencia/tecnología: coherencia o divergencia entre ambas "
        "curvas, desfase entre inflexiones y la fase resultante con su "
        "incertidumbre. Un párrafo.",
    ),
    (
        "hype_cycle", "Lectura del Hype Cycle", "prose",
        "Fase de expectativas, su confianza, y si es consistente o no con la "
        "evidencia bibliométrica. Un párrafo.",
    ),
    (
        "recomendaciones", "Recomendaciones estratégicas", "items",
        "2-4 recomendaciones accionables. Cada una: título corto y 1-2 "
        "oraciones. Formato: lista de objetos {\"title\", \"text\"}.",
    ),
    (
        "temas_delphi", "Temas sugeridos para ronda Delphi", "bullets",
        "3-5 temas sobre los que un panel de expertos aportaría lo que los "
        "datos no resuelven. Lista de cadenas.",
    ),
    (
        "limitaciones", "Limitaciones metodológicas", "bullets",
        "3-5 limitaciones reales de ESTE análisis, derivadas de la evidencia "
        "aportada. Lista de cadenas.",
    ),
)

SECTION_BY_ID = {s[0]: s for s in SECTIONS}

# Una sonda por eje temático del informe. Una sola consulta larga promedia
# todos los temas y no recupera bien ninguno; seis sondas cortas, sí. Las seis
# se embeben en UNA llamada (rag.search_multi).
REPORT_PROBES: tuple[str, ...] = (
    "curvas en S y fases del ciclo de vida de la tecnología",
    "punto de inflexión y nivel de saturación de la curva logística",
    "vigilancia tecnológica con papers y patentes, desfase entre ciencia y tecnología",
    "ciclo de sobreexpectación, hype cycle y expectativas del mercado",
    "método Delphi, panel de expertos y selección de temas",
    "limitaciones metodológicas y bondad del ajuste en el análisis del ciclo de vida",
)

REPORT_SYSTEM_PROMPT = """Eres un analista de prospectiva tecnológica. Redactas el informe de \
un estudio que combina curvas en S (papers y patentes) y Hype Cycle.

REGLAS INNEGOCIABLES

1. Las CIFRAS salen exclusivamente del bloque EVIDENCIA. No inventes ni estimes ningún número
   que no esté ahí. Si un dato falta, escríbelo: "no se dispone de ...".
2. Las afirmaciones METODOLÓGICAS (qué significa una fase, cómo se lee una inflexión, qué
   invalida un ajuste, cómo se usa Delphi) se apoyan en el bloque CORPUS y se citan con la
   etiqueta [cite_key] EXACTA que ese bloque indica, entre corchetes y al final de la oración.
3. SÓLO puedes usar las cite_key listadas como permitidas. Está terminantemente prohibido
   inventar, abreviar o alterar una clave, o citar cualquier otra fuente.
4. Si el CORPUS no respalda una afirmación metodológica que quieres hacer, tienes dos salidas:
   no hacerla, o hacerla diciendo explícitamente que el corpus disponible no la respalda.
   Nunca la cites "por aproximación".
5. La incertidumbre se reporta, no se suaviza: un ajuste con residuos autocorrelacionados o un
   techo de saturación no identificable invalidan la lectura de la fase, y eso debe decirse.
6. Español, tono sobrio y técnico. Sin adjetivos comerciales. Sin listas dentro de los párrafos
   en prosa.

Responde SIEMPRE con un único objeto JSON con exactamente estas claves y nada más."""


# ── Formateo de la evidencia numérica ─────────────────────────────────────────

def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "no disponible"
    if isinstance(value, float):
        return f"{value:,.4g}{suffix}"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{value}{suffix}"


def _format_s_curve(label: str, data: Optional[dict[str, Any]]) -> str:
    """
    Traduce el resultado de `core/s_curve.py` a texto para el prompt.

    Es permisivo a propósito: los módulos evolucionan (el ajuste multi-modelo
    añade campos) y el asistente no debe romperse porque llegue una clave de
    más o de menos.
    """
    if not data:
        return f"{label}: sin datos."

    params = data.get("sigmoid_params") or {}
    phase = data.get("phase") or {}
    lines = [
        f"{label}:",
        f"  · registros totales: {_fmt(data.get('total_publications') or data.get('total_retrieved'))}",
        f"  · rango de años: {_fmt(data.get('year_from'))}–{_fmt(data.get('year_to'))}",
        f"  · punto de inflexión (x0): {_fmt(data.get('inflection_year') or params.get('x0'))}",
        f"  · techo de saturación estimado (L): {_fmt(data.get('saturation_level') or params.get('L'))}",
        f"  · saturación alcanzada: {_fmt(phase.get('saturation_pct'), ' %')}",
        f"  · R²: {_fmt(data.get('r2') or params.get('r2'))}",
        f"  · RMSE: {_fmt(params.get('rmse'))}",
        f"  · L identificable: {params.get('L_identifiable', 'no reportado')}",
        f"  · fase detectada: {phase.get('phase', 'no determinada')}"
        f" (base: {phase.get('basis', 'n/d')}"
        f"{', provisional' if phase.get('provisional') else ''})",
    ]
    if params.get("durbin_watson") is not None:
        lines.append(f"  · Durbin-Watson: {_fmt(params.get('durbin_watson'))}")
    if data.get("best_model") or params.get("model"):
        lines.append(f"  · modelo elegido: {data.get('best_model') or params.get('model')}")
    if data.get("peak_growth_year"):
        lines.append(f"  · pico de crecimiento anual: {_fmt(data.get('peak_growth_year'))}")
    if data.get("truncated"):
        lines.append("  · AVISO: la descarga se truncó, la serie puede estar incompleta.")
    if data.get("models"):
        lines.append("  · ajuste multi-modelo comparado: " + json.dumps(
            data["models"], ensure_ascii=False, default=str)[:900])
    return "\n".join(lines)


def _format_hype(data: Optional[dict[str, Any]]) -> str:
    if not data:
        return "Hype Cycle: sin datos."
    return "\n".join([
        "Hype Cycle:",
        f"  · fase: {data.get('phase', 'no determinada')}",
        f"  · confianza: {_fmt(data.get('confidence'))}",
        f"  · menciones/artículos: {_fmt(data.get('total_mentions') or data.get('total_articles'))}",
        f"  · sentimiento medio: {_fmt(data.get('sentiment') or data.get('average_sentiment'))}",
        f"  · tiempo estimado a meseta: {data.get('time_to_plateau', 'no estimado')}",
    ])


def _build_evidence_block(
    concept: str,
    s_curve_papers: Optional[dict[str, Any]],
    s_curve_patents: Optional[dict[str, Any]],
    hype_cycle: Optional[dict[str, Any]],
    strategy: Optional[dict[str, Any]],
    extra: Optional[str],
) -> str:
    parts = [f"CONCEPTO ANALIZADO: {concept}"]
    if strategy:
        parts.append(
            "ESTRATEGIA DE BÚSQUEDA USADA:\n"
            f"  · Scopus: {strategy.get('scopus_query', 'n/d')}\n"
            f"  · EPO OPS: {strategy.get('epo_query', 'n/d')}\n"
            f"  · Hype Cycle: {', '.join(strategy.get('hype_cycle_terms') or []) or 'n/d'}"
        )
    parts.append(_format_s_curve("CURVA EN S — papers (Scopus)", s_curve_papers))
    parts.append(_format_s_curve("CURVA EN S — patentes (EPO OPS)", s_curve_patents))
    parts.append(_format_hype(hype_cycle))

    # Desfase ciencia→tecnología: se calcula aquí porque es aritmética, y la
    # aritmética no se delega en un modelo de lenguaje.
    x0_papers = (s_curve_papers or {}).get("inflection_year")
    x0_patents = (s_curve_patents or {}).get("inflection_year")
    if x0_papers and x0_patents:
        lag = round(float(x0_papers) - float(x0_patents), 1)
        direction = "las patentes se inflexionan antes" if lag > 0 else "los papers se inflexionan antes"
        parts.append(f"DESFASE CIENCIA↔TECNOLOGÍA: {abs(lag)} años ({direction}).")
    else:
        parts.append("DESFASE CIENCIA↔TECNOLOGÍA: no calculable (falta una de las dos inflexiones).")

    if extra and extra.strip():
        parts.append(f"CONTEXTO ADICIONAL:\n{extra.strip()}")
    return "\n\n".join(parts)


def _build_schema_block() -> str:
    lines = ["{"]
    for sid, heading, kind, intent in SECTIONS:
        if kind == "prose":
            shape = '"<texto>"'
        elif kind == "items":
            shape = '[{"title": "<título>", "text": "<texto>"}]'
        else:
            shape = '["<punto 1>", "<punto 2>"]'
        lines.append(f'  "{sid}": {shape},   // {heading} — {intent}')
    lines.append("}")
    return "\n".join(lines)


# ── Comprobación de citas ─────────────────────────────────────────────────────

_BRACKET = re.compile(r"\[([^\[\]]{2,90})\]")
# Una etiqueta "parece una cita" si tiene un año de cuatro cifras o un "et al."
_LOOKS_LIKE_CITATION = re.compile(r"(\bet al\b|\b(1[89]|20)\d{2}[a-z]?\b)", re.IGNORECASE)


def _verify_citations(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    """
    Comprueba a posteriori que cada cita del texto esté respaldada.

    Pedirle al modelo que no invente citas reduce el problema; no lo elimina.
    Aquí se cierra: toda etiqueta entre corchetes que parezca una cita y no
    corresponda a una `cite_key` recuperada de verdad se sustituye por
    `[cita no verificada]` y se reporta. Preferimos una marca visible a un
    borrado silencioso: el lector debe ver dónde falló el anclaje.
    """
    warnings: list[str] = []

    def replace(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        if label in allowed:
            return match.group(0)
        if not _LOOKS_LIKE_CITATION.search(label):
            return match.group(0)  # no es una cita: un inciso, una unidad…
        warnings.append(label)
        return CITATION_UNVERIFIED

    return _BRACKET.sub(replace, text), warnings


def _cited_keys(text: str, allowed: set[str]) -> set[str]:
    return {label.strip() for label in _BRACKET.findall(text) if label.strip() in allowed}


def _walk_section(value: Any, kind: str, allowed: set[str]) -> tuple[Any, list[str], set[str]]:
    """Aplica la comprobación de citas a una sección, cualquiera sea su forma."""
    warnings: list[str] = []
    cited: set[str] = set()

    def clean(text: Any) -> str:
        nonlocal warnings, cited
        raw = str(text or "").strip()
        fixed, warn = _verify_citations(raw, allowed)
        warnings.extend(warn)
        cited |= _cited_keys(fixed, allowed)
        return fixed

    if kind == "prose":
        return clean(value), warnings, cited

    if kind == "items":
        items = value if isinstance(value, list) else []
        out = []
        for item in items:
            if isinstance(item, dict):
                out.append({"title": clean(item.get("title")), "text": clean(item.get("text"))})
            else:
                out.append({"title": "", "text": clean(item)})
        return out, warnings, cited

    items = value if isinstance(value, list) else ([value] if value else [])
    return [clean(i) for i in items], warnings, cited


# ── Generación del informe ────────────────────────────────────────────────────

def generate_foresight_report(
    concept: str,
    s_curve_papers: Optional[dict[str, Any]] = None,
    s_curve_patents: Optional[dict[str, Any]] = None,
    hype_cycle: Optional[dict[str, Any]] = None,
    strategy: Optional[dict[str, Any]] = None,
    extra_context: Optional[str] = None,
    k_per_probe: int = 4,
    max_chunks: int = 14,
    min_similarity: float = rag.DEFAULT_MIN_SIMILARITY,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """
    Informe prospectivo de secciones fijas, anclado al corpus.

    Coste: 2 llamadas a OpenAI — una de embeddings (las seis sondas van
    batcheadas) y una de generación.

    El resultado incluye:
      · `sections`  — las ocho secciones fijas, ya verificadas,
      · `references` — sólo los documentos REALMENTE citados y recuperados,
      · `retrieved` — todo lo que la recuperación trajo, para poder auditarlo,
      · `citation_warnings` — las etiquetas que el modelo intentó inventarse,
      · el sello `is_ai_generated`.
    """
    concept = (concept or "").strip()
    if not concept:
        raise AssistantError("El concepto no puede estar vacío")

    model = model or default_model()

    # 1 · Recuperación (1 llamada de embeddings para las seis sondas).
    # Las sondas NO llevan el concepto: el corpus es metodológico —trata de
    # cómo se leen las curvas en S, no de biodiésel—, y mezclarle el concepto
    # a la consulta desplaza el vector hacia un tema que el corpus no cubre.
    chunks = rag.search_multi(
        list(REPORT_PROBES),
        k_per_query=k_per_probe,
        min_similarity=min_similarity,
        max_chunks=max_chunks,
    )
    allowed = rag.available_cite_keys(chunks)

    # 2 · Prompt.
    evidence = _build_evidence_block(
        concept, s_curve_papers, s_curve_patents, hype_cycle, strategy, extra_context
    )
    if chunks:
        corpus_block = (
            "CORPUS METODOLÓGICO RECUPERADO (lo único que puedes citar).\n"
            "Claves permitidas: "
            + ", ".join(f"[{k}]" for k in sorted(allowed))
            + "\n\n"
            + rag.build_context(chunks)
        )
    else:
        corpus_block = (
            "CORPUS METODOLÓGICO RECUPERADO: VACÍO.\n"
            "No hay ninguna clave permitida. NO cites absolutamente nada, y declara "
            "en «Limitaciones metodológicas» que este informe no pudo anclarse al "
            "corpus metodológico del estudio."
        )

    user = (
        f"{evidence}\n\n"
        f"{'=' * 70}\n{corpus_block}\n{'=' * 70}\n\n"
        "Redacta el informe devolviendo este JSON exacto:\n"
        f"{_build_schema_block()}"
    )

    # 3 · Generación (1 llamada).
    data = _call_json(REPORT_SYSTEM_PROMPT, user, model=model, max_tokens=2600)

    # 4 · Verificación de las citas contra lo realmente recuperado.
    sections: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    all_cited: set[str] = set()

    for sid, heading, kind, _intent in SECTIONS:
        content, warnings, cited = _walk_section(data.get(sid), kind, allowed)
        all_warnings.extend(warnings)
        all_cited |= cited
        section: dict[str, Any] = {"id": sid, "heading": heading, "kind": kind}
        if kind == "prose":
            section["text"] = content
        else:
            section["items"] = content
        sections.append(section)

    if all_warnings:
        logger.warning(
            "Informe de «%s»: %d citas no respaldadas retiradas — %s",
            concept, len(all_warnings), sorted(set(all_warnings)),
        )

    cited_chunks = [c for c in chunks if c.cite_key in all_cited]
    report: dict[str, Any] = {
        "concept": concept,
        "sections": sections,
        "references": rag.collect_references(cited_chunks),
        "retrieved": [c.to_dict() for c in chunks],
        "retrieved_references": rag.collect_references(chunks),
        "corpus_grounded": bool(chunks),
        "citation_warnings": sorted(set(all_warnings)),
        "rag": {
            "probes": list(REPORT_PROBES),
            "embedding_model": rag.embedding_model(),
            "chunks_retrieved": len(chunks),
            "documents_retrieved": len({c.cite_key for c in chunks}),
            "documents_cited": len(all_cited),
            "min_similarity": min_similarity,
        },
    }
    report.update(_ai_marker(model, openai_calls=2))
    report["markdown"] = report_to_markdown(report)
    return report


# ── Exportación ───────────────────────────────────────────────────────────────

def report_to_markdown(report: dict[str, Any]) -> str:
    """
    Markdown descargable. La marca de IA va en el propio documento: un informe
    que sale de la plataforma no puede perder por el camino quién lo escribió.
    """
    lines = [f"# Informe prospectivo — {report.get('concept', '')}", ""]
    lines.append(
        f"> Generado por **{AI_GENERATOR_NAME}** "
        f"({report.get('ai_provider')}/{report.get('ai_model')}) "
        f"el {report.get('generated_at', '')}. "
        "Contenido generado por IA: requiere validación experta."
    )
    lines.append("")

    for section in report.get("sections", []):
        lines.append(f"## {section['heading']}")
        lines.append("")
        if section.get("kind") == "prose":
            lines.append(section.get("text", ""))
        elif section.get("kind") == "items":
            for item in section.get("items", []):
                title = item.get("title", "").strip()
                text = item.get("text", "").strip()
                lines.append(f"- **{title}**: {text}" if title else f"- {text}")
        else:
            for item in section.get("items", []):
                lines.append(f"- {item}")
        lines.append("")

    references = report.get("references") or []
    lines.append(f"## Referencias del corpus ({len(references)})")
    lines.append("")
    if references:
        for ref in references:
            source = f" — {ref['source']}" if ref.get("source") else ""
            url = f" <{ref['url']}>" if ref.get("url") else ""
            lines.append(f"- **[{ref['cite_key']}]** {ref.get('title', '')}{source}{url}")
    else:
        lines.append(
            "_Ninguna. El informe no pudo anclarse al corpus metodológico._"
        )

    warnings = report.get("citation_warnings") or []
    if warnings:
        lines += [
            "",
            "## Citas retiradas por no estar respaldadas",
            "",
            *(f"- `{w}`" for w in warnings),
        ]
    return "\n".join(lines).rstrip() + "\n"
