"""
backend/core/delphi.py
Métricas del módulo Delphi — implementa la §9 de `docs/LOGICA_DELPHI.md`.

Este módulo es **núcleo puro**: no abre conexiones, no conoce FastAPI y no
importa nada del proyecto. Recibe respuestas ya materializadas y devuelve
diccionarios serializables. Así las métricas se pueden probar con datos
sintéticos de resultado conocido, que es justo lo que el notebook original no
permitía.

Tres bloques, en el orden de la §9:

  1. Por variable y ronda        → `variable_round_stats`
     Población de **agregado** (§6): quienes contestaron esa ronda.

  2. Por variable, entre rondas  → `variable_between_rounds`
     Población de **cambio** (§6): la intersección de las dos rondas, y sólo
     ella. Matriz a/b/c/d, estabilidad, dirección neta, McNemar, corrimiento
     de mediana, reducción de RIQ y Wilcoxon.

  3. Por panel                   → `panel_stats`
     Deserción, la comparación R1-total vs R1-restringido de §6.2 (la que
     detecta deserción no aleatoria), K medio y criterio de parada.

Más dos productos: el mapa de cuatro cuadrantes de §7 (`classify_quadrant`,
`variable_map`) y las correlaciones de Spearman entre variables
(`spearman_matrix`).

Todo el cálculo inferencial está implementado sobre la librería estándar
—chi² de 1 gl vía `erfc`, binomial exacta vía `math.comb`, Spearman con
aproximación t, Wilcoxon con aproximación normal corregida por empates— de
modo que el núcleo no depende de scipy. Si scipy está disponible se usa para
el Wilcoxon exacto, que es más preciso con n pequeño.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # sólo para el Wilcoxon exacto; el resto es librería estándar
    from scipy import stats as _scipy_stats  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy está en requirements, pero no se exige
    _scipy_stats = None  # type: ignore
    _HAS_SCIPY = False


# ══════════════════════════════════════════════════════════════════════════
# Modelo de entrada
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Response:
    """
    Una respuesta de un panelista a una variable en una ronda.

    `value` es None si y sólo si no hay juicio numérico. `is_abstention`
    distingue el NS/NR declarado del simple no-responder: son cosas distintas
    y el notebook las fundía en el 6 (§3, §8).

    `k` es el coeficiente de competencia experta del panelista, K = (Kc+Ka)/2
    (§5.2). Pondera la agregación; si es None se toma 1.0 (panelistas
    intercambiables, que es el comportamiento del ejercicio original).
    """
    participant_id: str
    variable_id: str
    round_number: int
    value: Optional[float] = None
    is_abstention: bool = False
    k: Optional[float] = None
    argument: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Hay un juicio numérico utilizable."""
        return (not self.is_abstention) and self.value is not None

    @property
    def weight(self) -> float:
        return 1.0 if self.k is None else float(self.k)


@dataclass(frozen=True)
class StudyConfig:
    """
    Parámetros del estudio. El umbral de dicotomía es el más consecuente del
    ejercicio y en el notebook vivía en un literal (§2); aquí es configuración
    y puede moverse a posteriori para ver si el mapa aguanta (§5.3).
    """
    dichotomy_threshold: float = 4.0
    scale_min: float = 1.0
    scale_max: float = 5.0
    # Criterio de parada (§9). Declarado antes de abrir la ronda 1.
    stability_min: float = 0.85
    iqr_max: float = 1.0
    # Corte del eje vertical del mapa de cuadrantes (§7).
    consensus_min: float = 0.70
    # Bajo este K se suele excluir al panelista de la agregación (§5.2).
    k_min: Optional[float] = None

    @property
    def scale_range(self) -> float:
        return max(1e-9, float(self.scale_max) - float(self.scale_min))


# Estados de transición (§8). Espejo del enum `transition_state` de la
# migración 001: un estado explícito, nunca una celda en blanco.
MANTIENE_ALTO = "mantiene_alto"
MANTIENE_BAJO = "mantiene_bajo"
SUBIO = "subio"
BAJO = "bajo"
ABSTUVO = "abstuvo"
NO_RESPONDIO = "no_respondio"
VARIABLE_AUSENTE = "variable_ausente"

COMPARABLE_STATES = (MANTIENE_ALTO, BAJO, SUBIO, MANTIENE_BAJO)

# Cuadrantes del mapa de variables (§7).
CERRADA = "CERRADA"
CONVERGIENDO = "CONVERGIENDO"
DISENSO_PERSISTENTE = "DISENSO_PERSISTENTE"
INESTABLE = "INESTABLE"

QUADRANT_ACTION = {
    CERRADA: "Juicio consolidado: no repreguntar.",
    CONVERGIENDO: "El panel se mueve hacia el acuerdo: justifica una ronda más.",
    DISENSO_PERSISTENTE: "Desacuerdo firme: no lo resuelve otra ronda; reportar o buscar la variable oculta.",
    INESTABLE: "Mucho movimiento sin dirección: revisar la redacción del ítem.",
}

# Clasificación por ronda, la de la versión perdida ampliada con el extremo
# bajo (que allí era indistinguible de «en discusión»).
PRIORITARIA = "PRIORITARIA"
EN_DISCUSION = "EN DISCUSIÓN"
DESCARTADA = "DESCARTADA"


# ══════════════════════════════════════════════════════════════════════════
# Estadística descriptiva (librería estándar, sin dependencias)
# ══════════════════════════════════════════════════════════════════════════

def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Cuantil por interpolación lineal (equivalente a numpy `linear`)."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("cuantil de una secuencia vacía")
    if n == 1:
        return float(sorted_values[0])
    pos = (n - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[int(pos)])
    frac = pos - lo
    return float(sorted_values[lo]) * (1 - frac) + float(sorted_values[hi]) * frac


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return _quantile(sorted(values), 0.5)


def _iqr(values: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Devuelve (q1, q3, iqr). El RIQ es la medida clásica de consenso Delphi."""
    if not values:
        return None, None, None
    s = sorted(values)
    q1 = _quantile(s, 0.25)
    q3 = _quantile(s, 0.75)
    return q1, q3, q3 - q1


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _stdev(values: Sequence[float]) -> Optional[float]:
    n = len(values)
    if n < 2:
        return 0.0 if n == 1 else None
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def _round_opt(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), nd)


# ══════════════════════════════════════════════════════════════════════════
# Contrastes
# ══════════════════════════════════════════════════════════════════════════

def _chi2_sf_1df(x: float) -> float:
    """
    P(X² > x) con 1 grado de libertad, en forma cerrada: erfc(sqrt(x/2)).
    Evita depender de scipy para el contraste central del módulo.
    """
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def _binom_two_sided_p(k: int, n: int) -> float:
    """p bilateral de una binomial(n, 0.5) — la versión exacta de McNemar."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar(b: int, c: int, exact_below: int = 25) -> Dict:
    """
    Contraste de McNemar sobre las dos celdas **discordantes** de la matriz de
    la §2: `b` (bajó) y `c` (subió).

    Es la pregunta del ejercicio: ¿el movimiento del panel es *asimétrico*
    —convergencia real hacia una posición— o *simétrico* —ruido—? El notebook
    dejaba b y c en blanco; la tabla estaba armada y sólo faltaba contarla.

    Se reportan siempre las dos variantes:
      · `p_value_chi2`  — χ² con **corrección de continuidad** de Yates:
                          χ² = (|b − c| − 1)² / (b + c), 1 gl.
      · `p_value_exact` — binomial bilateral B(b+c, ½), válida con n pequeño.

    `p_value` es la recomendada: la exacta cuando b + c < `exact_below`
    (convención habitual), la χ² corregida en caso contrario. Con paneles de
    29 personas la exacta es casi siempre la que manda, y es una diferencia
    que importa.
    """
    b, c = int(b), int(c)
    n_disc = b + c

    if n_disc == 0:
        return {
            "b": b, "c": c, "n_discordant": 0,
            "statistic": None,
            "p_value": 1.0,
            "p_value_chi2": None,
            "p_value_exact": 1.0,
            "method": "sin_discordantes",
            "continuity_correction": True,
            "significant_05": False,
            "interpretation": "Nadie cambió de lado: no hay nada que contrastar.",
        }

    chi2 = (abs(b - c) - 1) ** 2 / n_disc  # corrección de continuidad
    p_chi2 = _chi2_sf_1df(chi2)
    p_exact = _binom_two_sided_p(min(b, c), n_disc)

    use_exact = n_disc < exact_below
    p_value = p_exact if use_exact else p_chi2
    method = "binomial_exacta" if use_exact else "chi2_correccion_continuidad"

    if p_value < 0.05:
        direction = "al alza" if c > b else "a la baja"
        interpretation = (
            f"Movimiento asimétrico {direction} (b={b}, c={c}, p={p_value:.4f}): "
            "el panel convergió, no es ruido."
        )
    else:
        interpretation = (
            f"Movimiento simétrico (b={b}, c={c}, p={p_value:.4f}): "
            "los cambios se compensan; no hay evidencia de convergencia direccional."
        )

    return {
        "b": b, "c": c, "n_discordant": n_disc,
        "statistic": round(chi2, 6),
        "p_value": round(p_value, 6),
        "p_value_chi2": round(p_chi2, 6),
        "p_value_exact": round(p_exact, 6),
        "method": method,
        "continuity_correction": True,
        "significant_05": bool(p_value < 0.05),
        "interpretation": interpretation,
    }


def _rank_with_ties(values: Sequence[float]) -> List[float]:
    """Rangos promediados para empates (necesario para Spearman y Wilcoxon)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for idx in order[i:j + 1]:
            ranks[idx] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Dict:
    """
    Rho de Spearman sobre pares completos, con p bilateral por aproximación t
    (gl = n − 2). Es la correlación que muestra la matriz de la vista perdida.
    """
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    n = len(pairs)
    if n < 3:
        return {"rho": None, "p_value": None, "n": n}

    rx = _rank_with_ties([p[0] for p in pairs])
    ry = _rank_with_ties([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:  # una de las dos series es constante
        return {"rho": None, "p_value": None, "n": n}

    rho = num / den
    rho = max(-1.0, min(1.0, rho))
    if abs(rho) >= 1.0:
        p = 0.0
    else:
        t = rho * math.sqrt((n - 2) / (1 - rho * rho))
        p = _student_t_sf_two_sided(abs(t), n - 2)
    return {"rho": round(rho, 4), "p_value": round(p, 6), "n": n}


def _student_t_sf_two_sided(t: float, df: int) -> float:
    """p bilateral de una t de Student, vía la beta incompleta regularizada."""
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    return min(1.0, _betainc_reg(df / 2.0, 0.5, x))


def _betainc_reg(a: float, b: float, x: float) -> float:
    """Beta incompleta regularizada I_x(a,b) por fracción continua (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta)
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.log(1 - x) * b + math.log(x) * a - lbeta
    ) * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-12) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def wilcoxon_signed_rank(before: Sequence[float], after: Sequence[float]) -> Dict:
    """
    Wilcoxon de rangos con signo sobre los pares (§5.3): usa la **magnitud**
    del cambio, no sólo su signo como McNemar. Sólo tiene sentido si la escala
    se mantuvo entre rondas — el ejercicio original la cambió de Likert 1–5 a
    binaria y con eso destruyó toda estadística pareada.

    Con scipy usa el contraste exacto/por defecto; sin scipy, la aproximación
    normal con corrección por empates y por continuidad.
    """
    diffs = [
        float(a) - float(b)
        for b, a in zip(before, after)
        if b is not None and a is not None
    ]
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)

    base = {
        "n_pairs": len(diffs),
        "n_nonzero": n,
        "n_ties": len(diffs) - n,
        "statistic": None,
        "p_value": None,
        "method": None,
        "significant_05": False,
    }
    if n == 0:
        base["method"] = "sin_cambios"
        base["p_value"] = 1.0
        return base

    if _HAS_SCIPY and n >= 1:
        try:
            res = _scipy_stats.wilcoxon(
                nonzero,
                zero_method="wilcox",
                alternative="two-sided",
                mode="auto" if n < 25 else "approx",
            ) if hasattr(_scipy_stats, "wilcoxon") else None
            if res is not None:
                base.update(
                    statistic=float(res.statistic),
                    p_value=round(float(res.pvalue), 6),
                    method="scipy_wilcoxon",
                    significant_05=bool(res.pvalue < 0.05),
                )
                return base
        except Exception:
            pass  # cae a la aproximación normal

    ranks = _rank_with_ties([abs(d) for d in nonzero])
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    w = min(w_plus, w_minus)

    mu = n * (n + 1) / 4.0
    tie_corr = 0.0
    for _, group in Counter(abs(d) for d in nonzero).items():
        tie_corr += group ** 3 - group
    sigma2 = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr / 48.0
    if sigma2 <= 0:
        base.update(statistic=w, p_value=1.0, method="aprox_normal_degenerada")
        return base
    z = (w - mu + 0.5) / math.sqrt(sigma2)
    p = math.erfc(abs(z) / math.sqrt(2))
    base.update(
        statistic=w,
        p_value=round(min(1.0, p), 6),
        method="aprox_normal_correccion_empates",
        significant_05=bool(p < 0.05),
    )
    return base


# ══════════════════════════════════════════════════════════════════════════
# 1 · Por variable y ronda — población de AGREGADO (§6, §9)
# ══════════════════════════════════════════════════════════════════════════

def variable_round_stats(
    responses: Sequence[Response],
    config: StudyConfig,
    n_population: Optional[int] = None,
    variable_id: Optional[str] = None,
    round_number: Optional[int] = None,
) -> Dict:
    """
    Estadística de una variable en una ronda, sobre la **población de
    agregado**: quienes contestaron esa ronda. Es lo que se reporta como
    resultado de la ronda y lo que se devuelve al panel (§5.1).

    `n_population` es el denominador de la tasa de respuesta: el número de
    panelistas a los que se les preguntó. Si no se da, se usa el número de
    respuestas registradas (y la tasa vale 1 por construcción).

    `consenso` está **ponderado por K** (§5.2, §9): se calcula la proporción
    ponderada por competencia experta que queda sobre el umbral y se mide
    cuán lejos está de la división 50/50:

        consenso = 1 − 2·min(p_w, 1 − p_w)

    Vale 1 cuando el panel entero está del mismo lado y 0 en el empate
    perfecto. Se acompaña de `consenso_iqr` = 1 − RIQ/rango, la medida
    clásica de Delphi, que usa la dispersión y no la dicotomía.
    """
    responses = list(responses)
    if config.k_min is not None:
        responses = [r for r in responses if (r.k is None or r.k >= config.k_min)]

    valid = [r for r in responses if r.is_valid]
    abstentions = [r for r in responses if r.is_abstention]
    values = [float(r.value) for r in valid]  # type: ignore[arg-type]

    n_answers = len(valid) + len(abstentions)
    denom = n_population if n_population is not None else n_answers

    q1, q3, iqr = _iqr(values)
    median = _median(values)
    mean = _mean(values)
    std = _stdev(values)

    thr = float(config.dichotomy_threshold)
    n_above = sum(1 for v in values if v >= thr)
    pct_above = (n_above / len(values)) if values else None

    # Proporción ponderada por K (§5.2)
    w_total = sum(r.weight for r in valid)
    w_above = sum(r.weight for r in valid if float(r.value) >= thr)  # type: ignore[arg-type]
    pct_above_weighted = (w_above / w_total) if w_total > 0 else None

    if pct_above_weighted is None:
        consensus = None
    else:
        p = pct_above_weighted
        consensus = 1.0 - 2.0 * min(p, 1.0 - p)

    consensus_iqr = None if iqr is None else max(0.0, 1.0 - iqr / config.scale_range)

    # Distribución: los puntos enteros de la escala más NS/NR aparte, nunca
    # dentro del rango numérico (§3).
    dist: Dict[str, int] = {}
    lo, hi = int(math.floor(config.scale_min)), int(math.ceil(config.scale_max))
    for point in range(lo, hi + 1):
        dist[str(point)] = sum(1 for v in values if round(v) == point)
    dist["NS/NR"] = len(abstentions)

    if mean is None:
        classification = None
    elif mean >= thr:
        classification = PRIORITARIA
    elif mean >= thr - 1:
        classification = EN_DISCUSION
    else:
        classification = DESCARTADA

    return {
        "variable_id": variable_id,
        "round_number": round_number,
        "poblacion": "agregado",
        "n_population": denom,
        "n": len(valid),
        "n_abstentions": len(abstentions),
        "n_no_response": max(0, denom - n_answers) if n_population is not None else 0,
        "response_rate": _round_opt(n_answers / denom if denom else None),
        "abstention_rate": _round_opt(len(abstentions) / n_answers if n_answers else None),
        "mean": _round_opt(mean, 4),
        "median": _round_opt(median, 4),
        "std": _round_opt(std, 4),
        "q1": _round_opt(q1, 4),
        "q3": _round_opt(q3, 4),
        "iqr": _round_opt(iqr, 4),
        "pct_above_threshold": _round_opt(pct_above),
        "pct_above_threshold_weighted": _round_opt(pct_above_weighted),
        "n_above_threshold": n_above,
        "consensus": _round_opt(consensus),
        "consensus_iqr": _round_opt(consensus_iqr),
        "mean_k": _round_opt(_mean([r.weight for r in valid]), 4),
        "distribution": dist,
        "classification": classification,
        "threshold": thr,
    }


# ══════════════════════════════════════════════════════════════════════════
# 2 · Entre rondas — población de CAMBIO (§2, §6, §9)
# ══════════════════════════════════════════════════════════════════════════

def transition_state(
    r_from: Optional[Response],
    r_to: Optional[Response],
    config: StudyConfig,
) -> str:
    """
    El estado explícito de §8. Nunca una ausencia: la celda en blanco del
    notebook significaba cuatro cosas indistinguibles y dos de ellas eran
    bugs de etiqueta (§3).
    """
    if r_from is None or r_to is None:
        return VARIABLE_AUSENTE
    if r_from.is_abstention or r_to.is_abstention:
        return ABSTUVO
    if r_from.value is None or r_to.value is None:
        return NO_RESPONDIO

    thr = float(config.dichotomy_threshold)
    hi_from = float(r_from.value) >= thr
    hi_to = float(r_to.value) >= thr
    if hi_from and hi_to:
        return MANTIENE_ALTO
    if hi_from and not hi_to:
        return BAJO
    if (not hi_from) and hi_to:
        return SUBIO
    return MANTIENE_BAJO


def variable_between_rounds(
    responses_from: Sequence[Response],
    responses_to: Sequence[Response],
    config: StudyConfig,
    variable_id: Optional[str] = None,
    round_from: Optional[int] = None,
    round_to: Optional[int] = None,
) -> Dict:
    """
    Comparación de una variable entre dos rondas, **restringida a la
    población de cambio** (§6): los panelistas que aparecen en ambas. Nunca
    mezcla con la población de agregado, y el resultado lo dice en
    `poblacion: "cambio"`.

    Devuelve la matriz 2×2 de la §2 con sus cuatro celdas contadas —no dos—,
    estabilidad, dirección neta, McNemar, corrimiento de mediana, reducción
    de RIQ y Wilcoxon.
    """
    by_from = {r.participant_id: r for r in responses_from}
    by_to = {r.participant_id: r for r in responses_to}
    shared = sorted(set(by_from) & set(by_to))

    states = Counter()
    paired_from: List[float] = []
    paired_to: List[float] = []
    for pid in shared:
        st = transition_state(by_from[pid], by_to[pid], config)
        states[st] += 1
        if st in COMPARABLE_STATES:
            paired_from.append(float(by_from[pid].value))   # type: ignore[arg-type]
            paired_to.append(float(by_to[pid].value))       # type: ignore[arg-type]

    a = states[MANTIENE_ALTO]
    b = states[BAJO]
    c = states[SUBIO]
    d = states[MANTIENE_BAJO]
    paired = a + b + c + d
    no_comparables = states[ABSTUVO] + states[NO_RESPONDIO] + states[VARIABLE_AUSENTE]

    stability = ((a + d) / paired) if paired else None
    net_direction = c - b

    med_from = _median(paired_from)
    med_to = _median(paired_to)
    _, _, iqr_from = _iqr(paired_from)
    _, _, iqr_to = _iqr(paired_to)

    return {
        "variable_id": variable_id,
        "round_from": round_from,
        "round_to": round_to,
        "poblacion": "cambio",
        "n_change_population": len(shared),
        # Matriz de transición de la §2 — las cuatro celdas
        "matrix": {
            "a_mantiene_alto": a,
            "b_bajo": b,
            "c_subio": c,
            "d_mantiene_bajo": d,
            "pareados": paired,
            "no_comparables": no_comparables,
        },
        "states": {k: v for k, v in states.items()},
        "estabilidad": _round_opt(stability),
        "direccion_neta": net_direction,
        "mcnemar": mcnemar(b, c),
        # Sólo válidos si la escala se mantuvo entre rondas (§5.3)
        "median_from": _round_opt(med_from, 4),
        "median_to": _round_opt(med_to, 4),
        "median_shift": _round_opt(
            None if med_from is None or med_to is None else med_to - med_from, 4
        ),
        "iqr_from": _round_opt(iqr_from, 4),
        "iqr_to": _round_opt(iqr_to, 4),
        "iqr_reduction": _round_opt(
            None if iqr_from is None or iqr_to is None else iqr_from - iqr_to, 4
        ),
        "wilcoxon": wilcoxon_signed_rank(paired_from, paired_to),
    }


# ══════════════════════════════════════════════════════════════════════════
# 3 · Por panel (§6, §9)
# ══════════════════════════════════════════════════════════════════════════

def restricted_aggregate_comparison(
    responses_from: Sequence[Response],
    continuing_ids: Iterable[str],
    config: StudyConfig,
    variable_id: Optional[str] = None,
) -> Dict:
    """
    §6.2 — «Es una comparación barata y nadie la hizo».

    Compara el agregado de la ronda anterior sobre **todos** los que la
    contestaron con el mismo agregado **restringido a los que siguieron** en
    la ronda siguiente. Si difieren, la deserción no fue aleatoria y la
    estabilidad medida sobre la intersección está sesgada.
    """
    cont = set(continuing_ids)
    total = variable_round_stats(responses_from, config, variable_id=variable_id)
    restricted = variable_round_stats(
        [r for r in responses_from if r.participant_id in cont],
        config,
        variable_id=variable_id,
    )
    restricted["poblacion"] = "agregado_restringido"

    def diff(key: str) -> Optional[float]:
        x, y = total.get(key), restricted.get(key)
        return None if x is None or y is None else round(y - x, 4)

    d_pct = diff("pct_above_threshold")
    # Umbral pragmático: 10 puntos porcentuales de diferencia en la
    # proporción sobre el umbral bastan para desconfiar de la intersección.
    biased = d_pct is not None and abs(d_pct) >= 0.10

    return {
        "variable_id": variable_id,
        "total": total,
        "restricted": restricted,
        "delta_pct_above_threshold": d_pct,
        "delta_median": diff("median"),
        "delta_mean": diff("mean"),
        "attrition_non_random": bool(biased),
        "note": (
            "La deserción cambió la composición del panel: leer la estabilidad "
            "sobre la población de cambio con cautela."
            if biased else
            "Los que siguieron opinaban como el conjunto: la deserción no sesga "
            "visiblemente el agregado."
        ),
    }


def panel_stats(
    responses: Sequence[Response],
    config: StudyConfig,
    panelists_k: Optional[Mapping[str, Optional[float]]] = None,
    n_invited: Optional[int] = None,
) -> Dict:
    """
    Salud del panel: participación y deserción por ronda, K medio y la
    comparación §6.2 agregada. Las dos poblaciones de §6 se reportan siempre
    separadas y etiquetadas.
    """
    by_round: Dict[int, set] = defaultdict(set)
    for r in responses:
        by_round[r.round_number].add(r.participant_id)

    rounds = sorted(by_round)
    round_rows = []
    prev: Optional[int] = None
    for rn in rounds:
        participants = by_round[rn]
        row = {
            "round_number": rn,
            "poblacion_agregado": len(participants),
            "response_rate": _round_opt(len(participants) / n_invited) if n_invited else None,
            "poblacion_cambio": None,
            "attrition_rate": None,
            "retention_rate": None,
            "new_panelists": None,
        }
        if prev is not None:
            before = by_round[prev]
            shared = participants & before
            row["poblacion_cambio"] = len(shared)
            row["attrition_rate"] = _round_opt(
                (len(before) - len(shared)) / len(before) if before else None
            )
            row["retention_rate"] = _round_opt(
                len(shared) / len(before) if before else None
            )
            row["new_panelists"] = len(participants - before)
        round_rows.append(row)
        prev = rn

    ks = [
        float(v) for v in (panelists_k or {}).values() if v is not None
    ] or [r.weight for r in responses if r.k is not None]
    active_ids = by_round[rounds[-1]] if rounds else set()
    active_ks = [
        float(v) for pid, v in (panelists_k or {}).items()
        if v is not None and pid in active_ids
    ]

    return {
        "n_invited": n_invited,
        "n_panelists_seen": len({r.participant_id for r in responses}),
        "rounds": round_rows,
        "mean_k_all": _round_opt(_mean(ks), 4),
        "mean_k_active": _round_opt(_mean(active_ks), 4) if active_ks else None,
        "n_below_k_min": (
            sum(1 for v in (panelists_k or {}).values()
                if v is not None and config.k_min is not None and float(v) < config.k_min)
            if config.k_min is not None else None
        ),
    }


# ══════════════════════════════════════════════════════════════════════════
# El mapa de variables en cuatro cuadrantes (§7)
# ══════════════════════════════════════════════════════════════════════════

def classify_quadrant(
    consensus: Optional[float],
    stability: Optional[float],
    config: StudyConfig,
) -> Optional[str]:
    """
    Dos ejes: cuánto acuerdo hay (consenso, ponderado por K) y cuánto se movió
    el panel (estabilidad de la matriz de transición).

        consenso alto + estabilidad alta → CERRADA
        consenso alto + estabilidad baja → CONVERGIENDO
        consenso bajo + estabilidad alta → DISENSO PERSISTENTE
        consenso bajo + estabilidad baja → INESTABLE
    """
    if consensus is None or stability is None:
        return None
    high_c = consensus >= config.consensus_min
    high_s = stability >= config.stability_min
    if high_c and high_s:
        return CERRADA
    if high_c and not high_s:
        return CONVERGIENDO
    if (not high_c) and high_s:
        return DISENSO_PERSISTENTE
    return INESTABLE


def stopping_check(
    consensus: Optional[float],
    stability: Optional[float],
    iqr: Optional[float],
    config: StudyConfig,
) -> Dict:
    """
    Criterio de parada de §9: convencionalmente estabilidad ≥ 85 % **o**
    RIQ ≤ 1 en escala de 5. Configurable y declarado antes de abrir la
    ronda 1, no elegido al ver los resultados.
    """
    by_stability = stability is not None and stability >= config.stability_min
    by_iqr = iqr is not None and iqr <= config.iqr_max
    closed = bool(by_stability or by_iqr)
    return {
        "closed": closed,
        "by_stability": bool(by_stability),
        "by_iqr": bool(by_iqr),
        "stability_min": config.stability_min,
        "iqr_max": config.iqr_max,
        "reason": (
            "estabilidad ≥ umbral" if by_stability else
            "RIQ ≤ umbral" if by_iqr else
            "no cumple ningún criterio: candidata a la ronda siguiente"
        ),
    }


def variable_map(
    round_stats: Sequence[Dict],
    between: Sequence[Dict],
    config: StudyConfig,
    variable_names: Optional[Mapping[str, str]] = None,
) -> Dict:
    """
    Cruza la última estadística por ronda (eje consenso) con la comparación
    entre rondas (eje estabilidad) y clasifica cada variable en un cuadrante.

    Las variables sin par de rondas no se fuerzan a un cuadrante: se listan
    aparte como `sin_clasificar`. Inventarles una estabilidad sería repetir
    el pecado de la celda en blanco.
    """
    last_stats: Dict[str, Dict] = {}
    for s in round_stats:
        vid = s.get("variable_id")
        if vid is None:
            continue
        cur = last_stats.get(vid)
        if cur is None or (s.get("round_number") or 0) >= (cur.get("round_number") or 0):
            last_stats[vid] = s

    last_between: Dict[str, Dict] = {}
    for t in between:
        vid = t.get("variable_id")
        if vid is None:
            continue
        cur = last_between.get(vid)
        if cur is None or (t.get("round_to") or 0) >= (cur.get("round_to") or 0):
            last_between[vid] = t

    items: List[Dict] = []
    unclassified: List[Dict] = []
    for vid, s in last_stats.items():
        t = last_between.get(vid)
        stability = t.get("estabilidad") if t else None
        consensus = s.get("consensus")
        quadrant = classify_quadrant(consensus, stability, config)
        row = {
            "variable_id": vid,
            "variable_name": (variable_names or {}).get(vid, vid),
            "consensus": consensus,
            "consensus_iqr": s.get("consensus_iqr"),
            "stability": stability,
            "iqr": s.get("iqr"),
            "median": s.get("median"),
            "n": s.get("n"),
            "round_number": s.get("round_number"),
            "direccion_neta": t.get("direccion_neta") if t else None,
            "mcnemar_p": (t.get("mcnemar") or {}).get("p_value") if t else None,
            "mcnemar_significant": (t.get("mcnemar") or {}).get("significant_05") if t else None,
            "quadrant": quadrant,
            "action": QUADRANT_ACTION.get(quadrant) if quadrant else None,
            "stopping": stopping_check(consensus, stability, s.get("iqr"), config),
        }
        (items if quadrant else unclassified).append(row)

    counts = Counter(i["quadrant"] for i in items)
    open_vars = [i["variable_id"] for i in items if not i["stopping"]["closed"]]

    return {
        "axes": {
            "x": {"label": "Estabilidad", "threshold": config.stability_min},
            "y": {"label": "Consenso (ponderado por K)", "threshold": config.consensus_min},
        },
        "variables": sorted(items, key=lambda r: (-(r["consensus"] or 0), r["variable_name"])),
        "sin_clasificar": unclassified,
        "counts": {
            CERRADA: counts.get(CERRADA, 0),
            CONVERGIENDO: counts.get(CONVERGIENDO, 0),
            DISENSO_PERSISTENTE: counts.get(DISENSO_PERSISTENTE, 0),
            INESTABLE: counts.get(INESTABLE, 0),
            "sin_clasificar": len(unclassified),
        },
        # Rondas parciales (§7): la ronda N+1 sólo pregunta por estas.
        "next_round_variables": open_vars,
    }


# ══════════════════════════════════════════════════════════════════════════
# Correlaciones de Spearman entre variables
# ══════════════════════════════════════════════════════════════════════════

def spearman_matrix(
    responses: Sequence[Response],
    round_number: Optional[int] = None,
    variable_names: Optional[Mapping[str, str]] = None,
    min_pairs: int = 4,
    top_n: int = 8,
) -> Dict:
    """
    Matriz de correlaciones de Spearman entre variables dentro de una ronda,
    con pares completos por celda (un panelista que se abstuvo en una de las
    dos no cuenta para ese par, y sólo para ese par).
    """
    rs = [r for r in responses if r.is_valid and (round_number is None or r.round_number == round_number)]
    by_var: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rs:
        by_var[r.variable_id][r.participant_id] = float(r.value)  # type: ignore[arg-type]

    vids = sorted(by_var)
    labels = [(variable_names or {}).get(v, v) for v in vids]
    n = len(vids)
    matrix: List[List[Optional[float]]] = [[None] * n for _ in range(n)]
    pairs_out: List[Dict] = []

    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            common = sorted(set(by_var[vids[i]]) & set(by_var[vids[j]]))
            if len(common) < min_pairs:
                continue
            res = spearman(
                [by_var[vids[i]][p] for p in common],
                [by_var[vids[j]][p] for p in common],
            )
            matrix[i][j] = matrix[j][i] = res["rho"]
            if res["rho"] is not None:
                pairs_out.append({
                    "variable_a": vids[i], "variable_b": vids[j],
                    "label_a": labels[i], "label_b": labels[j],
                    "rho": res["rho"], "p_value": res["p_value"], "n": res["n"],
                })

    pairs_out.sort(key=lambda p: -abs(p["rho"]))
    return {
        "round_number": round_number,
        "variable_ids": vids,
        "labels": labels,
        "matrix": matrix,
        "top_pairs": pairs_out[:top_n],
        "n_pairs_computed": len(pairs_out),
    }


# ══════════════════════════════════════════════════════════════════════════
# Devolución anónima entre rondas (§5.1)
# ══════════════════════════════════════════════════════════════════════════

def feedback_package(
    participant_id: str,
    previous_responses: Sequence[Response],
    config: StudyConfig,
    variable_names: Optional[Mapping[str, str]] = None,
    max_arguments: int = 5,
) -> Dict:
    """
    El paquete que ve un panelista al abrir la ronda N+1 (§5.1): su propia
    respuesta anterior, la distribución del panel en esa variable, y los
    argumentos **de quienes opinaron distinto**, anónimos.

    «Sin ello la ronda 2 es una re-votación a ciegas, y es exactamente lo que
    ocurrió.»

    Ningún campo del resultado identifica a otro panelista: los argumentos
    salen sin autor y se etiquetan sólo por el lado del umbral.
    """
    by_var: Dict[str, List[Response]] = defaultdict(list)
    for r in previous_responses:
        by_var[r.variable_id].append(r)

    thr = float(config.dichotomy_threshold)
    items = []
    for vid, rs in sorted(by_var.items()):
        mine = next((r for r in rs if r.participant_id == participant_id), None)
        stats = variable_round_stats(rs, config, variable_id=vid)

        mine_high: Optional[bool] = None
        if mine is not None and mine.is_valid:
            mine_high = float(mine.value) >= thr  # type: ignore[arg-type]

        arguments = []
        for r in rs:
            if r.participant_id == participant_id or not (r.argument or "").strip():
                continue
            side = None
            if r.is_valid:
                side = "alto" if float(r.value) >= thr else "bajo"  # type: ignore[arg-type]
            differs = mine_high is not None and side is not None and (side == "alto") != mine_high
            arguments.append({
                "argument": r.argument.strip(),
                "side": side,
                "differs_from_you": bool(differs),
            })
        # Primero los que piensan distinto: son el motor del método.
        arguments.sort(key=lambda a: not a["differs_from_you"])

        items.append({
            "variable_id": vid,
            "variable_name": (variable_names or {}).get(vid, vid),
            "your_previous_value": (None if mine is None else mine.value),
            "your_previous_abstention": bool(mine.is_abstention) if mine else None,
            "your_side": (None if mine_high is None else ("alto" if mine_high else "bajo")),
            "panel": {
                "n": stats["n"],
                "median": stats["median"],
                "mean": stats["mean"],
                "iqr": stats["iqr"],
                "distribution": stats["distribution"],
                "pct_above_threshold": stats["pct_above_threshold"],
                "consensus": stats["consensus"],
            },
            "arguments": arguments[:max_arguments],
            "n_arguments_total": len(arguments),
        })

    return {
        "participant_id": participant_id,
        "threshold": thr,
        "items": items,
    }


# ══════════════════════════════════════════════════════════════════════════
# Orquestador — el informe completo del estudio
# ══════════════════════════════════════════════════════════════════════════

def compute_study_report(
    responses: Sequence[Response],
    config: StudyConfig,
    variable_names: Optional[Mapping[str, str]] = None,
    variable_families: Optional[Mapping[str, str]] = None,
    panelists_k: Optional[Mapping[str, Optional[float]]] = None,
    n_invited: Optional[int] = None,
    round_population: Optional[Mapping[int, int]] = None,
) -> Dict:
    """
    Todo el bloque §9 de una vez. Es el paso 4 del flujo de §8 («Analizar →
    dispara el cálculo sin intervención»).

    Estructura del resultado:
      · `rounds[]`      — por ronda, `variables[]` con la estadística de
                          agregado y los grupos por familia (agrupador).
      · `between[]`     — por par de rondas consecutivas, población de cambio.
      · `panel`         — salud del panel, deserción, K.
      · `attrition_check` — la comparación §6.2 por variable.
      · `map`           — los cuatro cuadrantes de §7.
      · `spearman`      — correlaciones por ronda.
    """
    responses = list(responses)
    names = dict(variable_names or {})
    families = dict(variable_families or {})

    by_round_var: Dict[Tuple[int, str], List[Response]] = defaultdict(list)
    round_numbers = set()
    variable_ids = set()
    for r in responses:
        by_round_var[(r.round_number, r.variable_id)].append(r)
        round_numbers.add(r.round_number)
        variable_ids.add(r.variable_id)

    round_participants: Dict[int, set] = defaultdict(set)
    for r in responses:
        round_participants[r.round_number].add(r.participant_id)

    rounds_out = []
    all_round_stats: List[Dict] = []
    for rn in sorted(round_numbers):
        pop = (round_population or {}).get(rn, len(round_participants[rn]))
        var_rows = []
        for vid in sorted(variable_ids, key=lambda v: names.get(v, v)):
            rs = by_round_var.get((rn, vid))
            if not rs:
                continue
            s = variable_round_stats(rs, config, n_population=pop, variable_id=vid, round_number=rn)
            s["variable_name"] = names.get(vid, vid)
            s["family"] = families.get(vid)
            var_rows.append(s)
            all_round_stats.append(s)

        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for s in var_rows:
            grouped[s["family"] or "Sin agrupador"].append(s)

        rounds_out.append({
            "round_number": rn,
            "poblacion": "agregado",
            "n_respondents": len(round_participants[rn]),
            "n_population": pop,
            "response_rate": _round_opt(len(round_participants[rn]) / pop if pop else None),
            "n_variables": len(var_rows),
            "variables": var_rows,
            "by_family": [
                {"family": fam, "variables": rows}
                for fam, rows in sorted(grouped.items())
            ],
            "n_prioritarias": sum(1 for s in var_rows if s["classification"] == PRIORITARIA),
            "n_en_discusion": sum(1 for s in var_rows if s["classification"] == EN_DISCUSION),
            "n_descartadas": sum(1 for s in var_rows if s["classification"] == DESCARTADA),
        })

    ordered_rounds = sorted(round_numbers)
    between_out: List[Dict] = []
    attrition_out: List[Dict] = []
    for rn_from, rn_to in zip(ordered_rounds, ordered_rounds[1:]):
        continuing = round_participants[rn_from] & round_participants[rn_to]
        for vid in sorted(variable_ids, key=lambda v: names.get(v, v)):
            rs_from = by_round_var.get((rn_from, vid))
            rs_to = by_round_var.get((rn_to, vid))
            if not rs_from or not rs_to:
                continue
            t = variable_between_rounds(
                rs_from, rs_to, config, variable_id=vid,
                round_from=rn_from, round_to=rn_to,
            )
            t["variable_name"] = names.get(vid, vid)
            t["family"] = families.get(vid)
            between_out.append(t)

            ac = restricted_aggregate_comparison(rs_from, continuing, config, variable_id=vid)
            ac["variable_name"] = names.get(vid, vid)
            ac["round_from"] = rn_from
            ac["round_to"] = rn_to
            attrition_out.append(ac)

    panel = panel_stats(responses, config, panelists_k=panelists_k, n_invited=n_invited)
    vmap = variable_map(all_round_stats, between_out, config, variable_names=names)

    n_biased = sum(1 for a in attrition_out if a["attrition_non_random"])
    panel["attrition_summary"] = {
        "n_variables_checked": len(attrition_out),
        "n_variables_non_random": n_biased,
        "verdict": (
            f"{n_biased} de {len(attrition_out)} variables cambian ≥10 pp al restringir "
            "el agregado a quienes siguieron: la deserción no fue aleatoria."
            if n_biased else
            "Ninguna variable cambia apreciablemente al restringir el agregado: "
            "no hay señal de deserción selectiva."
        ) if attrition_out else "Sin pares de rondas que comparar.",
    }
    panel["quadrants"] = vmap["counts"]
    panel["stopping_rule"] = {
        "stability_min": config.stability_min,
        "iqr_max": config.iqr_max,
        "n_closed": sum(1 for v in vmap["variables"] if v["stopping"]["closed"]),
        "n_open": len(vmap["next_round_variables"]),
        "next_round_variables": vmap["next_round_variables"],
    }

    return {
        "config": {
            "dichotomy_threshold": config.dichotomy_threshold,
            "scale_min": config.scale_min,
            "scale_max": config.scale_max,
            "stability_min": config.stability_min,
            "iqr_max": config.iqr_max,
            "consensus_min": config.consensus_min,
            "k_min": config.k_min,
        },
        "n_rounds": len(ordered_rounds),
        "n_variables": len(variable_ids),
        "rounds": rounds_out,
        "between": between_out,
        "attrition_check": attrition_out,
        "panel": panel,
        "map": vmap,
        "spearman": [
            spearman_matrix(responses, round_number=rn, variable_names=names)
            for rn in ordered_rounds
        ],
    }
