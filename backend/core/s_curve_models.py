"""
Ajuste multi-modelo de curvas en S.

Cinco modelos sigmoidales de TRES parámetros ajustados sobre la misma serie
acumulada, comparados con las mismas métricas y con un criterio de selección
explícito.

Por qué cinco y no uno (ver docs/LOGICA_CURVA_S.md §4):

    El R² sobre un acumulado monótono es engañosamente alto: una logística le
    saca 0.996 a casi cualquier serie. Ese 0.996 no dice que la logística sea
    el modelo correcto, dice que la serie crece. La diferencia entre modelos
    NO está en el R² —están todos por encima de 0.996— sino en el TECHO que
    extrapolan: para la misma serie de 62 038 papers, la logística proyecta
    L = 72 078 y Chapman-Richards L = 132 184. Casi el doble. Y de L sale el
    porcentaje de saturación, que es lo único que decide la fase que muestra
    la UI.

    Ajustar sólo la logística es elegir esa respuesta por omisión. Ajustar
    cinco y comparar convierte la elección en un dato.

Las métricas de la tabla:

    R² aj.          R² penalizado por número de parámetros. Con p igual en
                    los cinco modelos sólo reordena por SSE, pero se reporta
                    porque es lo que se lee primero.
    EE              Error estándar de la estimación, sqrt(SSE/(n-p)), en
                    unidades de la serie (publicaciones o patentes). Dice el
                    error real, que el R² esconde.
    Durbin-Watson   Autocorrelación de los residuos. ~2 = sin estructura.
                    Muy por debajo de 2 = los residuos se arrastran, el
                    modelo tiene sesgo sistemático por tramos y el R² alto
                    es un espejismo. Es la métrica que delata el ajuste que
                    "se ve bien".
    Inflexión       Año de máxima velocidad de adopción. Comparable entre
                    modelos aunque la forma difiera.
    L               Techo de saturación.
    % sat.          Último acumulado observado sobre L.

Selección del mejor modelo: AICc (equivalente a ordenar por SSE cuando todos
tienen los mismos p y n) y, entre los que quedan empatados dentro de ΔAICc ≤ 2,
el de Durbin-Watson más cercano a 2 — porque entre dos ajustes indistinguibles
en error, gana el que no deja estructura en los residuos.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import curve_fit

# Número de parámetros de todos los modelos. Que sea común es lo que hace la
# comparación por AICc equivalente a una comparación por error.
N_PARAMS = 3

# Umbrales de Durbin-Watson. El estadístico va de 0 a 4; 2 es ausencia de
# autocorrelación. Por debajo de 1 la estructura residual es suficientemente
# marcada como para avisar en la UI.
DW_SEVERE = 1.0
DW_MODERATE = 1.5

# Dos modelos dentro de este margen de AICc son indistinguibles en evidencia.
AICC_TIE = 2.0

# Tope para el argumento de exp: evita overflow silencioso durante el ajuste.
_EXP_CLIP = 700.0


def _exp(z: np.ndarray | float) -> np.ndarray:
    return np.exp(np.clip(z, -_EXP_CLIP, _EXP_CLIP))


# ── Las cinco formas funcionales ─────────────────────────────────────────────
#
# Todas se ajustan en tiempo RELATIVO t = año - t0, con t0 = primer año - 1,
# de modo que t ≥ 1. Weibull, Hill y Chapman-Richards lo necesitan porque
# pasan por el origen por construcción; logística y Gompertz son invariantes
# a la traslación y se benefician igual (mejor condicionamiento numérico).
# La inflexión se devuelve siempre reconvertida a año calendario.


def logistic(t, L, tm, k):
    """L / (1 + exp(-k(t - tm))). Simétrica: la inflexión cae siempre en L/2."""
    return L / (1.0 + _exp(-k * (t - tm)))


def gompertz(t, L, tm, k):
    """L·exp(-exp(-k(t - tm))). Asimétrica: inflexión temprana, en L/e ≈ 0.368·L."""
    return L * _exp(-_exp(-k * (t - tm)))


def weibull(t, L, lam, beta):
    """L·(1 - exp(-(t/λ)^β)). La asimetría la fija β; β = 1 la degrada a exponencial."""
    tt = np.clip(np.asarray(t, dtype=float), 1e-9, None)
    return L * (1.0 - _exp(-np.power(tt / max(lam, 1e-9), beta)))


def hill(t, L, kh, n):
    """L·tⁿ / (Kₕⁿ + tⁿ). Kₕ es el tiempo al que se alcanza L/2."""
    tt = np.clip(np.asarray(t, dtype=float), 1e-9, None)
    tn = np.power(tt, n)
    return L * tn / (np.power(max(kh, 1e-9), n) + tn)


def chapman_richards(t, L, k, m):
    """L·(1 - exp(-k·t))^m. Con m > 1 la inflexión está en ln(m)/k."""
    tt = np.clip(np.asarray(t, dtype=float), 0.0, None)
    base = np.clip(1.0 - _exp(-k * tt), 1e-12, None)
    return L * np.power(base, m)


# ── Inflexión analítica de cada forma, en tiempo relativo ────────────────────


def _infl_logistic(L, tm, k):
    return float(tm)


def _infl_gompertz(L, tm, k):
    return float(tm)


def _infl_weibull(L, lam, beta):
    if beta <= 1:
        return None  # cóncava desde el origen: no hay inflexión interior
    return float(lam * ((beta - 1.0) / beta) ** (1.0 / beta))


def _infl_hill(L, kh, n):
    if n <= 1:
        return None
    return float(kh * ((n - 1.0) / (n + 1.0)) ** (1.0 / n))


def _infl_chapman(L, k, m):
    if m <= 1 or k <= 0:
        return None
    return float(np.log(m) / k)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    func: Callable
    param_names: tuple[str, str, str]
    inflection: Callable
    seeds: Callable[[np.ndarray, np.ndarray], list[list[float]]]
    bounds: Callable[[np.ndarray, np.ndarray], tuple[list[float], list[float]]]


def _common_L_bounds(y: np.ndarray) -> tuple[float, float]:
    ymax = float(np.max(y))
    # El techo puede estar muy por encima de lo observado en una serie
    # pre-inflexión (Chapman-Richards da 2.1× lo observado en el caso real),
    # pero sin cota superior el ajuste diverge a millones.
    return max(ymax * 0.5, 1e-6), max(ymax * 50.0, 10.0)


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="logistic",
        label="Logístico (3p)",
        func=logistic,
        param_names=("L", "x0", "k"),
        inflection=_infl_logistic,
        seeds=lambda t, y: [
            [float(np.max(y)) * 1.3, float(np.median(t)), 0.3],
            [float(np.max(y)) * 1.05, float(np.median(t)), 0.1],
            [float(np.max(y)) * 3.0, float(np.max(t)), 0.5],
        ],
        bounds=lambda t, y: (
            [_common_L_bounds(y)[0], float(np.min(t)) - 20, 1e-4],
            [_common_L_bounds(y)[1], float(np.max(t)) + 20, 5.0],
        ),
    ),
    ModelSpec(
        key="gompertz",
        label="Gompertz (3p)",
        func=gompertz,
        param_names=("L", "x0", "k"),
        inflection=_infl_gompertz,
        seeds=lambda t, y: [
            [float(np.max(y)) * 1.5, float(np.median(t)), 0.15],
            [float(np.max(y)) * 1.1, float(np.median(t)), 0.3],
            [float(np.max(y)) * 4.0, float(np.max(t)), 0.08],
        ],
        bounds=lambda t, y: (
            [_common_L_bounds(y)[0], float(np.min(t)) - 30, 1e-4],
            [_common_L_bounds(y)[1], float(np.max(t)) + 30, 5.0],
        ),
    ),
    ModelSpec(
        key="weibull",
        label="Weibull (3p)",
        func=weibull,
        param_names=("L", "lambda", "beta"),
        inflection=_infl_weibull,
        seeds=lambda t, y: [
            [float(np.max(y)) * 1.5, float(np.median(t)), 2.0],
            [float(np.max(y)) * 1.1, float(np.max(t)) * 0.8, 3.0],
            [float(np.max(y)) * 4.0, float(np.max(t)) * 1.5, 1.5],
        ],
        bounds=lambda t, y: (
            [_common_L_bounds(y)[0], 0.1, 0.1],
            [_common_L_bounds(y)[1], float(np.max(t)) * 20, 30.0],
        ),
    ),
    ModelSpec(
        key="hill",
        label="Hill (3p)",
        func=hill,
        param_names=("L", "Kh", "n"),
        inflection=_infl_hill,
        seeds=lambda t, y: [
            [float(np.max(y)) * 1.5, float(np.median(t)), 3.0],
            [float(np.max(y)) * 1.1, float(np.max(t)) * 0.7, 5.0],
            [float(np.max(y)) * 5.0, float(np.max(t)) * 1.5, 2.0],
        ],
        bounds=lambda t, y: (
            [_common_L_bounds(y)[0], 0.1, 0.1],
            [_common_L_bounds(y)[1], float(np.max(t)) * 20, 50.0],
        ),
    ),
    ModelSpec(
        key="chapman_richards",
        label="Chapman-Richards (3p)",
        func=chapman_richards,
        param_names=("L", "k", "m"),
        inflection=_infl_chapman,
        seeds=lambda t, y: [
            [float(np.max(y)) * 1.5, 0.1, 3.0],
            [float(np.max(y)) * 1.1, 0.3, 2.0],
            [float(np.max(y)) * 5.0, 0.05, 8.0],
        ],
        bounds=lambda t, y: (
            [_common_L_bounds(y)[0], 1e-4, 0.1],
            [_common_L_bounds(y)[1], 5.0, 200.0],
        ),
    ),
)

MODELS_BY_KEY = {m.key: m for m in MODELS}


# ── Métricas ─────────────────────────────────────────────────────────────────


def durbin_watson(residuals: Sequence[float] | np.ndarray) -> float | None:
    """
    Σ(eₜ - eₜ₋₁)² / Σeₜ².

    0 ≤ DW ≤ 4. Cerca de 2, los residuos no guardan memoria. Cerca de 0, cada
    residuo se parece al anterior: el modelo va sistemáticamente por encima en
    un tramo y por debajo en otro. En una serie ACUMULADA eso es lo habitual
    incluso con R² > 0.999, y es exactamente el fallo que el R² no ve.
    """
    e = np.asarray(residuals, dtype=float)
    if e.size < 3:
        return None
    denom = float(np.sum(e**2))
    if denom <= 0 or not np.isfinite(denom):
        return None
    return float(np.sum(np.diff(e) ** 2) / denom)


def dw_verdict(dw: float | None) -> str:
    """'none' | 'moderate' | 'severe' | 'unknown'."""
    if dw is None or not np.isfinite(dw):
        return "unknown"
    if dw < DW_SEVERE or dw > (4.0 - DW_SEVERE):
        return "severe"
    if dw < DW_MODERATE or dw > (4.0 - DW_MODERATE):
        return "moderate"
    return "none"


def _aicc(n: int, ss_res: float, n_params: int) -> float | None:
    """AIC corregido para muestra pequeña. k = parámetros + 1 (la varianza)."""
    if n <= 0 or ss_res <= 0:
        return None
    k = n_params + 1
    aic = n * np.log(ss_res / n) + 2 * k
    if n - k - 1 <= 0:
        return float(aic)
    return float(aic + (2 * k * (k + 1)) / (n - k - 1))


# ── Ajuste de un modelo ──────────────────────────────────────────────────────


def _fit_one(spec: ModelSpec, t: np.ndarray, y: np.ndarray, t0: int) -> dict | None:
    """
    Ajusta un modelo probando varias semillas y quedándose con la de menor SSE.

    curve_fit con bounds usa 'trf', que es local: la semilla decide a qué
    mínimo cae. Con tres semillas por modelo el resultado deja de depender de
    la suerte del punto inicial.
    """
    lo, hi = spec.bounds(t, y)
    best: dict | None = None

    for p0 in spec.seeds(t, y):
        p0 = [min(max(v, l + 1e-9), h - 1e-9) for v, l, h in zip(p0, lo, hi)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    spec.func, t, y, p0=p0, bounds=(lo, hi), maxfev=50_000
                )
        except Exception:
            continue

        y_hat = np.asarray(spec.func(t, *popt), dtype=float)
        if not np.all(np.isfinite(y_hat)):
            continue
        ss_res = float(np.sum((y - y_hat) ** 2))
        if not np.isfinite(ss_res):
            continue
        if best is None or ss_res < best["ss_res"]:
            best = {"popt": popt, "pcov": pcov, "y_hat": y_hat, "ss_res": ss_res}

    if best is None:
        return None

    popt = best["popt"]
    pcov = best["pcov"]
    y_hat = best["y_hat"]
    residuals = y - y_hat
    ss_res = best["ss_res"]
    n = int(y.size)

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # R² ajustado en su forma para modelos no lineales: se penaliza con los p
    # parámetros del modelo (no con p-1, que presupone intercepto).
    r2_adj = (
        1.0 - (1.0 - r2) * (n - 1) / (n - N_PARAMS) if n > N_PARAMS else None
    )
    # Error estándar de la estimación, en unidades de la serie.
    se = float(np.sqrt(ss_res / (n - N_PARAMS))) if n > N_PARAMS else None
    rmse = float(np.sqrt(ss_res / n))
    dw = durbin_watson(residuals)
    aicc = _aicc(n, ss_res, N_PARAMS)

    with np.errstate(invalid="ignore"):
        perr = np.sqrt(np.diag(pcov))

    # Parámetros pegados a su cota: el optimizador no encontró un óptimo
    # interior, se quedó en la pared. Pasa sistemáticamente con L en series
    # pre-inflexión —el techo simplemente no está en los datos— y entonces el
    # L que se reporta no es una estimación, es el límite que le pusimos
    # nosotros. Marcarlo es la diferencia entre "el techo son 1.2 M" y "el
    # techo no es estimable".
    at_bound = [
        name
        for name, v, l, h in zip(spec.param_names, popt, lo, hi)
        if abs(v - l) <= 1e-6 * max(1.0, abs(l)) or abs(v - h) <= 1e-6 * max(1.0, abs(h))
    ]
    L_at_bound = spec.param_names[0] in at_bound

    L = float(popt[0])
    t_infl = spec.inflection(*(float(v) for v in popt))
    infl_year = float(t0 + t_infl) if t_infl is not None else None

    last_obs = float(y[-1])
    saturation_pct = round(last_obs / L * 100, 1) if L > 0 else None

    last_year = int(t0 + t[-1])
    # Una inflexión posterior al último dato es extrapolación pura: el modelo
    # la está proyectando, no la midió.
    infl_passed = bool(infl_year is not None and infl_year <= last_year)

    return {
        "key": spec.key,
        "label": spec.label,
        "converged": True,
        "params": {
            name: float(v) for name, v in zip(spec.param_names, popt)
        },
        "param_stderr": {
            name: (round(float(v), 4) if np.isfinite(v) else None)
            for name, v in zip(spec.param_names, perr)
        },
        "r2": round(float(r2), 5),
        "r2_adj": round(float(r2_adj), 5) if r2_adj is not None else None,
        "se": round(se, 3) if se is not None else None,
        "rmse": round(rmse, 3),
        "durbin_watson": round(dw, 3) if dw is not None else None,
        "autocorrelation": dw_verdict(dw),
        "aicc": round(aicc, 2) if aicc is not None else None,
        "inflection_year": round(infl_year, 1) if infl_year is not None else None,
        "inflection_passed": infl_passed,
        "saturation_L": round(L, 1),
        "saturation_pct": saturation_pct,
        "params_at_bound": at_bound,
        # L pegado a la cota → el techo no se midió, se topó. No usarlo.
        "L_at_bound": L_at_bound,
        "residuals": [
            {"year": int(t0 + ti), "residual": round(float(r), 2)}
            for ti, r in zip(t, residuals)
        ],
    }


# ── Comparación ──────────────────────────────────────────────────────────────


def fit_all_models(
    years: Sequence[int], cumulative: Sequence[float]
) -> dict:
    """
    Ajusta los cinco modelos y devuelve la tabla comparativa ordenada por
    evidencia, con el mejor marcado.

    Devuelve `{"models": [...], "best": <key|None>, ...}`. `models` incluye
    también los que no convergieron, con `converged: False`, porque saber que
    un modelo NO ajusta también es información.
    """
    x = np.asarray(years, dtype=float)
    y = np.asarray(cumulative, dtype=float)

    if x.size < 5:
        # Con cuatro puntos y tres parámetros quedan 1 grado de libertad: ni
        # EE ni Durbin-Watson significan nada. Se declara y no se ajusta.
        return {
            "models": [],
            "best": None,
            "best_label": None,
            "n_points": int(x.size),
            "note": (
                "Se requieren al menos 5 años de datos para comparar modelos "
                "de tres parámetros."
            ),
            "inflection_range": None,
            "saturation_range": None,
            "warning": None,
        }

    t0 = int(x[0]) - 1
    t = x - t0

    fits: list[dict] = []
    for spec in MODELS:
        got = _fit_one(spec, t, y, t0)
        if got is None:
            fits.append(
                {
                    "key": spec.key,
                    "label": spec.label,
                    "converged": False,
                    "params": None,
                    "param_stderr": None,
                    "r2": None,
                    "r2_adj": None,
                    "se": None,
                    "rmse": None,
                    "durbin_watson": None,
                    "autocorrelation": "unknown",
                    "aicc": None,
                    "inflection_year": None,
                    "inflection_passed": False,
                    "saturation_L": None,
                    "saturation_pct": None,
                    "params_at_bound": [],
                    "L_at_bound": False,
                    "residuals": [],
                }
            )
        else:
            fits.append(got)

    ok = [f for f in fits if f["converged"] and f["aicc"] is not None]

    best_key = None
    if ok:
        min_aicc = min(f["aicc"] for f in ok)
        # Empate técnico: dentro de ΔAICc ≤ 2 no hay evidencia para separar
        # los modelos por error, así que decide la estructura de los residuos.
        tied = [f for f in ok if f["aicc"] - min_aicc <= AICC_TIE]
        tied.sort(
            key=lambda f: (
                abs((f["durbin_watson"] if f["durbin_watson"] is not None else 0.0) - 2.0),
                f["aicc"],
            )
        )
        best_key = tied[0]["key"]

    for f in fits:
        f["is_best"] = f["key"] == best_key

    # Orden de presentación: el de la tabla original (logístico → Chapman),
    # que es el orden de MODELS. Se deja tal cual para que la fila ganadora
    # no salte de sitio entre análisis.
    best = next((f for f in fits if f["key"] == best_key), None)

    inflections = [
        f["inflection_year"] for f in ok if f["inflection_year"] is not None
    ]
    # Un L pegado a la cota no es una estimación del techo, así que no entra
    # en el rango: metería un límite arbitrario nuestro entre los resultados.
    saturations = [
        f["saturation_L"]
        for f in ok
        if f["saturation_L"] is not None and not f["L_at_bound"]
    ]

    warning = None
    if best is not None and best.get("L_at_bound"):
        # Gana un modelo cuyo techo se topó con la cota: el % de saturación
        # que saldría de ahí es un artefacto del límite, no una medida.
        warning = (
            "El techo de saturación no es estimable: el mejor modelo lo lleva "
            "hasta el límite del ajuste. La inflexión aún no está en los datos."
        )
    elif best is not None and best["autocorrelation"] == "severe":
        warning = (
            f"Ajuste con reservas: residuos autocorrelacionados "
            f"(Durbin-Watson {best['durbin_watson']})."
        )
    elif best is not None and best["autocorrelation"] == "moderate":
        warning = (
            f"Autocorrelación moderada en los residuos "
            f"(Durbin-Watson {best['durbin_watson']}): el R² sobreestima la "
            f"calidad del ajuste."
        )

    return {
        "models": fits,
        "best": best_key,
        "best_label": best["label"] if best else None,
        "best_model": best,
        "n_points": int(x.size),
        "note": None,
        # Dispersión de la inflexión entre modelos: es la incertidumbre REAL
        # sobre el año de inflexión, mucho más honesta que el ±stderr de un
        # solo modelo, que sólo mide ruido dentro de una forma funcional dada.
        "inflection_range": (
            [round(min(inflections), 1), round(max(inflections), 1)]
            if inflections
            else None
        ),
        "saturation_range": (
            [round(min(saturations), 1), round(max(saturations), 1)]
            if saturations
            else None
        ),
        "warning": warning,
        "converged_count": len(ok),
    }


def model_curve(
    model_key: str, params: dict, years: Sequence[float], t0: int
) -> list[float]:
    """Evalúa un modelo ya ajustado sobre un eje de años calendario."""
    spec = MODELS_BY_KEY[model_key]
    t = np.asarray(years, dtype=float) - t0
    values = spec.func(t, *[params[n] for n in spec.param_names])
    return [float(v) for v in np.atleast_1d(values)]
