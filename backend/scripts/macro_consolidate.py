"""
backend/scripts/macro_consolidate.py

Reúne en un solo JSON los resultados de `macro_test.py` de varias
tecnologías: specs, series anuales, tabla de cinco modelos, comparación
papers/patentes, Hype Cycle y la posición en la matriz patentes/papers.

La matriz sigue la convención de las presentaciones de Zartha (Toma de
decisiones en Agroindustria…): eje X = año de inflexión de papers, eje Y =
año de inflexión de patentes, ambos cortados en el año de referencia.

    inflexión papers  ≤ ref  y  patentes ≤ ref  → Madurez y Optimización
    inflexión papers  > ref  y  patentes ≤ ref  → Explotación Industrial y Ciencia Aplicada
    inflexión papers  ≤ ref  y  patentes > ref  → Oportunidad de Disrupción
    inflexión papers  > ref  y  patentes > ref  → Vigilancia Tecnológica y Apuestas a Futuro

Para cada serie se reportan dos inflexiones: la del mejor modelo por AICc
(lo que muestra la app) y la MEDIANA de los modelos que convergieron, que es
la que se usa para ubicar la tecnología. La mediana no deja que un solo
modelo que extrapola décadas fuera de los datos decida el cuadrante; el
rango entre modelos acompaña al punto como barra de incertidumbre.

Uso:
    cd backend
    .venv/bin/python -m scripts.macro_consolidate ../data/macro_test
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

QUADRANTS = {
    (True, True): ("maturity", "Maturity and Optimization", "Madurez y Optimización"),
    (False, True): ("industrial", "Industrial Exploitation and Applied Science", "Explotación Industrial y Ciencia Aplicada"),
    (True, False): ("disruption", "Disruption Opportunity", "Oportunidad de Disrupción"),
    (False, False): ("watch", "Technology Watch and Future Bets", "Vigilancia Tecnológica y Apuestas a Futuro"),
}


def _series(res: dict) -> dict:
    comp = res.get("model_comparison") or {}
    models = [m for m in comp.get("models", []) if m.get("converged")]
    infl = [m["inflection_year"] for m in models if m.get("inflection_year") is not None]
    best = comp.get("best_model") or {}
    return {
        "query": res.get("query"),
        "year_from": res.get("year_from"),
        "year_to": res.get("year_to"),
        "total": res.get("total_retrieved") or res.get("total_publications"),
        "yearly": [{"year": d["year"], "annual": d["annual"], "cumulative": d["cumulative"]} for d in res.get("yearly_data", [])],
        "peak_annual": res.get("peak_annual"),
        "peak_annual_year": res.get("peak_annual_year"),
        "phase": res.get("phase"),
        "best_model": {
            "key": best.get("key"),
            "label": best.get("label"),
            "inflection_year": best.get("inflection_year"),
            "saturation_L": best.get("saturation_L"),
            "saturation_pct": best.get("saturation_pct"),
            "durbin_watson": best.get("durbin_watson"),
        },
        "inflection_median": round(statistics.median(infl), 1) if infl else None,
        "inflection_range": comp.get("inflection_range"),
        "saturation_range": comp.get("saturation_range"),
        "warning": comp.get("warning"),
        "models": [
            {k: m.get(k) for k in (
                "key", "label", "params", "r2_adj", "se", "durbin_watson", "aicc",
                "inflection_year", "inflection_passed", "saturation_L", "saturation_pct", "is_best",
            )}
            for m in models
        ],
    }


def _hype(h: dict, news: list) -> dict:
    return {
        "query": h.get("query"),
        "window": h.get("window"),
        "app_phase": h.get("phase"),
        "app_confidence": h.get("confidence"),
        "app_time_to_plateau": h.get("time_to_plateau"),
        "inflection_points": h.get("inflection_points"),
        "metrics": h.get("metrics"),
        "yearly": [
            {k: r.get(k) for k in ("year", "mention_count", "sentiment_mean", "sentiment_std")}
            for r in h.get("yearly_stats", [])
        ],
        "news_total": h.get("news_total"),
        "news": [{k: n.get(k) for k in ("year", "date", "source", "title", "link", "sentiment")} for n in news],
    }


def consolidate(root: Path, ref_year: int) -> dict:
    techs = []
    for spec_path in sorted(root.glob("*/spec.json")):
        d = spec_path.parent
        spec = json.loads(spec_path.read_text())
        entry: dict = {"slug": d.name, "spec": spec, "status": {}}
        for name in ("papers", "patents", "compare", "hype"):
            entry["status"][name] = (d / f"{name}.json").exists()

        if entry["status"]["papers"]:
            entry["papers"] = _series(json.loads((d / "papers.json").read_text()))
        if entry["status"]["patents"]:
            entry["patents"] = _series(json.loads((d / "patents.json").read_text()))
        if entry["status"]["compare"]:
            c = json.loads((d / "compare.json").read_text())
            entry["app_comparison"] = {k: c.get(k) for k in ("lag_years", "direction", "interpretation", "matrix", "caveats")}
        if entry["status"]["hype"]:
            news = json.loads((d / "hype_news.json").read_text()) if (d / "hype_news.json").exists() else []
            entry["hype"] = _hype(json.loads((d / "hype.json").read_text()), news)

        if (d / "hype_trends.json").exists():
            ht = json.loads((d / "hype_trends.json").read_text())
            entry["hype_trends"] = {
                "source": ht.get("source"),
                "fit_end": ht.get("fit_end"),
                "terms": {
                    term: {
                        "phase": r["phase"],
                        "fit": {k: v for k, v in r["fit"].items() if k != "fitted"},
                        "fitted": r["fit"].get("fitted"),
                        "annual_mean": r["annual_mean"],
                        "monthly": [
                            {"date": x["date"], "value": x["values"][0].get("extracted_value", 0)}
                            for x in ht.get("raw", {}).get(term, [])
                        ],
                    }
                    for term, r in ht.get("results", {}).items()
                },
            }

        p, t = entry.get("papers"), entry.get("patents")
        if p and t and p["inflection_median"] and t["inflection_median"]:
            px, ty = p["inflection_median"], t["inflection_median"]
            qid, qen, qes = QUADRANTS[(px <= ref_year, ty <= ref_year)]
            entry["matrix"] = {
                "ref_year": ref_year,
                "papers_inflection": px,
                "patents_inflection": ty,
                "papers_range": p["inflection_range"],
                "patents_range": t["inflection_range"],
                "quadrant": qid,
                "quadrant_en": qen,
                "quadrant_es": qes,
                # El cuadrante es firme si TODO el rango de modelos cae del
                # mismo lado del corte en ambos ejes.
                "robust": all(
                    (r[0] > ref_year) == (r[1] > ref_year)
                    for r in (p["inflection_range"], t["inflection_range"]) if r
                ),
            }
        techs.append(entry)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reference_year": ref_year,
        "matrix_convention": (
            "x = papers inflection year, y = patents inflection year (median of "
            "converged S-curve models), split at the reference year."
        ),
        "technologies": techs,
    }


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/macro_test").resolve()
    ref = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    out = consolidate(root, ref)
    dest = root / "macro_results.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    for t in out["technologies"]:
        m = t.get("matrix") or {}
        print(f"{t['slug']:<28} {t['status']}  {m.get('quadrant', '—'):<12} "
              f"papers {m.get('papers_inflection')}  patents {m.get('patents_inflection')}  robust={m.get('robust')}")
    print(f"→ {dest}")
