"""
backend/scripts/macro_test.py

Prueba macro: corre, para UNA tecnología, los mismos análisis que la app
—curva S de papers (Scopus), curva S de patentes (EPO OPS), matriz
patentes/papers y Hype Cycle (SerpAPI)— y guarda cada resultado en JSON.

Llama a las funciones de `core` directamente, sin pasar por HTTP: es el mismo
código que ejecutan los endpoints, pero aquí el resultado no se pierde al
cerrar la pantalla.

Diferencias deliberadas frente a la app (y que el informe declara):

  · Curvas S desde `--year-from` (por defecto 1960) para traer la serie
    completa. Los años sin registros quedan en 0.
  · Hype Cycle limitado a los últimos 10 años: se usa la búsqueda por rangos
    de fecha de la app con `start_year = año_actual - 9`, y las noticias sin
    fecha legible se DESCARTAN en vez de asignarse al año en curso (la app
    las fuerza al año actual, lo que infla el último punto).

Uso:
    cd backend
    .venv/bin/python -m scripts.macro_test --spec ../data/macro_test/<slug>/spec.json
    .venv/bin/python -m scripts.macro_test --spec ... --only papers patents
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
load_dotenv(BACKEND / ".env")

from core.s_curve import (  # noqa: E402
    analyze_papers_by_count,
    analyze_patents_epo,
    compare_papers_patents,
)

STEPS = ("papers", "patents", "compare", "hype", "gpatents")


def _dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    print(f"  → {path}")


async def _drain(gen, label: str):
    """Consume el generador SSE, imprime el progreso y devuelve el resultado."""
    last = None
    async for evt in gen:
        status = evt.get("status")
        if status == "error":
            raise RuntimeError(f"{label}: {evt.get('message')}")
        msg = evt.get("message")
        if msg and msg != last:
            print(f"  [{label} {evt.get('progress', '')}%] {msg}", flush=True)
            last = msg
        if status == "completed":
            return evt["result"]
    raise RuntimeError(f"{label}: el análisis terminó sin resultado")


async def run_papers(spec: dict, out: Path) -> dict:
    key = os.getenv("SCOPUS_API_KEY", "").strip()
    res = await _drain(
        analyze_papers_by_count(
            query=spec["scopus_query"],
            api_key=key,
            year_from=spec["year_from"],
            year_to=spec["year_to"],
        ),
        "scopus",
    )
    _dump(out / "papers.json", res)
    return res


async def run_patents(spec: dict, out: Path) -> dict:
    """
    Conteo anual en EPO OPS que se puede reanudar.

    La app, cuando OPS bloquea por cuota o detección de robots, devuelve los
    años obtenidos hasta ahí como si fuera la serie completa. Aquí cada año
    contado se guarda en `patents_counts.json`; ante un bloqueo se espera y se
    sigue desde el año pendiente. La curva se arma con el mismo
    `_assemble_result` que usa `analyze_patents_epo`.
    """
    import numpy as np
    from core.epo import EPOClient, EPOQuotaError, build_cql
    from core.s_curve import _assemble_result

    cache_path = out / "patents_counts.json"
    counts: dict[str, int] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    delay = spec.get("epo_delay", 6.0)
    waits = 0
    requests = 0

    async with EPOClient() as epo:
        await epo.authenticate()
        for year in range(spec["year_from"], spec["year_to"] + 1):
            if str(year) in counts:
                continue
            while True:
                try:
                    n = await epo.count(build_cql(spec["epo_query"], year))
                    requests += 1
                    break
                except EPOQuotaError as exc:
                    waits += 1
                    if waits > spec.get("epo_max_waits", 40):
                        raise RuntimeError(f"epo: demasiados bloqueos ({exc})")
                    print(f"  [epo] bloqueo en {year}; espera 180 s ({waits})", flush=True)
                    await asyncio.sleep(180)
                    await epo.authenticate(force=True)
            counts[str(year)] = n
            cache_path.write_text(json.dumps(counts))
            print(f"  [epo] {year}: {n:,}", flush=True)
            await asyncio.sleep(delay)

    c = {int(k): v for k, v in counts.items()}
    years = list(range(min(c), max(c) + 1))
    annual = [int(c.get(y, 0)) for y in years]
    res = _assemble_result(
        query=spec["epo_query"],
        years=years,
        annual=annual,
        cumulative=[int(v) for v in np.cumsum(annual)],
        total_reported=sum(annual),
        total_retrieved=sum(annual),
        source="epo",
        source_label="patentes (EPO OPS)",
        extra={"epo_requests": requests, "epo_failed_years": [], "epo_cql": build_cql(spec["epo_query"]),
               "epo_blocked_waits": waits},
    )
    _dump(out / "patents.json", res)
    return res


def run_hype(spec: dict, out: Path) -> dict:
    from core.analysis import NewsAnalyzer
    from core.positioning import HypeCyclePositioner

    current = spec.get("hype_year_to", spec["year_to"])
    start = current - (spec.get("hype_years", 10) - 1)

    # La app espera 30 s por petición; Google News con num=100 los supera a
    # menudo y el rango entero se pierde. Se amplía sólo en esta prueba.
    import requests
    import core.analysis as analysis_mod

    timeout = spec.get("hype_timeout", 90)

    class _Requests:
        @staticmethod
        def get(*a, **kw):
            kw["timeout"] = timeout
            return requests.get(*a, **kw)

    analysis_mod.requests = _Requests

    analyzer = NewsAnalyzer()
    t0 = time.time()
    ok, news = analyzer._search_complex_query(
        os.getenv("SERP_API_KEY", ""), spec["hype_query"], start, current
    )
    if not ok:
        raise RuntimeError(f"serpapi: {news}")
    print(f"  [serpapi] {len(news)} noticias únicas en {time.time() - t0:.0f}s")

    dated = [n for n in news if n.get("year") and start <= int(n["year"]) <= current]
    print(f"  [serpapi] {len(dated)} con fecha dentro de {start}–{current}")

    hype = analyzer.analyze_hype_cycle(dated)
    if not hype:
        raise RuntimeError("hype: sin datos utilizables")

    pos = HypeCyclePositioner()
    hype["time_to_plateau"] = pos.estimate_time_to_plateau(hype["phase"], hype["confidence"])
    hype["window"] = [start, current]
    hype["query"] = spec["hype_query"]
    hype["news_total"] = len(news)
    hype["news_dated_in_window"] = len(dated)

    _dump(out / "hype_news.json", news)
    _dump(out / "hype.json", hype)
    return hype


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--only", nargs="*", choices=STEPS)
    args = ap.parse_args()

    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text())
    spec.setdefault("year_from", 1960)
    spec.setdefault("year_to", datetime.now().year)
    out = spec_path.parent
    steps = args.only or STEPS

    print(f"== {spec['name']}  ({spec['year_from']}–{spec['year_to']})")

    if "papers" in steps:
        print("· Papers (Scopus)")
        await run_papers(spec, out)
    if "patents" in steps:
        print("· Patentes (EPO OPS)")
        await run_patents(spec, out)
    if "compare" in steps:
        print("· Matriz patentes/papers")
        papers = json.loads((out / "papers.json").read_text())
        patents = json.loads((out / "patents.json").read_text())
        _dump(out / "compare.json", compare_papers_patents(papers, patents))
    if "hype" in steps:
        print("· Hype Cycle (SerpAPI)")
        await asyncio.to_thread(run_hype, spec, out)
    if "gpatents" in steps:
        print("· Patentes (Google Patents vía SerpAPI)")
        res = await asyncio.to_thread(run_google_patents, spec, out)
        papers = json.loads((out / "papers.json").read_text())
        _dump(out / "compare_google.json", compare_papers_patents(papers, res))


def run_google_patents(spec: dict, out: Path) -> dict:
    """
    Serie de patentes desde Google Patents, guardada aparte de la de EPO
    (`patents_google.json`): cubre más oficinas y busca en texto completo, así
    que no es comparable con la serie de EPO.

    Los años antes de `gpatents_split_year` se piden en una sola consulta y
    entran como acumulado del año anterior al corte.
    """
    import numpy as np
    from core.google_patents import yearly_counts
    from core.s_curve import _assemble_result

    t0 = time.time()
    got = yearly_counts(
        spec["gpatents_query"],
        year_from=spec["year_from"],
        year_to=spec["year_to"],
        split_year=spec.get("gpatents_split_year", 1985),
        on_progress=lambda m: print(f"  [gpatents] {m}", flush=True),
    )
    c = got["counts"]
    years = sorted(c)
    annual = [int(c[y]) for y in years]
    cumulative = [int(v) + got["base"] for v in np.cumsum(annual)]
    res = _assemble_result(
        query=spec["gpatents_query"],
        years=years,
        annual=annual,
        cumulative=cumulative,
        total_reported=sum(annual) + got["base"],
        total_retrieved=sum(annual) + got["base"],
        source="google_patents",
        source_label="patentes (Google Patents)",
        extra={"gpatents_requests": got["requests"], "gpatents_base": got["base"],
               "gpatents_base_range": got["base_range"], "seconds": round(time.time() - t0, 1)},
    )
    print(f"  [gpatents] {got['requests']} créditos en {time.time() - t0:.0f}s")
    _dump(out / "patents_google.json", res)
    return res


if __name__ == "__main__":
    asyncio.run(main())
