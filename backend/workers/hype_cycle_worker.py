"""
backend/workers/hype_cycle_worker.py

Runs the full Hype Cycle analysis pipeline as a FastAPI BackgroundTask.
Updates the in-memory _task_store with progress steps and the final result.
"""

import asyncio
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


async def run_hype_cycle_analysis(task_id: str, payload: dict) -> None:
    """
    Async worker launched via FastAPI BackgroundTasks.
    Runs blocking I/O in a thread pool via asyncio.to_thread so the event
    loop stays free.
    """
    from api.routes.hype_cycle import update_task

    def _step(progress: int, message: str, status: str = "running"):
        update_task(task_id, status=status, progress=progress, message=message)

    try:
        _step(5, "Preparando análisis…")

        serp_api_key = os.getenv("SERP_API_KEY", "")
        if not serp_api_key:
            update_task(
                task_id,
                status="failed",
                progress=0,
                message="SERP_API_KEY no configurada",
                error="SERP_API_KEY environment variable is missing",
            )
            return

        # ── 1. Build query ────────────────────────────────────────────────────
        _step(10, "Construyendo ecuación de búsqueda…")

        from core.analysis import QueryBuilder
        search_terms = payload.get("search_terms", [])
        min_year = payload.get("search_parameters", {}).get("min_year", 2015)
        raw_query = QueryBuilder.build_google_query(search_terms, min_year=min_year)

        if not raw_query:
            update_task(
                task_id,
                status="failed",
                progress=0,
                message="No hay términos de búsqueda válidos",
                error="Empty query",
            )
            return

        logger.info("[%s] Query: %s", task_id, raw_query)

        # ── 2. SerpAPI search (blocking → thread) ─────────────────────────────
        _step(20, "Buscando noticias en SerpAPI…")

        from core.analysis import NewsAnalyzer
        analyzer = NewsAnalyzer()

        success, result = await asyncio.to_thread(
            analyzer.perform_news_search, serp_api_key, raw_query
        )

        if not success:
            update_task(
                task_id,
                status="failed",
                progress=20,
                message=f"Error en búsqueda: {result}",
                error=str(result),
            )
            return

        news_results = result
        if not news_results:
            update_task(
                task_id,
                status="failed",
                progress=25,
                message="La búsqueda no devolvió resultados",
                error="No news results found",
            )
            return

        _step(60, f"Analizando {len(news_results)} artículos…")

        # ── 3. Hype Cycle analysis ─────────────────────────────────────────────
        hype_data = await asyncio.to_thread(analyzer.analyze_hype_cycle, news_results)

        if not hype_data:
            update_task(
                task_id,
                status="failed",
                progress=65,
                message=f"Datos insuficientes — {len(news_results)} artículos encontrados pero ninguno pasó el filtro (revisa logs del backend)",
                error="analyze_hype_cycle returned None — 0 usable rows after year/sentiment validation",
            )
            return

        _step(80, f"Fase detectada: {hype_data['phase']} ({hype_data['confidence']:.0%} confianza)")

        # ── 4. Position on curve ──────────────────────────────────────────────
        from core.positioning import HypeCyclePositioner
        positioner = HypeCyclePositioner()
        total_mentions = hype_data["metrics"]["total_mentions"]
        pos_x, pos_y = positioner.calculate_position(
            hype_data["phase"], hype_data["confidence"], total_mentions
        )
        time_to_plateau = positioner.estimate_time_to_plateau(
            hype_data["phase"], hype_data["confidence"]
        )

        _step(90, "Guardando resultado…")

        # ── 5. Build result document (matches HypeCycleQuery frontend type) ───
        technology_name = payload.get("technology_name") or _infer_name(search_terms)
        category_id = payload.get("category_id", "default")
        raw_metrics = hype_data["metrics"]
        inflection = hype_data["inflection_points"]

        # hype_metrics → matches HypeCycleMetrics TypeScript interface
        hype_metrics = {
            "phase": hype_data["phase"],
            "confidence": hype_data["confidence"],
            "total_mentions": raw_metrics["total_mentions"],
            "peak_mentions": raw_metrics["peak_mentions"],
            "latest_year": raw_metrics["latest_year"],
            "sentiment_avg": raw_metrics["avg_sentiment"],
            "sentiment_trend": 0.0,
            "hype_cycle_position_x": pos_x,
            "hype_cycle_position_y": pos_y,
            "time_to_plateau": time_to_plateau,
            "innovation_trigger_year": inflection.get("innovation_trigger", {}).get("year") if inflection.get("innovation_trigger") else None,
            "peak_year": inflection.get("peak", {}).get("year") if inflection.get("peak") else None,
            "trough_year": inflection.get("trough", {}).get("year") if inflection.get("trough") else None,
        }

        # yearly_stats → matches YearlyStat TypeScript interface: {year, mentions, sentiment, growth_rate}
        yearly_stats_frontend = [
            {
                "year": row["year"],
                "mentions": row["mention_count"],
                "sentiment": row["sentiment_mean"],
                "growth_rate": row.get("mention_change"),
            }
            for row in hype_data["yearly_stats"]
        ]

        from collections import Counter, defaultdict

        # news_items — top 50, trimmed fields for transport
        news_items = []
        for n in news_results[:50]:
            kws = n.get("keywords") or []
            kw_strings = [kw[0] if isinstance(kw, (list, tuple)) else str(kw) for kw in kws[:3]]
            news_items.append({
                "title": n.get("title", ""),
                "link": n.get("link", ""),
                "snippet": (n.get("snippet") or "")[:250],
                "source": n.get("source", ""),
                "date": n.get("date", ""),
                "year": n.get("year"),
                "sentiment": float(n.get("sentiment", 0.0)),
                "country": n.get("country"),
                "keywords": kw_strings,
            })

        # country_stats — aggregate by country
        country_data: dict = defaultdict(list)
        for n in news_results:
            c = n.get("country")
            if c:
                country_data[c].append(float(n.get("sentiment", 0.0)))

        country_stats = sorted(
            [
                {
                    "country": country,
                    "mentions": len(sents),
                    "sentiment_avg": round(sum(sents) / len(sents), 3),
                }
                for country, sents in country_data.items()
            ],
            key=lambda x: -x["mentions"]
        )[:20]

        # keyword_stats — top 20 across all results
        all_kws: list = []
        for n in news_results:
            kws = n.get("keywords") or []
            for kw in kws:
                word = (kw[0] if isinstance(kw, (list, tuple)) else str(kw)).strip().lower()
                if word and len(word) > 2:
                    all_kws.append(word)

        keyword_stats = [
            {"keyword": w, "count": c}
            for w, c in Counter(all_kws).most_common(20)
        ]

        query_result = {
            "query_id": task_id,
            "technology_name": technology_name,
            "technology_description": payload.get("technology_description", ""),
            "category_id": category_id,
            "search_query": raw_query,
            "search_terms": search_terms,
            "execution_date": datetime.utcnow().isoformat(),
            "hype_metrics": hype_metrics,
            "yearly_stats": yearly_stats_frontend,
            "search_parameters": payload.get("search_parameters", {}),
            "api_usage": {"serp_api": len(news_results)},
            "version": "2.0",
            "notes": payload.get("notes", ""),
            "news_items": news_items,
            "country_stats": country_stats,
            "keyword_stats": keyword_stats,
        }

        # ── 6. Optionally persist to DynamoDB ────────────────────────────────
        if payload.get("auto_save", True):
            try:
                from storage.dynamodb import get_db
                from storage.hype_cycle import HypeCycleStorage
                storage = HypeCycleStorage(get_db())
                await asyncio.to_thread(
                    storage.save_query,
                    raw_query,
                    search_terms,
                    hype_data,
                    news_results,
                    category_id,
                    payload.get("search_parameters"),
                    payload.get("notes", ""),
                    technology_name,
                    payload.get("technology_description", ""),
                )
                logger.info("[%s] Saved to DynamoDB", task_id)
            except Exception as e:
                logger.warning("[%s] DynamoDB save failed (non-fatal): %s", task_id, e)

        # ── 7. Mark completed ─────────────────────────────────────────────────
        update_task(
            task_id,
            status="completed",
            progress=100,
            message="Análisis completado",
            result=query_result,
        )
        logger.info("[%s] Completed — phase: %s", task_id, hype_data["phase"])

    except Exception as e:
        logger.error("[%s] Unhandled worker error: %s", task_id, e, exc_info=True)
        update_task(
            task_id,
            status="failed",
            progress=0,
            message=f"Error inesperado: {e}",
            error=str(e),
        )


def _infer_name(search_terms: list) -> str:
    """Use the first non-empty search term value as technology name fallback."""
    for t in search_terms:
        v = t.get("value", "").strip().strip('"')
        if v:
            return v
    return "Tecnología"
