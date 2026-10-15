"""
HypeCycle storage layer — ported from src/hype_cycle_storage.py
Streamlit, Plotly and st.cache_data dependencies removed.
"""

import logging
import math
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from .dynamodb import DynamoDBStorage

try:
    from boto3.dynamodb.conditions import Attr
    BOTO3_CONDITIONS_AVAILABLE = True
except ImportError:
    BOTO3_CONDITIONS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Domain models ─────────────────────────────────────────────

class HypeCyclePhase(str, Enum):
    INNOVATION_TRIGGER    = "Innovation Trigger"
    PEAK_EXPECTATIONS     = "Peak of Inflated Expectations"
    TROUGH_DISILLUSIONMENT = "Trough of Disillusionment"
    SLOPE_ENLIGHTENMENT   = "Slope of Enlightenment"
    PLATEAU_PRODUCTIVITY  = "Plateau of Productivity"
    PRE_INNOVATION        = "Pre-Innovation Trigger"
    UNKNOWN               = "Unknown"


@dataclass
class HypeCycleMetrics:
    phase: str
    confidence: float
    total_mentions: int
    peak_mentions: int
    latest_year: int
    sentiment_avg: float
    sentiment_trend: float
    hype_cycle_position_x: float = 50.0
    hype_cycle_position_y: float = 50.0
    time_to_plateau: str = "N/A"
    innovation_trigger_year: Optional[int] = None
    peak_year: Optional[int] = None
    trough_year: Optional[int] = None
    inflection_points: Optional[Dict] = None


# ── HypeCycleStorage ──────────────────────────────────────────

class HypeCycleStorage:
    """
    Specialized storage for Hype Cycle queries on top of DynamoDBStorage.
    Includes an in-memory TTL cache to reduce DynamoDB reads.
    """

    CACHE_TTL = 300  # seconds

    def __init__(self, db: DynamoDBStorage):
        if not hasattr(db, "dynamodb"):
            raise ValueError("HypeCycleStorage requires a DynamoDBStorage instance")
        self.db = db
        self._cache: Dict[str, Any] = {}
        self._cache_ts: Dict[str, float] = {}

    # ── Cache helpers ─────────────────────────────────────────

    def _cache_valid(self, key: str) -> bool:
        return key in self._cache and (time.time() - self._cache_ts.get(key, 0)) < self.CACHE_TTL

    def _cache_set(self, key: str, data: Any):
        self._cache[key] = data
        self._cache_ts[key] = time.time()

    def _cache_get(self, key: str) -> Any:
        return self._cache.get(key)

    def _cache_invalidate(self):
        self._cache.clear()
        self._cache_ts.clear()

    # ── Save ──────────────────────────────────────────────────

    def save_query(
        self,
        search_query: str,
        search_terms: List[Dict],
        hype_analysis_results: Dict,
        news_results: List[Dict],
        category_id: str = "default",
        search_parameters: Optional[Dict] = None,
        notes: str = "",
        technology_name: Optional[str] = None,
        technology_description: str = "",
        processing_time: float = 0.0,
    ) -> Optional[str]:
        """Save a Hype Cycle query to DynamoDB. Returns query_id or None."""
        try:
            hype_metrics = self._extract_metrics(hype_analysis_results)

            # Attempt to calculate position if positioner is available
            try:
                from core.positioning import HypeCyclePositioner  # type: ignore
                positioner = HypeCyclePositioner()
                pos_x, pos_y = positioner.calculate_position(
                    hype_metrics.phase,
                    hype_metrics.confidence,
                    hype_metrics.total_mentions,
                )
                hype_metrics.hype_cycle_position_x = pos_x
                hype_metrics.hype_cycle_position_y = pos_y
                hype_metrics.time_to_plateau = positioner.estimate_time_to_plateau(
                    hype_metrics.phase, hype_metrics.confidence
                )
            except Exception:
                pass  # positioner not yet available; defaults apply

            category_name = self._get_category_name(category_id)
            if not technology_name:
                technology_name = self._extract_technology_name(search_query, search_terms)

            query_id = f"hype_{int(time.time() * 1000)}_{uuid.uuid4().hex[:12]}"
            execution_timestamp = datetime.now(timezone.utc).isoformat()

            item: Dict[str, Any] = {
                "analysis_id": query_id,
                "timestamp": execution_timestamp,
                "query_id": query_id,
                "analysis_type": "hype_cycle",
                "category_id": str(category_id),
                "category_name": category_name,
                "search_query": search_query,
                "technology_name": technology_name,
                "technology_description": technology_description,
                "notes": notes,
                "execution_date": execution_timestamp,
                "last_updated": execution_timestamp,
                "is_active": True,
                "version": "1.0",
                "created_by": "hype_cycle_analyzer",
                "processing_time": processing_time,
                "hype_metrics": {
                    "phase": hype_metrics.phase,
                    "confidence": _safe_float(hype_metrics.confidence),
                    "total_mentions": _safe_int(hype_metrics.total_mentions),
                    "peak_mentions": _safe_int(hype_metrics.peak_mentions),
                    "latest_year": _safe_int(hype_metrics.latest_year),
                    "sentiment_avg": _safe_float(hype_metrics.sentiment_avg),
                    "sentiment_trend": _safe_float(hype_metrics.sentiment_trend),
                    "hype_cycle_position_x": _safe_float(hype_metrics.hype_cycle_position_x),
                    "hype_cycle_position_y": _safe_float(hype_metrics.hype_cycle_position_y),
                    "time_to_plateau": hype_metrics.time_to_plateau,
                },
                "search_terms": _clean_search_terms(search_terms),
                "api_usage": {
                    "total_results": len(news_results),
                    "search_timestamp": execution_timestamp,
                    "api_provider": "SerpAPI",
                },
                "search_parameters": search_parameters or {},
                "data_quality": {
                    "total_results": len(news_results),
                    "quality_score": _calculate_quality(news_results),
                },
                "news_sample": _make_news_sample(news_results[:5]),
            }

            saved = self.db.put_analysis_raw(item)
            if saved:
                self._cache_invalidate()
                logger.info("HC query saved: %s", query_id)
                return query_id
            return None

        except Exception as e:
            logger.error("Error saving HC query: %s", e)
            return None

    # ── Read ──────────────────────────────────────────────────

    def get_queries(self, category_id: Optional[str] = None) -> List[Dict]:
        """Return all HC queries, optionally filtered by category."""
        cache_key = f"queries_{category_id or 'all'}"
        if self._cache_valid(cache_key):
            return self._cache_get(cache_key)

        try:
            if category_id:
                result = self._get_by_category(category_id)
            else:
                result = self._get_all_hc_queries()

            result.sort(key=lambda x: x.get("execution_date", ""), reverse=True)
            self._cache_set(cache_key, result)
            return result
        except Exception as e:
            logger.error("Error fetching HC queries: %s", e)
            return []

    def get_query_by_id(self, query_id: str) -> Optional[Dict]:
        cache_key = f"query_{query_id}"
        if self._cache_valid(cache_key):
            return self._cache_get(cache_key)

        item = self.db.get_analysis_by_id(query_id)
        if item:
            self._cache_set(cache_key, item)
        return item

    def _get_by_category(self, category_id: str) -> List[Dict]:
        if not BOTO3_CONDITIONS_AVAILABLE:
            items = self.db.get_analyses_by_category(category_id)
            return [i for i in items if i.get("analysis_type") == "hype_cycle"]

        items: List[Dict] = []
        resp = self.db.analyses_table.scan(
            FilterExpression=(
                Attr("category_id").eq(str(category_id))
                & Attr("analysis_type").eq("hype_cycle")
            )
        )
        items.extend(resp.get("Items", []))
        while "LastEvaluatedKey" in resp:
            resp = self.db.analyses_table.scan(
                FilterExpression=(
                    Attr("category_id").eq(str(category_id))
                    & Attr("analysis_type").eq("hype_cycle")
                ),
                ExclusiveStartKey=resp["LastEvaluatedKey"],
            )
            items.extend(resp.get("Items", []))
        return [DynamoDBStorage.from_decimal(i) for i in items]

    def _get_all_hc_queries(self) -> List[Dict]:
        all_items = self.db.get_all_analyses()
        return [i for i in all_items if i.get("analysis_type") == "hype_cycle"]

    # ── Helpers ───────────────────────────────────────────────

    def _get_category_name(self, category_id: str) -> str:
        try:
            cat = self.db.get_category_by_id(category_id)
            return cat.get("name", "Sin categoría") if cat else "Sin categoría"
        except Exception:
            return "Sin categoría"

    def _extract_metrics(self, hype_results: Dict) -> HypeCycleMetrics:
        try:
            metrics_data = hype_results.get("metrics", {})
            return HypeCycleMetrics(
                phase=hype_results.get("phase", "Unknown"),
                confidence=_safe_float(hype_results.get("confidence", 0.0)),
                total_mentions=_safe_int(metrics_data.get("total_mentions", 0)),
                peak_mentions=_safe_int(metrics_data.get("peak_mentions", 0)),
                latest_year=_safe_int(metrics_data.get("latest_year", datetime.now().year)),
                sentiment_avg=_safe_float(hype_results.get("sentiment_avg", 0.0)),
                sentiment_trend=_safe_float(hype_results.get("sentiment_trend", 0.0)),
            )
        except Exception as e:
            logger.warning("Error extracting HC metrics: %s", e)
            return HypeCycleMetrics(
                phase="Unknown",
                confidence=0.0,
                total_mentions=0,
                peak_mentions=0,
                latest_year=datetime.now().year,
                sentiment_avg=0.0,
                sentiment_trend=0.0,
            )

    @staticmethod
    def _extract_technology_name(search_query: str, search_terms: List[Dict]) -> str:
        try:
            if search_terms:
                parts = []
                valid = [
                    t for t in search_terms
                    if (
                        t.get("value", "").strip().strip('"')
                        and len(t.get("value", "").strip().strip('"')) > 1
                        and t.get("value", "").lower() not in ("and", "or", "not")
                        and "after:" not in t.get("value", "").lower()
                        and "before:" not in t.get("value", "").lower()
                    )
                ]
                for i, term in enumerate(valid):
                    value = term.get("value", "").strip().strip('"').title()
                    parts.append(value)
                    if i < len(valid) - 1:
                        parts.append(term.get("operator", "AND"))
                if parts:
                    if len(parts) > 7:
                        return " ".join(parts[:5]) + " ..."
                    return " ".join(parts)

            if search_query:
                clean = search_query.replace('"', '')
                clean = re.sub(r'\s*after:\d{4}(?:-\d{2}-\d{2})?\s*', ' ', clean)
                clean = re.sub(r'\s*before:\d{4}(?:-\d{2}-\d{2})?\s*', ' ', clean)
                clean = re.sub(r'\s+', ' ', clean).strip()
                if clean and len(clean) <= 80:
                    return clean.title()
                words = clean.split()
                return " ".join(words[:6]).title() + "..." if len(words) > 6 else clean.title()

            return "Tecnología"
        except Exception:
            return "Tecnología"


# ── Pure utility functions ─────────────────────────────────────

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (int, float)):
            return default if (math.isnan(float(value)) or math.isinf(float(value))) else float(value)
        if isinstance(value, str):
            return float(value.replace(",", ""))
    except Exception:
        pass
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, Decimal):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            return int(float(value.replace(",", "")))
    except Exception:
        pass
    return default


def _clean_search_terms(search_terms: List[Dict]) -> List[Dict]:
    return [
        {
            "value": str(t.get("value", "")),
            "operator": str(t.get("operator", "AND")),
            "exact_match": bool(t.get("exact_match", False)),
        }
        for t in search_terms
        if isinstance(t, dict) and t.get("value")
    ]


def _calculate_quality(news_results: List[Dict]) -> float:
    if not news_results:
        return 0.0
    with_date = sum(1 for r in news_results if r.get("date"))
    return with_date / len(news_results)


def _make_news_sample(news_results: List[Dict]) -> List[Dict]:
    return [
        {
            "title": str(r.get("title", ""))[:100],
            "date": str(r.get("date", "")),
            "source": str(r.get("source", ""))[:50],
        }
        for r in news_results
        if isinstance(r, dict)
    ]
