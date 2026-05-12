"""
Hype Cycle routes.
POST /api/hype-cycle/analyze        → launch task (Phase 2: Celery), return task_id
GET  /api/hype-cycle/history        → list saved queries from DynamoDB
GET  /api/hype-cycle/queries/{id}   → single query detail
"""

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ── DynamoDB dependency ───────────────────────────────────────

def get_hc_storage():
    """Dependency that returns a HypeCycleStorage instance."""
    from storage.dynamodb import get_db
    from storage.hype_cycle import HypeCycleStorage
    return HypeCycleStorage(get_db())


# ── Request / Response models ─────────────────────────────────

class SearchTermModel(BaseModel):
    value: str
    operator: str = "AND"
    exact_match: bool = True


class SearchParameters(BaseModel):
    min_year: int = 2015
    sources_filter: list[str] = ["Tech News", "Business News"]
    max_results: int = 200


class AnalyzeRequest(BaseModel):
    search_terms: list[SearchTermModel]
    technology_name: Optional[str] = None
    technology_description: Optional[str] = None
    category_id: str = "default"
    search_parameters: SearchParameters = SearchParameters()
    notes: Optional[str] = None
    auto_save: bool = True


class AnalyzeResponse(BaseModel):
    task_id: str
    status: str
    message: str


# ── Routes ────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
async def start_analysis(payload: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Launch async Hype Cycle analysis via BackgroundTasks and return task_id immediately.
    """
    filled = [t for t in payload.search_terms if t.value.strip()]
    if not filled:
        raise HTTPException(status_code=400, detail="At least one search term is required")

    task_id = f"hc_task_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    _task_store[task_id] = {
        "status": "queued",
        "progress": 0,
        "message": "En cola...",
    }

    from workers.hype_cycle_worker import run_hype_cycle_analysis
    background_tasks.add_task(run_hype_cycle_analysis, task_id, payload.model_dump())

    return AnalyzeResponse(
        task_id=task_id,
        status="queued",
        message="Análisis iniciado. Conecta con el stream para ver el progreso.",
    )


@router.get("/history")
async def get_history(
    category_id: Optional[str] = Query(None, description="Filter by category"),
    storage=Depends(get_hc_storage),
):
    """Return saved HC queries from DynamoDB, optionally filtered by category."""
    try:
        return storage.get_queries(category_id=category_id)
    except Exception as e:
        logger.error("Error fetching HC history: %s", e)
        raise HTTPException(status_code=500, detail="Error fetching history from DynamoDB")


@router.get("/queries/{query_id}")
async def get_query(
    query_id: str,
    storage=Depends(get_hc_storage),
):
    """Return a single HC query by ID."""
    try:
        item = storage.get_query_by_id(query_id)
        if not item:
            raise HTTPException(status_code=404, detail="Query not found")
        return item
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching query %s: %s", query_id, e)
        raise HTTPException(status_code=500, detail="Error fetching query")


# ── In-memory task store (Phase 1 placeholder) ────────────────
# Replace with Redis in Phase 2
_task_store: dict = {}


def get_task(task_id: str) -> dict:
    return _task_store.get(task_id, {})


def update_task(task_id: str, **kwargs):
    if task_id in _task_store:
        _task_store[task_id].update(kwargs)
