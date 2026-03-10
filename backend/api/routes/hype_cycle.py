"""
Hype Cycle routes.
POST /api/hype-cycle/analyze  → launch Celery task, return task_id
GET  /api/hype-cycle/history  → list saved queries from DynamoDB
GET  /api/hype-cycle/queries/{query_id} → single query detail
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import uuid, time

router = APIRouter()

# ── Request / Response models ────────────────────────────────

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

# ── Routes ───────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
async def start_analysis(payload: AnalyzeRequest):
    """Launch async Hype Cycle analysis and return task_id immediately."""
    # Validate at least one non-empty term
    filled = [t for t in payload.search_terms if t.value.strip()]
    if not filled:
        raise HTTPException(status_code=400, detail="At least one search term is required")

    task_id = f"hc_task_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    # TODO (Phase 2): replace with Celery task dispatch
    # from tasks.hype_cycle_task import run_hype_cycle_analysis
    # run_hype_cycle_analysis.delay(task_id, payload.model_dump())

    # For Phase 1 (scaffolding): store pending state in Redis / in-memory
    _task_store[task_id] = {
        "status": "queued",
        "progress": 0,
        "message": "En cola...",
        "payload": payload.model_dump(),
    }

    return AnalyzeResponse(
        task_id=task_id,
        status="queued",
        message="Análisis en cola. Conecta con el stream para ver el progreso.",
    )


@router.get("/history")
async def get_history(category_id: Optional[str] = Query(None)):
    """Return saved HC queries from DynamoDB."""
    # TODO: wire to storage.dynamodb
    # from storage.hype_cycle import HypeCycleStorage
    # storage = HypeCycleStorage()
    # return storage.get_queries(category_id=category_id)
    return []  # Placeholder until storage layer is connected


@router.get("/queries/{query_id}")
async def get_query(query_id: str):
    """Return a single HC query by ID."""
    # TODO: wire to storage.dynamodb
    raise HTTPException(status_code=404, detail="Query not found")


# ── Simple in-memory task store (Phase 1 placeholder) ────────
# Replace with Redis in Phase 2
_task_store: dict = {}


def get_task(task_id: str) -> dict:
    return _task_store.get(task_id, {})


def update_task(task_id: str, **kwargs):
    if task_id in _task_store:
        _task_store[task_id].update(kwargs)
