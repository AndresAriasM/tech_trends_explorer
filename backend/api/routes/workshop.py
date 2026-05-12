"""
backend/api/routes/workshop.py

Endpoints del taller prospectivo LACCEI 2032.

POST   /api/workshop/sessions                       → crear sesión
GET    /api/workshop/sessions                       → listar sesiones
GET    /api/workshop/sessions/{session_id}          → detalle sesión
PATCH  /api/workshop/sessions/{session_id}/close    → cerrar sesión
PATCH  /api/workshop/sessions/{session_id}/gforms   → guardar URL Google Form
GET    /api/workshop/join/{code}                    → info sesión por código
GET    /api/workshop/variables                      → catálogo de variables
POST   /api/workshop/{session_id}/import            → importar votos desde Google Forms
GET    /api/workshop/{session_id}/votes             → todos los votos de una sesión
DELETE /api/workshop/{session_id}/votes/human       → limpiar votos humanos (re-import)
GET    /api/workshop/{session_id}/synthesis         → síntesis estadística
POST   /api/workshop/{session_id}/ai-vote           → disparar votación AI (BackgroundTask)
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List

from core.workshop import WorkshopService, WORKSHOP_VARIABLES
from core.ai_voter import get_ai_voter
from storage.workshop import WorkshopStorage, get_workshop_db

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Dependency ─────────────────────────────────────────────────────────────────

def get_service(db: WorkshopStorage = Depends(get_workshop_db)) -> WorkshopService:
    return WorkshopService(db)


# ── Schemas ────────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    title:       str = Field(..., min_length=3, max_length=200)
    description: str = Field("", max_length=500)
    gforms_url:  str = Field("", max_length=1000)


class GFormsUrlRequest(BaseModel):
    gforms_url: str = Field(..., max_length=1000)


class ImportVoteItem(BaseModel):
    variable_id:    int = Field(..., ge=1, le=20)
    rating_current: int = Field(..., ge=1, le=4)
    rating_2032:    int = Field(..., ge=1, le=4)
    justification:  str = Field("", max_length=600)


class ImportExpertVotes(BaseModel):
    expert_name: str = Field(..., min_length=2, max_length=100)
    votes:       List[ImportVoteItem]


class ImportRequest(BaseModel):
    """
    Payload para importar resultados de Google Forms.
    Cada elemento representa las respuestas de un experto.
    """
    experts: List[ImportExpertVotes]
    clear_existing: bool = False   # si True, borra votos humanos previos antes de importar


# ── Sesiones ───────────────────────────────────────────────────────────────────

@router.post("/sessions", status_code=201)
def create_session(
    body:    CreateSessionRequest,
    service: WorkshopService = Depends(get_service),
):
    try:
        session = service.create_session(body.title, body.description, body.gforms_url)
        return session
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Error creando sesión: %s", e)
        raise HTTPException(500, "Error interno al crear la sesión")


@router.get("/sessions")
def list_sessions(service: WorkshopService = Depends(get_service)):
    return service.list_sessions()


@router.get("/sessions/{session_id}")
def get_session(session_id: str, service: WorkshopService = Depends(get_service)):
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(404, "Sesión no encontrada")
    return session


@router.patch("/sessions/{session_id}/close")
def close_session(session_id: str, service: WorkshopService = Depends(get_service)):
    ok = service.close_session(session_id)
    if not ok:
        raise HTTPException(404, "Sesión no encontrada o ya cerrada")
    return {"closed": True, "session_id": session_id}


@router.patch("/sessions/{session_id}/gforms")
def update_gforms_url(
    session_id: str,
    body: GFormsUrlRequest,
    db: WorkshopStorage = Depends(get_workshop_db),
):
    ok = db.update_gforms_url(session_id, body.gforms_url)
    if not ok:
        raise HTTPException(500, "Error al actualizar la URL")
    return {"updated": True, "gforms_url": body.gforms_url}


# ── Join por código ────────────────────────────────────────────────────────────

@router.get("/join/{code}")
def join_by_code(code: str, service: WorkshopService = Depends(get_service)):
    session = service.get_session_by_code(code)
    if not session:
        raise HTTPException(404, "Código de sala no válido")
    return {
        "session_id":     session["session_id"],
        "code":           session["code"],
        "title":          session["title"],
        "description":    session.get("description", ""),
        "gforms_url":     session.get("gforms_url", ""),
        "status":         session["status"],
        "ai_votes_ready": session.get("ai_votes_ready", False),
        "created_at":     session.get("created_at", ""),
    }


# ── Variables ──────────────────────────────────────────────────────────────────

@router.get("/variables")
def get_variables(service: WorkshopService = Depends(get_service)):
    return service.get_variables()


# ── Importación desde Google Forms ─────────────────────────────────────────────

@router.post("/{session_id}/import", status_code=201)
def import_gforms(
    session_id: str,
    body:       ImportRequest,
    service:    WorkshopService = Depends(get_service),
    db:         WorkshopStorage = Depends(get_workshop_db),
):
    """
    Importa los resultados recolectados por Google Forms.
    El facilitador convierte el CSV de respuestas al formato JSON de este endpoint.
    """
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(404, "Sesión no encontrada")

    if body.clear_existing:
        # Eliminar todos los votos humanos existentes en esta sesión
        votes = db.get_votes_for_session(session_id)
        human_votes = [v for v in votes if not v.get("is_ai_vote")]
        for v in human_votes:
            db._votes.delete_item(
                Key={"vote_id": v["vote_id"], "session_id": session_id}
            )
        logger.info("Votos humanos eliminados para sesión %s (%d)", session_id, len(human_votes))

    saved = 0
    errors = []
    for expert in body.experts:
        try:
            result = service.register_expert_votes(
                session_id=session_id,
                expert_name=expert.expert_name,
                votes=[v.model_dump() for v in expert.votes],
                allow_revote=body.clear_existing,
            )
            saved += result["votes_saved"]
        except ValueError as e:
            errors.append({"expert": expert.expert_name, "error": str(e)})
        except Exception as e:
            errors.append({"expert": expert.expert_name, "error": str(e)})
            logger.error("Error importando votos de %s: %s", expert.expert_name, e)

    return {
        "experts_imported": len(body.experts) - len(errors),
        "votes_saved":      saved,
        "errors":           errors,
    }


@router.delete("/{session_id}/votes/human")
def clear_human_votes(
    session_id: str,
    db: WorkshopStorage = Depends(get_workshop_db),
    service: WorkshopService = Depends(get_service),
):
    """Elimina todos los votos humanos de una sesión (para re-importar)."""
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(404, "Sesión no encontrada")
    votes = db.get_votes_for_session(session_id)
    human_votes = [v for v in votes if not v.get("is_ai_vote")]
    for v in human_votes:
        db._votes.delete_item(
            Key={"vote_id": v["vote_id"], "session_id": session_id}
        )
    return {"deleted": len(human_votes)}


# ── Votos ──────────────────────────────────────────────────────────────────────

@router.get("/{session_id}/votes")
def get_votes(session_id: str, db: WorkshopStorage = Depends(get_workshop_db)):
    return db.get_votes_for_session(session_id)


# ── Síntesis ───────────────────────────────────────────────────────────────────

@router.get("/{session_id}/synthesis")
def get_synthesis(
    session_id: str,
    service:    WorkshopService = Depends(get_service),
):
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(404, "Sesión no encontrada")
    return service.compute_synthesis(session_id)


# ── Voto AI ────────────────────────────────────────────────────────────────────

@router.post("/{session_id}/ai-vote", status_code=202)
def trigger_ai_vote(
    session_id:       str,
    background_tasks: BackgroundTasks,
    service:          WorkshopService = Depends(get_service),
    db:               WorkshopStorage = Depends(get_workshop_db),
):
    """Dispara la generación de votos AI en background. Retorna 202 inmediatamente."""
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(404, "Sesión no encontrada")
    if session.get("ai_votes_ready"):
        raise HTTPException(409, "Los votos AI ya fueron generados para esta sesión")

    try:
        voter = get_ai_voter()
    except (ImportError, ValueError) as e:
        raise HTTPException(503, f"AI Voter no disponible: {e}")

    background_tasks.add_task(
        _run_ai_vote,
        voter=voter,
        session_id=session_id,
        variables=WORKSHOP_VARIABLES,
        db=db,
    )

    return {
        "accepted":   True,
        "session_id": session_id,
        "message":    f"Generando votos AI para {len(WORKSHOP_VARIABLES)} variables en background",
    }


async def _run_ai_vote(voter, session_id: str, variables: list, db) -> None:
    try:
        result = await voter.vote_all_variables(session_id, variables, db)
        logger.info("AI vote completado: %s", result)
    except Exception as e:
        logger.error("Error en AI vote background task: %s", e)
