"""
backend/api/routes/delphi.py

Endpoints del módulo Delphi. Implementa el flujo de la §8 de
`docs/LOGICA_DELPHI.md`:

    1. Abrir      → POST /rounds/{round_id}/open
    2. Recoger    → POST /invite/{token}/responses
    3. Cerrar     → POST /rounds/{round_id}/close   (el único gesto manual)
    4. Analizar   → GET  /studies/{study_id}/stats  (se dispara solo al cerrar)
    5. Devolver   → GET  /invite/{token}            (paquete anónimo de §5.1)
    6. Decidir    → GET  /studies/{study_id}/map    (variables sin cerrar)

Mapa completo:

  Estudios
    POST   /studies                              crear
    GET    /studies                              listar con contadores
    GET    /studies/{sid}                        detalle
    PATCH  /studies/{sid}                        configurar (umbral, parada…)
    DELETE /studies/{sid}

  Variables del estudio (§8: el ítem Delphi ES una Variable)
    GET    /studies/{sid}/variables
    POST   /studies/{sid}/variables              alta en lote
    PATCH  /variables/{vid}
    DELETE /variables/{vid}

  Rondas (incluidas las parciales de §7)
    GET    /studies/{sid}/rounds
    POST   /studies/{sid}/rounds
    POST   /rounds/{rid}/open
    POST   /rounds/{rid}/close

  Panel
    GET    /studies/{sid}/participants
    POST   /studies/{sid}/participants           invitar → token
    PATCH  /participants/{pid}                   Kc/Ka, perfil, estado

  Panelista (por token de invitación — identidad estable, nunca por nombre)
    GET    /invite/{token}                       formulario + devolución §5.1
    GET    /invite/{token}/feedback              sólo la devolución
    POST   /invite/{token}/responses             enviar respuestas

  Análisis (§9)
    GET    /studies/{sid}/stats                  informe completo
    GET    /studies/{sid}/rounds/{n}/stats       una ronda
    GET    /studies/{sid}/map                    los 4 cuadrantes de §7
    GET    /schema-check                         ¿está aplicada la migración 001?

**El router NO se registra en main.py** a propósito: la migración 001 aún no
está aplicada y estos endpoints fallarían contra el esquema actual.
`GET /schema-check` dice exactamente qué falta.

Nota de arquitectura: el acceso a datos vive aquí, en `DelphiRepo`, en vez de
en `storage/`, porque el módulo se entrega como una unidad cerrada. El cálculo
está en `core/delphi.py` y no sabe nada de SQL: es puro y se prueba solo.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from core.delphi import (
    Response as DelphiResponse,
    StudyConfig,
    compute_study_report,
    feedback_package,
    spearman_matrix,
    variable_round_stats,
)

logger = logging.getLogger(__name__)
router = APIRouter()

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Json
    _HAS_PSYCOPG = True
except ImportError:  # pragma: no cover
    _HAS_PSYCOPG = False


# ══════════════════════════════════════════════════════════════════════════
# Conexión
# ══════════════════════════════════════════════════════════════════════════

# Columnas y tablas que aporta migrations/001_delphi_prospecta.sql. Si faltan,
# todo este router devuelve 503 con instrucciones en vez de un error de SQL.
REQUIRED_SCHEMA = {
    "delphi_variables": ["id", "study_id", "code", "name", "family", "order_index"],
    "delphi_studies": ["horizon_year", "scale_type", "dichotomy_threshold",
                       "max_rounds", "stopping_rule"],
    "delphi_questions": ["variable_id"],
    "delphi_answers": ["is_abstention", "argument", "answered_at"],
    "delphi_participants": ["kc", "ka", "k", "education", "affiliation", "display_name"],
}


@contextmanager
def _connection() -> Iterator[Any]:
    if not _HAS_PSYCOPG:
        raise HTTPException(503, "psycopg no está instalado en el backend")
    dsn = (os.getenv("DATABASE_URL") or "").strip()
    if not dsn:
        raise HTTPException(503, "DATABASE_URL no configurada (ver backend/.env.example)")
    try:
        with psycopg.connect(dsn, row_factory=dict_row, connect_timeout=15) as conn:
            yield conn
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Delphi: fallo de conexión: %s", type(exc).__name__)
        raise HTTPException(503, "No se pudo conectar a la base de datos")


class DelphiRepo:
    """Acceso a datos. Una conexión por petición; sin ORM, el esquema es fijo."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    # ── utilidades ───────────────────────────────────────────────────────

    def q(self, sql: str, params: tuple = ()) -> List[Dict]:
        # `params or None`: con una tupla vacía psycopg entra en modo
        # placeholder y un `like 'delphi%'` del SQL revienta como si el `%`
        # fuese un marcador.
        with self.conn.cursor() as cur:
            cur.execute(sql, params or None)
            return cur.fetchall() if cur.description else []

    def q1(self, sql: str, params: tuple = ()) -> Optional[Dict]:
        rows = self.q(sql, params)
        return rows[0] if rows else None

    def missing_schema(self) -> Dict[str, List[str]]:
        rows = self.q(
            "select table_name, column_name from information_schema.columns "
            "where table_schema = 'public' and table_name like 'delphi%'"
        )
        present: Dict[str, set] = {}
        for r in rows:
            present.setdefault(r["table_name"], set()).add(r["column_name"])
        missing: Dict[str, List[str]] = {}
        for table, cols in REQUIRED_SCHEMA.items():
            if table not in present:
                missing[table] = ["<tabla entera>"]
                continue
            gone = [c for c in cols if c not in present[table]]
            if gone:
                missing[table] = gone
        return missing

    def require_schema(self) -> None:
        missing = self.missing_schema()
        if missing:
            raise HTTPException(
                503,
                "Falta aplicar backend/migrations/001_delphi_prospecta.sql. "
                f"Ausente: {missing}",
            )

    # ── estudios ─────────────────────────────────────────────────────────

    def resolve_owner(self, owner_id: Optional[str]) -> str:
        if owner_id:
            return owner_id
        env_owner = (os.getenv("DELPHI_DEFAULT_OWNER_ID") or "").strip()
        if env_owner:
            return env_owner
        rows = self.q("select id from profiles limit 2")
        if len(rows) == 1:
            return str(rows[0]["id"])
        raise HTTPException(
            400,
            "Falta owner_id: hay 0 o varios perfiles. Pásalo en el cuerpo "
            "o define DELPHI_DEFAULT_OWNER_ID.",
        )

    def list_studies(self) -> List[Dict]:
        return self.q("""
            select s.*,
                   (select count(*) from delphi_rounds     r where r.study_id = s.id) as n_rounds,
                   (select count(*) from delphi_variables  v where v.study_id = s.id) as n_variables,
                   (select count(*) from delphi_participants p
                      where p.study_id = s.id)                                        as n_invited,
                   (select count(*) from delphi_participants p
                      where p.study_id = s.id and p.status <> 'invited')              as n_registered,
                   (select count(*) from delphi_questions q
                      join delphi_rounds r2 on r2.id = q.round_id
                     where r2.study_id = s.id)                                        as n_questions
              from delphi_studies s
             order by s.updated_at desc nulls last, s.created_at desc
        """)

    def get_study(self, sid: str) -> Dict:
        row = self.q1("select * from delphi_studies where id = %s", (sid,))
        if not row:
            raise HTTPException(404, "Estudio no encontrado")
        return row

    def study_config(self, study: Dict) -> StudyConfig:
        """Traduce la fila del estudio a la configuración del cálculo."""
        rule = study.get("stopping_rule") or {}
        scale = (study.get("scale_type") or "likert_5").lower()
        if scale == "scale_10":
            lo, hi = 1.0, 10.0
        elif scale in ("binary", "binaria"):
            lo, hi = 0.0, 1.0
        else:
            lo, hi = 1.0, 5.0
        return StudyConfig(
            dichotomy_threshold=float(study.get("dichotomy_threshold") or 4),
            scale_min=lo,
            scale_max=hi,
            stability_min=float(rule.get("stability_min", 0.85)),
            iqr_max=float(rule.get("iqr_max", 1.0)),
            consensus_min=float(rule.get("consensus_min", 0.70)),
            k_min=(float(rule["k_min"]) if rule.get("k_min") is not None else None),
        )

    # ── respuestas materializadas para el cálculo ────────────────────────

    def load_responses(self, sid: str) -> List[DelphiResponse]:
        """
        Una fila por (panelista, variable, ronda) entregada. Sólo entra lo
        enviado: un borrador no es un juicio.
        """
        rows = self.q("""
            select r.round_number,
                   q.variable_id,
                   s.participant_id,
                   p.k,
                   a.value,
                   a.is_abstention,
                   a.argument
              from delphi_answers a
              join delphi_submissions  s on s.id = a.submission_id
              join delphi_questions    q on q.id = a.question_id
              join delphi_rounds       r on r.id = s.round_id
              join delphi_participants p on p.id = s.participant_id
             where r.study_id = %s
               and q.variable_id is not null
               and s.status = 'submitted'
        """, (sid,))

        out: List[DelphiResponse] = []
        for r in rows:
            out.append(DelphiResponse(
                participant_id=str(r["participant_id"]),
                variable_id=str(r["variable_id"]),
                round_number=int(r["round_number"]),
                value=_as_number(r["value"]),
                is_abstention=bool(r["is_abstention"]),
                k=(float(r["k"]) if r["k"] is not None else None),
                argument=r["argument"],
            ))
        return out

    def variable_lookup(self, sid: str) -> tuple[Dict[str, str], Dict[str, Optional[str]]]:
        rows = self.q(
            "select id, name, family from delphi_variables where study_id = %s "
            "order by order_index, name", (sid,)
        )
        names = {str(r["id"]): r["name"] for r in rows}
        families = {str(r["id"]): r["family"] for r in rows}
        return names, families

    def panelists_k(self, sid: str) -> Dict[str, Optional[float]]:
        rows = self.q("select id, k from delphi_participants where study_id = %s", (sid,))
        return {str(r["id"]): (float(r["k"]) if r["k"] is not None else None) for r in rows}

    def round_population(self, sid: str) -> Dict[int, int]:
        """
        Denominador de la tasa de respuesta por ronda: a cuántos panelistas
        activos se les abrió esa ronda. Si nadie tiene submission, cae al
        total de panelistas no dados de baja.
        """
        rows = self.q("""
            select r.round_number,
                   (select count(*) from delphi_participants p
                     where p.study_id = r.study_id and p.status <> 'dropped') as n_active
              from delphi_rounds r where r.study_id = %s
        """, (sid,))
        return {int(r["round_number"]): int(r["n_active"]) for r in rows}


def _as_number(value: Any) -> Optional[float]:
    """`delphi_answers.value` es jsonb: un número, o algo que no lo es."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", "."))
        except ValueError:
            return None
    return None


def get_repo() -> Iterator[DelphiRepo]:
    with _connection() as conn:
        yield DelphiRepo(conn)


def get_checked_repo(repo: DelphiRepo = Depends(get_repo)) -> DelphiRepo:
    repo.require_schema()
    return repo


# ══════════════════════════════════════════════════════════════════════════
# Esquemas
# ══════════════════════════════════════════════════════════════════════════

class StudyCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field("", max_length=2000)
    owner_id: Optional[str] = None
    horizon_year: Optional[int] = Field(None, ge=2000, le=2100)
    scale_type: str = Field("likert_5", max_length=32)
    dichotomy_threshold: int = Field(4, ge=0, le=10)
    max_rounds: int = Field(3, ge=1, le=10)
    stopping_rule: Dict[str, float] = Field(
        default_factory=lambda: {"stability_min": 0.85, "iqr_max": 1.0}
    )


class StudyUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=3, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    status: Optional[str] = Field(None, pattern="^(draft|active|closed)$")
    horizon_year: Optional[int] = Field(None, ge=2000, le=2100)
    scale_type: Optional[str] = Field(None, max_length=32)
    dichotomy_threshold: Optional[int] = Field(None, ge=0, le=10)
    max_rounds: Optional[int] = Field(None, ge=1, le=10)
    stopping_rule: Optional[Dict[str, float]] = None


class VariableIn(BaseModel):
    code: str = Field(..., min_length=1, max_length=120)
    name: str = Field(..., min_length=1, max_length=300)
    description: str = Field("", max_length=2000)
    family: Optional[str] = Field(None, max_length=160)
    order_index: int = 0


class VariableUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=300)
    description: Optional[str] = None
    family: Optional[str] = None
    order_index: Optional[int] = None


class RoundCreate(BaseModel):
    round_number: Optional[int] = Field(None, ge=1)
    title: str = Field("", max_length=300)
    instructions: str = Field("", max_length=4000)
    # Ronda parcial (§7): si va vacío, entran todas las variables del estudio.
    variable_ids: Optional[List[str]] = None
    question_type: str = Field("likert_5", max_length=32)


class ParticipantIn(BaseModel):
    email: str = Field(..., min_length=5, max_length=200)
    display_name: Optional[str] = Field(None, max_length=200)
    kc: Optional[float] = Field(None, ge=0, le=1)
    ka: Optional[float] = Field(None, ge=0, le=1)
    education: Optional[str] = Field(None, max_length=200)
    affiliation: Optional[str] = Field(None, max_length=300)


class ParticipantUpdate(BaseModel):
    display_name: Optional[str] = None
    kc: Optional[float] = Field(None, ge=0, le=1)
    ka: Optional[float] = Field(None, ge=0, le=1)
    education: Optional[str] = None
    affiliation: Optional[str] = None
    status: Optional[str] = Field(None, pattern="^(invited|registered|active|dropped)$")


class AnswerIn(BaseModel):
    variable_id: str
    value: Optional[float] = None
    is_abstention: bool = False
    argument: Optional[str] = Field(None, max_length=4000)


class SubmissionIn(BaseModel):
    answers: List[AnswerIn]
    # False guarda un borrador; sólo lo enviado entra en el análisis.
    submit: bool = True


# ══════════════════════════════════════════════════════════════════════════
# Diagnóstico
# ══════════════════════════════════════════════════════════════════════════

@router.get("/schema-check", tags=["Delphi"])
def schema_check(repo: DelphiRepo = Depends(get_repo)):
    """¿Está aplicada la migración 001? El router entero depende de ella."""
    missing = repo.missing_schema()
    return {
        "ready": not missing,
        "missing": missing,
        "migration": "backend/migrations/001_delphi_prospecta.sql",
    }


# ══════════════════════════════════════════════════════════════════════════
# Estudios
# ══════════════════════════════════════════════════════════════════════════

@router.get("/studies")
def list_studies(repo: DelphiRepo = Depends(get_checked_repo)):
    return repo.list_studies()


@router.post("/studies", status_code=201)
def create_study(body: StudyCreate, repo: DelphiRepo = Depends(get_checked_repo)):
    owner = repo.resolve_owner(body.owner_id)
    row = repo.q1("""
        insert into delphi_studies
            (owner_id, title, description, horizon_year, scale_type,
             dichotomy_threshold, max_rounds, stopping_rule)
        values (%s, %s, %s, %s, %s, %s, %s, %s)
        returning *
    """, (owner, body.title, body.description, body.horizon_year, body.scale_type,
          body.dichotomy_threshold, body.max_rounds, Json(body.stopping_rule)))
    repo.conn.commit()
    return row


@router.get("/studies/{study_id}")
def get_study(study_id: UUID, repo: DelphiRepo = Depends(get_checked_repo)):
    sid = str(study_id)
    study = repo.get_study(sid)
    study["variables"] = repo.q(
        "select * from delphi_variables where study_id = %s order by order_index, name", (sid,))
    study["rounds"] = repo.q("""
        select r.*,
               (select count(*) from delphi_questions q where q.round_id = r.id) as n_questions,
               (select count(*) from delphi_submissions s
                 where s.round_id = r.id and s.status = 'submitted')             as n_submitted
          from delphi_rounds r where r.study_id = %s order by r.round_number
    """, (sid,))
    study["participants_summary"] = repo.q1("""
        select count(*)                                    as n_invited,
               count(*) filter (where status <> 'invited') as n_registered,
               count(*) filter (where status = 'dropped')  as n_dropped,
               round(avg(k), 3)                            as mean_k
          from delphi_participants where study_id = %s
    """, (sid,))
    return study


@router.patch("/studies/{study_id}")
def update_study(study_id: UUID, body: StudyUpdate,
                 repo: DelphiRepo = Depends(get_checked_repo)):
    sid = str(study_id)
    repo.get_study(sid)
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(400, "Nada que actualizar")
    sets, params = [], []
    for key, value in fields.items():
        sets.append(f"{key} = %s")
        params.append(Json(value) if key == "stopping_rule" else value)
    sets.append("updated_at = now()")
    params.append(sid)
    row = repo.q1(f"update delphi_studies set {', '.join(sets)} where id = %s returning *",
                  tuple(params))
    repo.conn.commit()
    return row


@router.delete("/studies/{study_id}", status_code=204)
def delete_study(study_id: UUID, repo: DelphiRepo = Depends(get_checked_repo)):
    repo.q("delete from delphi_studies where id = %s", (str(study_id),))
    repo.conn.commit()


# ══════════════════════════════════════════════════════════════════════════
# Variables del estudio
# ══════════════════════════════════════════════════════════════════════════

@router.get("/studies/{study_id}/variables")
def list_variables(study_id: UUID, repo: DelphiRepo = Depends(get_checked_repo)):
    return repo.q(
        "select * from delphi_variables where study_id = %s order by order_index, name",
        (str(study_id),))


@router.post("/studies/{study_id}/variables", status_code=201)
def create_variables(study_id: UUID,
                     variables: List[VariableIn] = Body(...),
                     repo: DelphiRepo = Depends(get_checked_repo)):
    sid = str(study_id)
    repo.get_study(sid)
    out = []
    for i, v in enumerate(variables):
        row = repo.q1("""
            insert into delphi_variables (study_id, code, name, description, family, order_index)
            values (%s, %s, %s, %s, %s, %s)
            on conflict (study_id, code) do update
              set name = excluded.name,
                  description = excluded.description,
                  family = excluded.family,
                  order_index = excluded.order_index
            returning *
        """, (sid, v.code, v.name, v.description, v.family, v.order_index or i))
        out.append(row)
    repo.conn.commit()
    return out


@router.patch("/variables/{variable_id}")
def update_variable(variable_id: UUID, body: VariableUpdate,
                    repo: DelphiRepo = Depends(get_checked_repo)):
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(400, "Nada que actualizar")
    sets = ", ".join(f"{k} = %s" for k in fields)
    row = repo.q1(f"update delphi_variables set {sets} where id = %s returning *",
                  tuple([*fields.values(), str(variable_id)]))
    if not row:
        raise HTTPException(404, "Variable no encontrada")
    repo.conn.commit()
    return row


@router.delete("/variables/{variable_id}", status_code=204)
def delete_variable(variable_id: UUID, repo: DelphiRepo = Depends(get_checked_repo)):
    repo.q("delete from delphi_variables where id = %s", (str(variable_id),))
    repo.conn.commit()


# ══════════════════════════════════════════════════════════════════════════
# Rondas
# ══════════════════════════════════════════════════════════════════════════

@router.get("/studies/{study_id}/rounds")
def list_rounds(study_id: UUID, repo: DelphiRepo = Depends(get_checked_repo)):
    return repo.q("""
        select r.*,
               (select count(*) from delphi_questions q where q.round_id = r.id) as n_questions,
               (select count(*) from delphi_submissions s
                 where s.round_id = r.id and s.status = 'submitted')             as n_submitted
          from delphi_rounds r where r.study_id = %s order by r.round_number
    """, (str(study_id),))


@router.post("/studies/{study_id}/rounds", status_code=201)
def create_round(study_id: UUID, body: RoundCreate,
                 repo: DelphiRepo = Depends(get_checked_repo)):
    """
    Crea la ronda y genera sus preguntas desde `variable_ids` (§8, paso 1).
    Si no se pasan, entran todas las variables del estudio; si se pasan, es
    una **ronda parcial** sobre las variables que no cerraron (§7).
    """
    sid = str(study_id)
    study = repo.get_study(sid)

    n = body.round_number
    if n is None:
        last = repo.q1(
            "select coalesce(max(round_number), 0) as n from delphi_rounds where study_id = %s",
            (sid,))
        n = int(last["n"]) + 1
    if n > int(study.get("max_rounds") or 10):
        raise HTTPException(400, f"El estudio declara un máximo de {study['max_rounds']} rondas")

    if body.variable_ids:
        variables = repo.q(
            "select * from delphi_variables where study_id = %s and id = any(%s) "
            "order by order_index, name", (sid, [str(v) for v in body.variable_ids]))
    else:
        variables = repo.q(
            "select * from delphi_variables where study_id = %s order by order_index, name",
            (sid,))
    if not variables:
        raise HTTPException(400, "El estudio no tiene variables: una ronda vacía no es una ronda")

    rnd = repo.q1("""
        insert into delphi_rounds (study_id, round_number, title, instructions, status)
        values (%s, %s, %s, %s, 'draft') returning *
    """, (sid, n, body.title or f"Ronda {n}", body.instructions))

    for i, v in enumerate(variables):
        repo.q("""
            insert into delphi_questions
                (round_id, order_index, type, title, description, variable_id)
            values (%s, %s, %s::question_type, %s, %s, %s)
        """, (rnd["id"], i, body.question_type, v["name"], v["description"] or "", v["id"]))

    repo.conn.commit()
    rnd["n_questions"] = len(variables)
    rnd["partial"] = bool(body.variable_ids)
    return rnd


@router.post("/rounds/{round_id}/open")
def open_round(round_id: UUID, repo: DelphiRepo = Depends(get_checked_repo)):
    row = repo.q1("""
        update delphi_rounds set status = 'open', opens_at = coalesce(opens_at, now())
         where id = %s and status = 'draft' returning *
    """, (str(round_id),))
    if not row:
        raise HTTPException(409, "La ronda no existe o no está en borrador")
    repo.conn.commit()
    return row


@router.post("/rounds/{round_id}/close")
def close_round(round_id: UUID, repo: DelphiRepo = Depends(get_checked_repo)):
    """
    Paso 3 del flujo (§8): «cerrar la ronda es el único gesto manual; el
    informe y el paquete de devolución se generan solos». Aquí se cierra y se
    devuelve el informe ya calculado.
    """
    rid = str(round_id)
    row = repo.q1("""
        update delphi_rounds set status = 'closed', closes_at = now()
         where id = %s and status = 'open' returning *
    """, (rid,))
    if not row:
        raise HTTPException(409, "La ronda no existe o no está abierta")
    # Un borrador nunca entregado no es un juicio, pero tampoco se pierde:
    # se queda en 'draft' y queda fuera del análisis por el filtro de estado.
    repo.conn.commit()
    report = _report(repo, str(row["study_id"]))
    return {"round": row, "report": report}


# ══════════════════════════════════════════════════════════════════════════
# Panel
# ══════════════════════════════════════════════════════════════════════════

@router.get("/studies/{study_id}/participants")
def list_participants(study_id: UUID, include_token: bool = Query(False),
                      repo: DelphiRepo = Depends(get_checked_repo)):
    """
    El token es la credencial del panelista: no viaja salvo que se pida
    explícitamente (para componer el correo de invitación).
    """
    cols = ("id, study_id, email, display_name, status, kc, ka, k, education, "
            "affiliation, invited_at, registered_at")
    if include_token:
        cols += ", invitation_token"
    return repo.q(
        f"select {cols} from delphi_participants where study_id = %s order by invited_at",
        (str(study_id),))


@router.post("/studies/{study_id}/participants", status_code=201)
def invite_participants(study_id: UUID,
                        participants: List[ParticipantIn] = Body(...),
                        repo: DelphiRepo = Depends(get_checked_repo)):
    """
    Invitación por token (§4): identidad estable desde el primer día. El
    cruce por nombre tecleado del ejercicio original perdía panelistas por un
    espacio final, y cada pérdida se leía como deserción.
    """
    sid = str(study_id)
    repo.get_study(sid)
    out = []
    for p in participants:
        row = repo.q1("""
            insert into delphi_participants
                (study_id, email, display_name, kc, ka, education, affiliation)
            values (%s, %s, %s, %s, %s, %s, %s)
            returning id, email, display_name, status, kc, ka, k,
                      education, affiliation, invitation_token, invited_at
        """, (sid, p.email.strip().lower(), p.display_name, p.kc, p.ka,
              p.education, p.affiliation))
        out.append(row)
    repo.conn.commit()
    return out


@router.patch("/participants/{participant_id}")
def update_participant(participant_id: UUID, body: ParticipantUpdate,
                       repo: DelphiRepo = Depends(get_checked_repo)):
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(400, "Nada que actualizar")
    sets = ", ".join(f"{k} = %s" for k in fields)
    row = repo.q1(
        f"update delphi_participants set {sets} where id = %s "
        "returning id, email, display_name, status, kc, ka, k, education, affiliation",
        tuple([*fields.values(), str(participant_id)]))
    if not row:
        raise HTTPException(404, "Panelista no encontrado")
    repo.conn.commit()
    return row


# ══════════════════════════════════════════════════════════════════════════
# Panelista — por token de invitación
# ══════════════════════════════════════════════════════════════════════════

def _participant_by_token(repo: DelphiRepo, token: str) -> Dict:
    row = repo.q1("""
        select p.*, s.id as study_id, s.title as study_title,
               s.dichotomy_threshold, s.scale_type, s.stopping_rule, s.horizon_year
          from delphi_participants p
          join delphi_studies s on s.id = p.study_id
         where p.invitation_token = %s
    """, (token,))
    if not row:
        raise HTTPException(404, "Invitación no válida")
    if row["status"] == "dropped":
        raise HTTPException(403, "Este panelista está dado de baja del estudio")
    return row


def _open_round(repo: DelphiRepo, sid: str) -> Optional[Dict]:
    return repo.q1(
        "select * from delphi_rounds where study_id = %s and status = 'open' "
        "order by round_number desc limit 1", (sid,))


def _build_feedback(repo: DelphiRepo, sid: str, participant_id: str,
                    round_number: int, config: StudyConfig) -> Optional[Dict]:
    """
    §5.1 — la devolución. Se construye sobre la ronda inmediatamente anterior
    a la que el panelista va a contestar. En la ronda 1 no hay devolución, y
    eso se dice con `None`, no con un paquete vacío.
    """
    if round_number <= 1:
        return None
    all_responses = repo.load_responses(sid)
    previous = [r for r in all_responses if r.round_number == round_number - 1]
    if not previous:
        return None
    names, _ = repo.variable_lookup(sid)
    return feedback_package(participant_id, previous, config, variable_names=names)


@router.get("/invite/{token}")
def get_invitation(token: UUID, repo: DelphiRepo = Depends(get_checked_repo)):
    """
    Lo que ve el panelista: el formulario de la ronda abierta y, junto a cada
    variable, su respuesta anterior, la distribución del panel y los
    argumentos anónimos de quienes opinaron distinto (§5.1).

    «Sin ello la ronda 2 es una re-votación a ciegas.»
    """
    part = _participant_by_token(repo, str(token))
    sid = str(part["study_id"])
    study = repo.get_study(sid)
    config = repo.study_config(study)

    rnd = _open_round(repo, sid)
    if not rnd:
        return {
            "study": {"id": sid, "title": part["study_title"],
                      "horizon_year": part["horizon_year"]},
            "participant": {"id": str(part["id"]), "email": part["email"],
                            "display_name": part["display_name"], "k": part["k"]},
            "round": None,
            "message": "No hay ninguna ronda abierta en este momento.",
        }

    questions = repo.q("""
        select q.id as question_id, q.variable_id, q.title, q.description,
               q.type, q.order_index, q.required,
               v.code, v.family
          from delphi_questions q
          left join delphi_variables v on v.id = q.variable_id
         where q.round_id = %s order by q.order_index
    """, (rnd["id"],))

    existing = repo.q("""
        select q.variable_id, a.value, a.is_abstention, a.argument, s.status
          from delphi_answers a
          join delphi_submissions s on s.id = a.submission_id
          join delphi_questions   q on q.id = a.question_id
         where s.round_id = %s and s.participant_id = %s
    """, (rnd["id"], part["id"]))
    draft = {
        str(e["variable_id"]): {
            "value": _as_number(e["value"]),
            "is_abstention": e["is_abstention"],
            "argument": e["argument"],
        } for e in existing if e["variable_id"]
    }
    already_submitted = any(e["status"] == "submitted" for e in existing)

    feedback = _build_feedback(repo, sid, str(part["id"]),
                               int(rnd["round_number"]), config)
    fb_by_var = {i["variable_id"]: i for i in (feedback or {}).get("items", [])}
    for q in questions:
        q["question_id"] = str(q["question_id"])
        q["variable_id"] = str(q["variable_id"]) if q["variable_id"] else None
        q["draft"] = draft.get(q["variable_id"] or "")
        q["feedback"] = fb_by_var.get(q["variable_id"] or "")

    return {
        "study": {
            "id": sid, "title": study["title"], "description": study["description"],
            "horizon_year": study.get("horizon_year"),
            "scale_type": study.get("scale_type"),
            "dichotomy_threshold": study.get("dichotomy_threshold"),
        },
        "participant": {"id": str(part["id"]), "email": part["email"],
                        "display_name": part["display_name"], "k": part["k"]},
        "round": {"id": str(rnd["id"]), "round_number": rnd["round_number"],
                  "title": rnd["title"], "instructions": rnd["instructions"]},
        "questions": questions,
        "already_submitted": already_submitted,
        "has_feedback": feedback is not None,
    }


@router.get("/invite/{token}/feedback")
def get_feedback(token: UUID, round_number: Optional[int] = Query(None),
                 repo: DelphiRepo = Depends(get_checked_repo)):
    """El paquete de devolución de §5.1 por separado, para previsualizarlo."""
    part = _participant_by_token(repo, str(token))
    sid = str(part["study_id"])
    config = repo.study_config(repo.get_study(sid))
    if round_number is None:
        rnd = _open_round(repo, sid)
        round_number = int(rnd["round_number"]) if rnd else 2
    fb = _build_feedback(repo, sid, str(part["id"]), int(round_number), config)
    if fb is None:
        raise HTTPException(404, "No hay ronda anterior de la que devolver nada")
    return fb


@router.post("/invite/{token}/responses")
def submit_responses(token: UUID, body: SubmissionIn,
                     repo: DelphiRepo = Depends(get_checked_repo)):
    """
    Paso 2 del flujo (§8). Una respuesta por (panelista, variable), con
    argumento opcional.

    La abstención es un campo aparte, nunca un número dentro de la escala
    (§3, §8): si `is_abstention` es cierto, `value` se guarda como NULL.
    """
    part = _participant_by_token(repo, str(token))
    sid = str(part["study_id"])
    rnd = _open_round(repo, sid)
    if not rnd:
        raise HTTPException(409, "No hay ninguna ronda abierta")

    qmap = {
        str(q["variable_id"]): q["id"]
        for q in repo.q(
            "select id, variable_id from delphi_questions "
            "where round_id = %s and variable_id is not null", (rnd["id"],))
    }

    sub = repo.q1("""
        insert into delphi_submissions (participant_id, round_id, status)
        values (%s, %s, 'draft')
        on conflict (participant_id, round_id) do update set status = delphi_submissions.status
        returning *
    """, (part["id"], rnd["id"]))
    if sub["status"] == "submitted":
        raise HTTPException(409, "Este panelista ya entregó esta ronda")

    saved, unknown = 0, []
    for ans in body.answers:
        qid = qmap.get(str(ans.variable_id))
        if not qid:
            unknown.append(str(ans.variable_id))
            continue
        if ans.is_abstention:
            value, abst = None, True
        elif ans.value is None:
            # Sin valor y sin declarar abstención: se registra como NS/NR
            # explícito, jamás como un número inventado.
            value, abst = None, True
        else:
            value, abst = Json(float(ans.value)), False
        repo.q("""
            insert into delphi_answers
                (submission_id, question_id, value, is_abstention, argument, answered_at)
            values (%s, %s, %s, %s, %s, now())
            on conflict (submission_id, question_id) do update
              set value = excluded.value,
                  is_abstention = excluded.is_abstention,
                  argument = excluded.argument,
                  answered_at = now()
        """, (sub["id"], qid, value, abst, (ans.argument or "").strip() or None))
        saved += 1

    if body.submit:
        repo.q("update delphi_submissions set status = 'submitted', submitted_at = now() "
               "where id = %s", (sub["id"],))
        repo.q("update delphi_participants set status = 'active', "
               "registered_at = coalesce(registered_at, now()) where id = %s", (part["id"],))
    repo.conn.commit()

    return {
        "submission_id": str(sub["id"]),
        "round_number": rnd["round_number"],
        "answers_saved": saved,
        "submitted": body.submit,
        "unknown_variable_ids": unknown,
    }


# ══════════════════════════════════════════════════════════════════════════
# Análisis (§9)
# ══════════════════════════════════════════════════════════════════════════

def _report(repo: DelphiRepo, sid: str, threshold: Optional[float] = None) -> Dict:
    study = repo.get_study(sid)
    config = repo.study_config(study)
    if threshold is not None:
        # §5.3: mover el umbral a posteriori y ver si el mapa aguanta.
        config = StudyConfig(
            dichotomy_threshold=float(threshold),
            scale_min=config.scale_min, scale_max=config.scale_max,
            stability_min=config.stability_min, iqr_max=config.iqr_max,
            consensus_min=config.consensus_min, k_min=config.k_min,
        )
    names, families = repo.variable_lookup(sid)
    n_invited = repo.q1(
        "select count(*) as n from delphi_participants where study_id = %s", (sid,))["n"]

    report = compute_study_report(
        repo.load_responses(sid),
        config,
        variable_names=names,
        variable_families=families,
        panelists_k=repo.panelists_k(sid),
        n_invited=int(n_invited),
        round_population=repo.round_population(sid),
    )
    report["study"] = {
        "id": sid, "title": study["title"], "status": study["status"],
        "horizon_year": study.get("horizon_year"),
        "scale_type": study.get("scale_type"),
    }
    return report


@router.get("/studies/{study_id}/stats")
def study_stats(study_id: UUID,
                threshold: Optional[float] = Query(
                    None, description="Umbral alternativo para reanalizar (§5.3)"),
                repo: DelphiRepo = Depends(get_checked_repo)):
    """
    El bloque §9 completo: por variable y ronda (población de agregado), entre
    rondas (población de cambio, con McNemar), y por panel (deserción, §6.2,
    K medio, criterio de parada). Más el mapa de §7 y Spearman.

    Las dos poblaciones de §6 salen siempre etiquetadas y por separado; no hay
    ninguna cifra en la respuesta que las mezcle.
    """
    return _report(repo, str(study_id), threshold)


@router.get("/studies/{study_id}/rounds/{round_number}/stats")
def round_stats(study_id: UUID, round_number: int,
                repo: DelphiRepo = Depends(get_checked_repo)):
    """Una sola ronda: la tabla por agrupador y las correlaciones de Spearman."""
    sid = str(study_id)
    study = repo.get_study(sid)
    config = repo.study_config(study)
    names, families = repo.variable_lookup(sid)

    responses = [r for r in repo.load_responses(sid) if r.round_number == round_number]
    if not responses:
        raise HTTPException(404, f"La ronda {round_number} no tiene respuestas entregadas")

    population = repo.round_population(sid).get(round_number, len({r.participant_id for r in responses}))
    by_var: Dict[str, List[DelphiResponse]] = {}
    for r in responses:
        by_var.setdefault(r.variable_id, []).append(r)

    rows = []
    for vid, rs in by_var.items():
        s = variable_round_stats(rs, config, n_population=population,
                                 variable_id=vid, round_number=round_number)
        s["variable_name"] = names.get(vid, vid)
        s["family"] = families.get(vid)
        rows.append(s)
    rows.sort(key=lambda s: (s["family"] or "~", -(s["mean"] or 0)))

    grouped: Dict[str, List[Dict]] = {}
    for s in rows:
        grouped.setdefault(s["family"] or "Sin agrupador", []).append(s)

    return {
        "round_number": round_number,
        "poblacion": "agregado",
        "n_respondents": len({r.participant_id for r in responses}),
        "n_population": population,
        "variables": rows,
        "by_family": [{"family": f, "variables": v} for f, v in sorted(grouped.items())],
        "spearman": spearman_matrix(responses, round_number=round_number, variable_names=names),
    }


@router.get("/studies/{study_id}/map")
def study_map(study_id: UUID,
              threshold: Optional[float] = Query(None),
              repo: DelphiRepo = Depends(get_checked_repo)):
    """
    El mapa de variables en cuatro cuadrantes (§7) y, con él, la lista de
    variables que no cerraron: el contenido de la ronda siguiente.
    """
    report = _report(repo, str(study_id), threshold)
    return report["map"]


@router.get("/studies/{study_id}/transitions")
def study_transitions(study_id: UUID,
                      round_from: Optional[int] = Query(None),
                      repo: DelphiRepo = Depends(get_checked_repo)):
    """
    Sólo el bloque entre rondas: matriz a/b/c/d y McNemar por variable, sobre
    la población de cambio. Es lo que el notebook dejaba a medias.
    """
    report = _report(repo, str(study_id))
    rows = report["between"]
    if round_from is not None:
        rows = [r for r in rows if r["round_from"] == round_from]
    return {
        "poblacion": "cambio",
        "transitions": rows,
        "attrition_check": report["attrition_check"],
    }
