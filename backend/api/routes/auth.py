"""
backend/api/routes/auth.py

Sesión, perfil y administración de roles.

GET    /api/auth/session                  → los claims verificados del token
GET    /api/auth/me                       → perfil del usuario autenticado
PATCH  /api/auth/me                       → editar nombre e institución
GET    /api/auth/me/permissions           → qué puede hacer, resuelto en un sitio
GET    /api/auth/users                    → (admin) listado de perfiles
PATCH  /api/auth/users/{user_id}/role     → (admin) cambiar el rol de alguien
GET    /api/auth/role-assignments         → (admin) roles pre-asignados por correo
PUT    /api/auth/role-assignments         → (admin) crear o actualizar uno
DELETE /api/auth/role-assignments/{email} → (admin) eliminarlo

El registro y el inicio de sesión NO están aquí: los hace el navegador contra
GoTrue directamente (frontend/lib/auth.ts). Este backend nunca ve una
contraseña, y es lo correcto — no tiene nada que aportar en ese camino salvo
un sitio más donde se pueda filtrar.

Sobre role_assignments: es la lista de invitados. Quien se registra con un
correo que figura ahí recibe ese rol en el acto, por obra del trigger
handle_new_user(). Es la única forma de que alguien nazca siendo profesor sin
que un admin tenga que promoverlo a mano después de que se registre.

Las escrituras van con service_role (se saltan la RLS) y por eso cada una
está detrás de require_role("admin"): aquí la autorización es enteramente
responsabilidad de este archivo.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status
from pydantic import BaseModel, EmailStr, Field

from core.auth import (
    ROLE_LABELS,
    ROLES,
    AuthUser,
    CurrentUser,
    Role,
    fetch_profile,
    require_role,
    service_client,
)

logger = logging.getLogger(__name__)
router = APIRouter()

AdminOnly = Annotated[AuthUser, Depends(require_role("admin"))]


# ── Esquemas ───────────────────────────────────────────────────────────────


class Profile(BaseModel):
    id: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    institution: Optional[str] = None
    role: Role
    role_label: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=120)
    institution: Optional[str] = Field(None, max_length=160)


class SessionInfo(BaseModel):
    """Lo que el backend ve del token, ya verificado. Sin el token dentro."""

    user_id: str
    email: Optional[str]
    role: Role
    role_label: str
    issued_at: Optional[int] = None
    expires_at: Optional[int] = None
    issuer: Optional[str] = None


class Permissions(BaseModel):
    role: Role
    role_label: str
    is_admin: bool
    can_manage_users: bool
    can_create_studies: bool
    can_manage_own_studies: bool
    can_answer_rounds: bool
    can_ingest_corpus: bool


class RoleAssignment(BaseModel):
    email: EmailStr
    role: Role
    note: Optional[str] = None
    created_at: Optional[str] = None


class RoleAssignmentUpsert(BaseModel):
    email: EmailStr
    role: Role
    note: Optional[str] = Field(None, max_length=300)


class RoleChange(BaseModel):
    role: Role


# ── Utilidades ─────────────────────────────────────────────────────────────


def _as_profile(row: dict[str, Any]) -> Profile:
    role: Role = row.get("role") if row.get("role") in ROLES else "experto"
    return Profile(
        id=str(row["id"]),
        email=row.get("email"),
        full_name=row.get("full_name"),
        institution=row.get("institution"),
        role=role,
        role_label=ROLE_LABELS[role],
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _ensure_profile(user: AuthUser) -> dict[str, Any]:
    """Devuelve el perfil, creándolo si falta.

    No debería faltar: handle_new_user() lo crea al darse de alta. Falta en un
    caso real —usuarios de auth.users anteriores al trigger— y el síntoma sería
    una sesión válida contra una aplicación que dice no conocerte. Se repara
    aquí, respetando role_assignments igual que hace el trigger.
    """
    profile = fetch_profile(user.id)
    if profile:
        return profile

    assigned: Optional[str] = None
    if user.email:
        try:
            res = (
                service_client()
                .table("role_assignments")
                .select("role")
                .eq("email", user.email.lower())
                .limit(1)
                .execute()
            )
            if res.data:
                assigned = res.data[0].get("role")
        except Exception:
            logger.exception("No se pudo consultar role_assignments para %s", user.id)

    payload = {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "institution": user.institution,
        "role": assigned if assigned in ROLES else "experto",
    }
    try:
        res = service_client().table("profiles").insert(payload).execute()
        if res.data:
            return res.data[0]
    except Exception:
        logger.exception("No se pudo crear el perfil de %s", user.id)

    # Último recurso: responder con lo que dice el token en vez de un 500.
    return payload


def _count_admins() -> int:
    res = (
        service_client()
        .table("profiles")
        .select("id", count="exact")
        .eq("role", "admin")
        .execute()
    )
    return res.count or 0


# ── Sesión y perfil ────────────────────────────────────────────────────────


@router.get("/session", response_model=SessionInfo, summary="Sesión actual")
def get_session(user: CurrentUser) -> SessionInfo:
    """Confirma que el token es válido y dice qué contiene.

    El frontend lo usa como ping de sesión: si responde 200, la sesión sirve;
    si responde 401, hay que renovar o volver al login.
    """
    return SessionInfo(
        user_id=user.id,
        email=user.email,
        role=user.role,
        role_label=ROLE_LABELS[user.role],
        issued_at=user.claims.get("iat"),
        expires_at=user.claims.get("exp"),
        issuer=user.claims.get("iss"),
    )


@router.get("/me", response_model=Profile, summary="Mi perfil")
def get_me(user: CurrentUser) -> Profile:
    return _as_profile(_ensure_profile(user))


@router.patch("/me", response_model=Profile, summary="Editar mi perfil")
def update_me(payload: ProfileUpdate, user: CurrentUser) -> Profile:
    """Nombre e institución. El rol no: no es un campo de perfil, es un
    permiso, y nadie se lo concede a sí mismo (el trigger protect_role_change
    lo impediría de todos modos)."""
    changes = payload.model_dump(exclude_none=True)
    if not changes:
        return _as_profile(_ensure_profile(user))

    _ensure_profile(user)
    try:
        res = (
            service_client()
            .table("profiles")
            .update(changes)
            .eq("id", user.id)
            .execute()
        )
    except Exception as exc:
        logger.exception("Error actualizando el perfil %s", user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo actualizar el perfil.",
        ) from exc

    if not res.data:
        raise HTTPException(status_code=404, detail="Perfil no encontrado.")
    return _as_profile(res.data[0])


@router.get(
    "/me/permissions", response_model=Permissions, summary="Qué puedo hacer"
)
def get_permissions(user: CurrentUser) -> Permissions:
    """Traduce el rol a capacidades concretas.

    Existe para que el frontend no reimplemente la tabla de permisos: si
    mañana un profesor puede ingerir corpus, cambia esta función y la interfaz
    se entera sola. Es conveniencia de interfaz, no un control de seguridad —
    quien decide de verdad son require_role() y la RLS.
    """
    role = user.role
    return Permissions(
        role=role,
        role_label=ROLE_LABELS[role],
        is_admin=role == "admin",
        can_manage_users=role == "admin",
        can_create_studies=role in ("admin", "profesor"),
        can_manage_own_studies=role in ("admin", "profesor"),
        can_answer_rounds=True,  # cualquiera puede ser panelista de un estudio
        can_ingest_corpus=role == "admin",
    )


# ── Administración de usuarios ─────────────────────────────────────────────


@router.get("/users", response_model=list[Profile], summary="(admin) Usuarios")
def list_users(
    _: AdminOnly,
    role: Optional[Role] = Query(None, description="Filtrar por rol"),
    search: Optional[str] = Query(None, max_length=120, description="Correo o nombre"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[Profile]:
    query = (
        service_client()
        .table("profiles")
        .select("id, email, full_name, institution, role, created_at, updated_at")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
    )
    if role:
        query = query.eq("role", role)
    if search:
        needle = search.replace(",", " ").strip()
        query = query.or_(f"email.ilike.%{needle}%,full_name.ilike.%{needle}%")

    res = query.execute()
    return [_as_profile(row) for row in (res.data or [])]


@router.patch(
    "/users/{user_id}/role",
    response_model=Profile,
    summary="(admin) Cambiar el rol de un usuario",
)
def change_user_role(
    payload: RoleChange,
    admin: AdminOnly,
    user_id: str = Path(..., min_length=36, max_length=36),
) -> Profile:
    """Dos cerrojos, por la misma razón: que la plataforma no se quede sin
    administrador. Un despiste aquí no se arregla desde la propia aplicación,
    hay que entrar a Supabase por SQL."""
    if user_id == admin.id and payload.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No puedes quitarte a ti mismo el rol de administrador.",
        )

    current = fetch_profile(user_id)
    if not current:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")

    if current.get("role") == "admin" and payload.role != "admin":
        if _count_admins() <= 1:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Es el único administrador: nombra otro antes de degradarlo.",
            )

    try:
        res = (
            service_client()
            .table("profiles")
            .update({"role": payload.role})
            .eq("id", user_id)
            .execute()
        )
    except Exception as exc:
        logger.exception("Error cambiando el rol de %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo cambiar el rol.",
        ) from exc

    if not res.data:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")

    logger.info("Rol de %s cambiado a %s por %s", user_id, payload.role, admin.id)
    return _as_profile(res.data[0])


# ── Roles pre-asignados por correo ─────────────────────────────────────────


@router.get(
    "/role-assignments",
    response_model=list[RoleAssignment],
    summary="(admin) Roles pre-asignados",
)
def list_role_assignments(_: AdminOnly) -> list[RoleAssignment]:
    res = (
        service_client()
        .table("role_assignments")
        .select("email, role, note, created_at")
        .order("email")
        .execute()
    )
    return [RoleAssignment(**row) for row in (res.data or [])]


@router.put(
    "/role-assignments",
    response_model=RoleAssignment,
    summary="(admin) Asignar un rol a un correo",
)
def upsert_role_assignment(
    payload: RoleAssignmentUpsert, admin: AdminOnly
) -> RoleAssignment:
    """Prepara el rol de alguien que todavía no se ha registrado.

    Si ya tiene cuenta, esto NO le cambia el rol: handle_new_user() sólo lee
    esta tabla al crear el perfil. Para alguien que ya existe hay que usar
    PATCH /users/{id}/role. Son dos operaciones distintas a propósito —
    confundirlas es prometer un cambio que no ocurre.
    """
    row = {
        "email": payload.email.lower(),
        "role": payload.role,
        "note": payload.note,
    }
    try:
        res = (
            service_client()
            .table("role_assignments")
            .upsert(row, on_conflict="email")
            .execute()
        )
    except Exception as exc:
        logger.exception("Error al asignar rol a %s", payload.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo guardar la asignación.",
        ) from exc

    logger.info("Asignación %s → %s por %s", row["email"], payload.role, admin.id)
    data = (res.data or [row])[0]
    return RoleAssignment(**data)


@router.delete(
    "/role-assignments/{email}",
    status_code=status.HTTP_204_NO_CONTENT,
    # `-> None` haría que FastAPI dedujese NoneType como response_model y un
    # 204 no admite cuerpo. Se desactiva explícitamente.
    response_model=None,
    response_class=Response,
    summary="(admin) Eliminar una asignación",
)
def delete_role_assignment(_: AdminOnly, email: EmailStr = Path(...)) -> None:
    try:
        service_client().table("role_assignments").delete().eq(
            "email", email.lower()
        ).execute()
    except Exception as exc:
        logger.exception("Error eliminando la asignación de %s", email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo eliminar la asignación.",
        ) from exc
