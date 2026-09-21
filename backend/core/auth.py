"""
backend/core/auth.py

Autenticación y autorización de Prospecta.

Quién eres lo decide Supabase (GoTrue emite el JWT); qué puedes hacer lo
decide esta capa, y lo que puedes *tocar* lo decide la RLS de la migración
002. Son tres controles distintos y deliberadamente redundantes:

    JWT        → identidad verificada criptográficamente
    require_*  → autorización de endpoint, falla rápido y con un 403 legible
    RLS        → última línea; protege aunque un endpoint se olvide del guard

El proyecto usa **claves de firma asimétricas** (ES256, verificadas contra el
JWKS público del proyecto). Eso significa que el backend NO necesita ningún
secreto compartido para validar un token: sólo la clave pública, que se
descarga y se cachea. Se mantiene el camino HS256 por si el proyecto se
revierte a la firma simétrica heredada, y en ese caso hace falta
SUPABASE_JWT_SECRET en el entorno.

Uso típico en una ruta:

    from core.auth import CurrentUser, require_role

    @router.get("/estudios")
    def listar(user: CurrentUser):
        ...

    @router.post("/estudios", dependencies=[Depends(require_role("profesor"))])
    def crear(...):
        ...
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# main.py ya carga el .env; esto sólo cubre el caso de importar core.auth
# desde un script suelto. load_dotenv no pisa variables ya presentes.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── Roles ──────────────────────────────────────────────────────────────────
# Reflejan el enum user_role de Postgres. Si se añade uno allí, se añade aquí:
# son el mismo vocabulario visto desde dos lados.

Role = Literal["admin", "profesor", "experto"]

ROLES: tuple[Role, ...] = ("admin", "profesor", "experto")

DEFAULT_ROLE: Role = "experto"
"""El mismo que aplican handle_new_user() y custom_access_token(): quien entra
sin asignación previa en role_assignments es experto. Es el rol que menos
puede, que es lo que debe pasar cuando no se sabe."""

ROLE_LABELS: dict[Role, str] = {
    "admin": "Administrador",
    "profesor": "Profesor",
    "experto": "Experto",
}


class AuthError(Exception):
    """Fallo de verificación del token. Se traduce a 401 en la dependencia."""


# ── Configuración ──────────────────────────────────────────────────────────


class _Settings(BaseModel):
    supabase_url: str
    service_key: str
    anon_key: str = ""
    jwt_secret: str = ""

    @property
    def issuer(self) -> str:
        return f"{self.supabase_url}/auth/v1"

    @property
    def jwks_url(self) -> str:
        return f"{self.supabase_url}/auth/v1/.well-known/jwks.json"


@lru_cache(maxsize=1)
def settings() -> _Settings:
    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    if not url:
        raise RuntimeError(
            "SUPABASE_URL no está configurada: el backend no puede verificar tokens."
        )
    return _Settings(
        supabase_url=url,
        service_key=os.getenv("SUPABASE_SERVICE_KEY") or "",
        anon_key=os.getenv("SUPABASE_ANON_KEY") or "",
        jwt_secret=os.getenv("SUPABASE_JWT_SECRET") or "",
    )


AUDIENCE = "authenticated"
LEEWAY_SECONDS = 10  # tolerancia de reloj entre el emisor y este servidor


@lru_cache(maxsize=1)
def _jwks_client() -> PyJWKClient:
    # Cachea las claves y sólo vuelve a pedirlas cuando aparece un `kid`
    # desconocido — que es justo lo que ocurre durante una rotación de claves.
    return PyJWKClient(settings().jwks_url, cache_keys=True, lifespan=3600)


# ── Verificación del token ─────────────────────────────────────────────────


def decode_token(token: str) -> dict[str, Any]:
    """Verifica firma, emisor, audiencia y vigencia. Devuelve los claims.

    Nunca devuelve un payload sin verificar: si algo no cuadra, lanza
    AuthError. Decodificar sin validar es el error clásico de este tipo de
    código y aquí no hay ninguna ruta que lo permita.
    """
    if not token or token.count(".") != 2:
        raise AuthError("El token no tiene forma de JWT.")

    cfg = settings()

    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise AuthError(f"Cabecera del token ilegible: {exc}") from exc

    alg = header.get("alg", "")

    try:
        if alg == "HS256":
            # Proyecto con firma simétrica heredada.
            if not cfg.jwt_secret:
                raise AuthError(
                    "El token está firmado con HS256 pero SUPABASE_JWT_SECRET "
                    "no está configurada."
                )
            key: Any = cfg.jwt_secret
        elif alg in ("ES256", "RS256"):
            key = _jwks_client().get_signing_key_from_jwt(token).key
        else:
            raise AuthError(f"Algoritmo de firma no admitido: {alg!r}")

        return jwt.decode(
            token,
            key,
            algorithms=[alg],
            audience=AUDIENCE,
            issuer=cfg.issuer,
            leeway=LEEWAY_SECONDS,
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("La sesión ha expirado.") from exc
    except jwt.InvalidAudienceError as exc:
        raise AuthError("El token no es de esta aplicación.") from exc
    except jwt.InvalidIssuerError as exc:
        raise AuthError("El token no lo emitió este proyecto de Supabase.") from exc
    except jwt.PyJWTError as exc:
        raise AuthError(f"Token inválido: {exc}") from exc
    except AuthError:
        raise
    except Exception as exc:  # p. ej. el JWKS no responde
        logger.exception("Fallo al verificar el JWT")
        raise AuthError(f"No se pudo verificar el token: {exc}") from exc


# ── El usuario autenticado ─────────────────────────────────────────────────


class AuthUser(BaseModel):
    """Identidad ya verificada. Lo que llega a las rutas."""

    id: str
    email: Optional[str] = None
    role: Role = DEFAULT_ROLE
    full_name: Optional[str] = None
    institution: Optional[str] = None
    token: str = Field(repr=False)
    claims: dict[str, Any] = Field(default_factory=dict, repr=False)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def has_role(self, *roles: Role, admin_override: bool = True) -> bool:
        """El admin pasa por cualquier puerta; el resto sólo por la suya."""
        if admin_override and self.role == "admin":
            return True
        return self.role in roles


def role_from_claims(claims: dict[str, Any]) -> Optional[Role]:
    """Extrae el rol del JWT si el hook custom_access_token lo puso.

    El hook de la base de datos inyecta `user_role` en cada token que emite.
    Se mira también dentro de app_metadata porque es donde acaba si algún día
    se configura el rol por esa vía en lugar del hook.
    """
    candidates = [
        claims.get("user_role"),
        (claims.get("app_metadata") or {}).get("user_role"),
        (claims.get("app_metadata") or {}).get("role"),
    ]
    for value in candidates:
        if isinstance(value, str) and value in ROLES:
            return value  # type: ignore[return-value]
    return None


def resolve_role(user_id: str, claims: dict[str, Any]) -> Role:
    """Rol efectivo: el claim si está, y si no, la tabla profiles.

    El claim se emite al iniciar sesión y NO cambia si un admin reasigna el
    rol a media sesión. Para lecturas eso es tolerable —el token dura una
    hora—, pero una degradación de privilegios no debería esperar a que
    caduque un token. Ahí está `resolve_role_live`.
    """
    claimed = role_from_claims(claims)
    if claimed is not None:
        return claimed

    profile = fetch_profile(user_id)
    if profile and profile.get("role") in ROLES:
        return profile["role"]  # type: ignore[return-value]
    return DEFAULT_ROLE


def resolve_role_live(user_id: str) -> Role:
    """Rol leído de profiles, ignorando el token. Para operaciones sensibles."""
    profile = fetch_profile(user_id)
    if profile and profile.get("role") in ROLES:
        return profile["role"]  # type: ignore[return-value]
    return DEFAULT_ROLE


# ── Clientes de Supabase ───────────────────────────────────────────────────


@lru_cache(maxsize=1)
def service_client():
    """Cliente con service_role: **se salta la RLS por completo**.

    Sólo para lo que la RLS no puede hacer por definición:
      · el agregado anónimo de la devolución entre rondas (LOGICA_DELPHI §5.1),
      · el canje del token de invitación de un panelista,
      · la administración de perfiles y roles.

    Cada uso tiene que ir detrás de un require_role(), porque aquí la base de
    datos ya no protege nada.
    """
    from supabase import create_client  # import perezoso: acelera el arranque

    cfg = settings()
    if not cfg.service_key:
        raise RuntimeError("SUPABASE_SERVICE_KEY no está configurada.")
    return create_client(cfg.supabase_url, cfg.service_key)


def user_client(token: str):
    """Cliente que actúa **como el usuario**: la RLS se le aplica entera.

    Es el cliente por defecto para leer datos de Delphi en nombre de alguien.
    Si la política de 002 dice que no puede ver una fila, aquí tampoco la ve,
    aunque el endpoint tenga un fallo de lógica.
    """
    from supabase import create_client

    cfg = settings()
    if not cfg.anon_key:
        raise RuntimeError(
            "SUPABASE_ANON_KEY no está configurada: sin la clave publicable no "
            "se puede construir un cliente con la identidad del usuario."
        )
    client = create_client(cfg.supabase_url, cfg.anon_key)
    client.postgrest.auth(token)
    return client


def fetch_profile(user_id: str) -> Optional[dict[str, Any]]:
    """Perfil desde profiles, con service_role. None si no existe."""
    try:
        res = (
            service_client()
            .table("profiles")
            .select("id, email, full_name, institution, role, created_at, updated_at")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
    except Exception:
        logger.exception("No se pudo leer el perfil %s", user_id)
        return None
    data = res.data or []
    return data[0] if data else None


# ── Dependencias de FastAPI ────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False, description="JWT de Supabase")


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(_bearer)
    ] = None,
) -> AuthUser:
    """Exige una sesión válida. 401 si no la hay."""
    if credentials is None or not credentials.credentials:
        raise _unauthorized("Falta la cabecera Authorization: Bearer.")

    token = credentials.credentials
    try:
        claims = decode_token(token)
    except AuthError as exc:
        raise _unauthorized(str(exc)) from exc

    user_id = claims.get("sub")
    if not user_id:
        raise _unauthorized("El token no identifica a ningún usuario.")

    metadata = claims.get("user_metadata") or {}
    return AuthUser(
        id=user_id,
        email=claims.get("email") or metadata.get("email"),
        role=resolve_role(user_id, claims),
        full_name=metadata.get("full_name"),
        institution=metadata.get("institution"),
        token=token,
        claims=claims,
    )


def get_optional_user(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(_bearer)
    ] = None,
) -> Optional[AuthUser]:
    """Sesión si la hay, None si no. Para endpoints que enriquecen la
    respuesta cuando hay usuario pero funcionan sin él."""
    if credentials is None or not credentials.credentials:
        return None
    try:
        return get_current_user(credentials)
    except HTTPException:
        return None


def require_role(*roles: Role, admin_override: bool = True):
    """Dependencia que exige uno de los roles dados.

    `require_role("profesor")` deja pasar también al admin: en esta plataforma
    admin es un superconjunto, no un rol paralelo. Para el caso raro en que
    eso no valga —una acción que sólo tiene sentido como profesor— está
    `admin_override=False`.

    401 si no hay sesión (no sabemos quién eres), 403 si la hay pero no basta
    (sabemos quién eres y no te toca). La distinción importa: el frontend
    manda al login en el primer caso y muestra un mensaje en el segundo.
    """
    unknown = set(roles) - set(ROLES)
    if unknown:
        raise ValueError(f"Rol desconocido en require_role: {sorted(unknown)}")

    def dependency(user: Annotated[AuthUser, Depends(get_current_user)]) -> AuthUser:
        if not user.has_role(*roles, admin_override=admin_override):
            esperado = " o ".join(ROLE_LABELS[r] for r in roles)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Se requiere el rol {esperado}. "
                    f"Tu rol actual es {ROLE_LABELS[user.role]}."
                ),
            )
        return user

    return dependency


def require_fresh_role(*roles: Role, admin_override: bool = True):
    """Como require_role, pero releyendo el rol de profiles en cada petición.

    Para lo que no puede esperar a que caduque un token: cambiar roles, borrar
    un estudio, cerrar una ronda. Cuesta una consulta; se usa donde el coste
    de un privilegio revocado hace una hora es mayor que esa consulta.
    """

    def dependency(user: Annotated[AuthUser, Depends(get_current_user)]) -> AuthUser:
        live = resolve_role_live(user.id)
        fresh = user.model_copy(update={"role": live})
        if not fresh.has_role(*roles, admin_override=admin_override):
            esperado = " o ".join(ROLE_LABELS[r] for r in roles)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Se requiere el rol {esperado}. "
                    f"Tu rol actual es {ROLE_LABELS[live]}."
                ),
            )
        return fresh

    return dependency


# Atajos legibles en las firmas de las rutas.
CurrentUser = Annotated[AuthUser, Depends(get_current_user)]
OptionalUser = Annotated[Optional[AuthUser], Depends(get_optional_user)]
AdminUser = Annotated[AuthUser, Depends(require_role("admin"))]
StudyManager = Annotated[AuthUser, Depends(require_role("profesor"))]

__all__ = [
    "AdminUser",
    "AuthError",
    "AuthUser",
    "CurrentUser",
    "DEFAULT_ROLE",
    "OptionalUser",
    "ROLES",
    "ROLE_LABELS",
    "Role",
    "StudyManager",
    "decode_token",
    "fetch_profile",
    "get_current_user",
    "get_optional_user",
    "require_fresh_role",
    "require_role",
    "resolve_role",
    "resolve_role_live",
    "service_client",
    "settings",
    "user_client",
]
