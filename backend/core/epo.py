"""
Cliente de EPO OPS 3.2 (Open Patent Services) para conteos anuales de patentes.

Por qué existe: la vía de patentes por Scopus está cerrada —`DOCTYPE(ip)`
responde 200 y devuelve siempre 0 resultados, ya verificado (docs §6)— y la
alternativa actual es que el usuario exporte un CSV a mano. EPO OPS da el
mismo dato por API y con cobertura mundial (DOCDB).

Cómo funciona OPS:

  1. OAuth2 client_credentials contra /3.2/auth/accesstoken con Basic auth
     sobre `consumer_key:consumer_secret`. Devuelve un bearer que caduca a
     los 20 minutos.
  2. Búsqueda CQL en /3.2/rest-services/published-data/search. Para un conteo
     basta pedir `Range=1-1` y leer `@total-result-count`: una sola petición
     por año, sin paginar.

Credenciales (verificado contra el servicio real, 2026-09-21):

    EPO_CONSUMER_KEY  → consumer key   (client id)
    EPO_API_KEY       → consumer secret

  El orden inverso responde 401 `ClientId is Invalid`. Se prueban ambos
  órdenes en `authenticate()` por si la convención cambia, pero el primero
  que se intenta es el correcto.

Cuotas: OPS no devuelve 429 sino que publica su presupuesto en la cabecera
`X-Throttling-Control`, por ejemplo

    idle (images=green:200, …, search=green:30)

donde `search=green:30` son 30 búsquedas por minuto. El cliente lee ese
número en cada respuesta y ajusta la pausa entre peticiones (60/n). Bajo
carga OPS baja a 5/min (color `red`) y las pausas suben a 12 s: un rango de
27 años tarda entonces unos 5 minutos. El color rojo NO es un fallo —esas
respuestas traen datos— y sólo un 403 con `X-Rejection-Reason` detiene la
serie de verdad.

Un detalle contraintuitivo: **404 no es un error**. OPS responde
`SERVER.EntityNotFound / No results found` cuando la búsqueda es válida pero
no casa nada. Eso es un cero, y tratarlo como fallo dejaría huecos en la
serie justo en los años iniciales, que son los que fijan el arranque de la
curva.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import time
from typing import AsyncGenerator

import httpx

OPS_BASE = "https://ops.epo.org/3.2"
OPS_AUTH_URL = f"{OPS_BASE}/auth/accesstoken"
OPS_SEARCH_URL = f"{OPS_BASE}/rest-services/published-data/search"

# Margen de seguridad sobre la caducidad del token (OPS da 1200 s).
TOKEN_SAFETY_MARGIN = 60

# Pausa entre búsquedas si la cabecera de throttling no dice nada.
DEFAULT_DELAY = 2.0
MIN_DELAY = 0.5
MAX_DELAY = 12.0


class EPOError(Exception):
    """Fallo genérico de OPS."""


class EPOAuthError(EPOError):
    """Credenciales ausentes, mal emparejadas o rechazadas."""


class EPOQuotaError(EPOError):
    """
    Cuota agotada: 403 con `X-Rejection-Reason`.

    No se lanza por el COLOR de `X-Throttling-Control`: rojo significa que
    OPS recortó el presupuesto, no que la petición fallara — esa respuesta
    trae datos válidos. Ver `_note_throttling`.
    """


def credentials_configured() -> bool:
    return bool(
        os.getenv("EPO_CONSUMER_KEY", "").strip()
        and os.getenv("EPO_API_KEY", "").strip()
    )


# ── Construcción de la consulta CQL ──────────────────────────────────────────

# Operadores CQL de OPS que delatan que el usuario ya escribió una consulta
# estructurada y no un término suelto.
_CQL_FIELDS = (
    "txt", "ti", "ab", "ta", "ti,ab", "pa", "in", "ipc", "cpc", "pn", "ap",
    "pd", "num", "pr", "citation", "ct", "sap", "sin",
)
_CQL_RE = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in _CQL_FIELDS) + r")\s*(=|all|any|within)",
    re.IGNORECASE,
)


def build_cql(query: str, year: int | None = None) -> str:
    """
    Normaliza la consulta y le añade la ventana de un año.

    Si el texto ya parece CQL (`txt="x" and pa="y"`) se respeta tal cual; si
    es un término suelto se envuelve en `txt="…"`, que busca en título,
    resumen, solicitante e inventor.

    El filtro anual va por fecha de PUBLICACIÓN (`pd`), no de prioridad: es la
    fecha en que la patente se hace pública y por tanto la que corresponde a
    la difusión tecnológica que mide la curva.
    """
    q = query.strip()
    if not q:
        raise ValueError("La consulta de patentes no puede estar vacía")

    if not _CQL_RE.search(q):
        safe = q.replace('"', " ").strip()
        q = f'txt="{safe}"'

    if year is not None:
        q = f'{q} and pd within "{year}0101 {year}1231"'
    return q


def _parse_search_quota(header: str | None) -> int | None:
    """
    Extrae el presupuesto de búsquedas por minuto de `X-Throttling-Control`.

    Formato: `busy (images=green:100, …, search=green:15)`.
    """
    if not header:
        return None
    m = re.search(r"search\s*=\s*(\w+)\s*:\s*(\d+)", header)
    if not m:
        return None
    try:
        return int(m.group(2))
    except ValueError:
        return None


def _throttling_colour(header: str | None) -> str | None:
    if not header:
        return None
    m = re.search(r"search\s*=\s*(\w+)\s*:", header)
    return m.group(1).lower() if m else None


# ── Cliente ──────────────────────────────────────────────────────────────────


class EPOClient:
    """
    Cliente asíncrono mínimo de OPS: autenticar y contar.

    No pagina resultados porque para una curva en S no hacen falta: sólo el
    `@total-result-count` de cada año.
    """

    def __init__(
        self,
        consumer_key: str | None = None,
        consumer_secret: str | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self._key = (consumer_key or os.getenv("EPO_CONSUMER_KEY", "")).strip()
        self._secret = (consumer_secret or os.getenv("EPO_API_KEY", "")).strip()
        self._client = client
        self._owns_client = client is None
        self._token: str | None = None
        self._token_expiry: float = 0.0
        self._delay: float = DEFAULT_DELAY
        self.last_throttle: str | None = None
        self.throttled: bool = False
        self.requests_made: int = 0

    async def __aenter__(self) -> "EPOClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60)
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60)
            self._owns_client = True
        return self._client

    # ── OAuth2 ───────────────────────────────────────────────

    async def _request_token(self, key: str, secret: str) -> tuple[str | None, int, str]:
        basic = base64.b64encode(f"{key}:{secret}".encode()).decode()
        r = await self.client.post(
            OPS_AUTH_URL,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
        )
        if r.status_code == 200:
            payload = r.json()
            return payload.get("access_token"), int(payload.get("expires_in", 1200)), ""
        # El cuerpo del error puede llevar el mensaje de OPS; nunca las claves.
        return None, 0, r.text[:200]

    async def authenticate(self, force: bool = False) -> str:
        """
        Obtiene (y cachea) el bearer token.

        Prueba primero `EPO_CONSUMER_KEY:EPO_API_KEY`, que es el
        emparejamiento correcto verificado contra el servicio, y si OPS lo
        rechaza reintenta con el orden inverso antes de rendirse: cuál de las
        dos variables guarda el secreto es una convención del despliegue, no
        del protocolo.
        """
        if not self._key or not self._secret:
            raise EPOAuthError(
                "Faltan credenciales de EPO OPS: define EPO_CONSUMER_KEY y "
                "EPO_API_KEY en el entorno del backend."
            )

        now = time.monotonic()
        if not force and self._token and now < self._token_expiry:
            return self._token

        errors: list[str] = []
        for key, secret in ((self._key, self._secret), (self._secret, self._key)):
            try:
                token, expires_in, err = await self._request_token(key, secret)
            except httpx.HTTPError as exc:
                raise EPOAuthError(f"No se pudo contactar con EPO OPS: {exc}") from exc
            if token:
                self._token = token
                self._token_expiry = now + max(expires_in - TOKEN_SAFETY_MARGIN, 60)
                # Si funcionó el orden invertido, fijarlo para los refrescos.
                self._key, self._secret = key, secret
                return token
            errors.append(err)

        raise EPOAuthError(
            "EPO OPS rechazó las credenciales (revisa que EPO_CONSUMER_KEY sea "
            "la consumer key y EPO_API_KEY el consumer secret). "
            f"Respuesta: {errors[0][:120] if errors else 'sin detalle'}"
        )

    # ── Búsqueda ─────────────────────────────────────────────

    def _note_throttling(self, response: httpx.Response) -> None:
        """
        Ajusta el ritmo al presupuesto que OPS acaba de declarar.

        El color NO es un error: `red` significa que OPS está saturado y ha
        recortado el presupuesto a 5 búsquedas por minuto, pero la petición
        que trae esa cabecera SÍ se sirvió. Abortar ahí tiraría un dato ya
        pagado. Sólo un 403 con `X-Rejection-Reason` es un alto de verdad.
        """
        header = response.headers.get("X-Throttling-Control")
        if not header:
            return
        self.last_throttle = header
        colour = _throttling_colour(header)
        quota = _parse_search_quota(header)
        if quota and quota > 0:
            # 60/n segundos entre peticiones mantiene el ritmo justo por
            # debajo del presupuesto que OPS acaba de declarar.
            self._delay = min(max(60.0 / quota, MIN_DELAY), MAX_DELAY)
        if colour in ("red", "black"):
            self.throttled = True
            self._delay = MAX_DELAY

    async def count(self, cql: str, retries: int = 3) -> int:
        """
        Número de publicaciones que casan una consulta CQL.

        404 `No results found` se traduce a 0: es una respuesta válida, no un
        fallo. Tratarla como error dejaría huecos en los años iniciales de la
        serie, que son justo los que fijan el arranque de la curva.
        """
        token = await self.authenticate()

        for attempt in range(retries):
            try:
                r = await self.client.get(
                    OPS_SEARCH_URL,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                    },
                    params={"q": cql, "Range": "1-1"},
                )
            except httpx.HTTPError as exc:
                if attempt == retries - 1:
                    raise EPOError(f"Error de red consultando EPO OPS: {exc}") from exc
                await asyncio.sleep(2**attempt)
                continue

            self.requests_made += 1

            if r.status_code == 200:
                self._note_throttling(r)
                return self._extract_total(r.json())

            if r.status_code == 404:
                self._note_throttling(r)
                return 0

            if r.status_code in (401, 403):
                reason = r.headers.get("X-Rejection-Reason", "")
                if r.status_code == 401 or "invalid" in reason.lower() or not reason:
                    # Token caducado: uno y sólo un reintento con token nuevo.
                    if attempt == 0:
                        token = await self.authenticate(force=True)
                        continue
                raise EPOQuotaError(
                    f"EPO OPS rechazó la consulta (HTTP {r.status_code}"
                    + (f", {reason}" if reason else "")
                    + "). Suele ser la cuota horaria o semanal agotada."
                )

            if r.status_code == 429:
                await asyncio.sleep(min(2**attempt * 5, 30))
                continue

            if attempt == retries - 1:
                raise EPOError(
                    f"EPO OPS devolvió HTTP {r.status_code}: {r.text[:150]}"
                )
            await asyncio.sleep(2**attempt)

        raise EPOError("EPO OPS no respondió tras varios intentos.")

    @staticmethod
    def _extract_total(payload: dict) -> int:
        try:
            biblio = payload["ops:world-patent-data"]["ops:biblio-search"]
            return int(biblio["@total-result-count"])
        except (KeyError, TypeError, ValueError):
            return 0

    async def total_for_query(self, query: str) -> int:
        """Conteo sin filtro temporal — sondeo para saber si la consulta casa algo."""
        return await self.count(build_cql(query))

    # ── Serie anual ──────────────────────────────────────────

    async def yearly_counts(
        self,
        query: str,
        year_from: int,
        year_to: int,
    ) -> AsyncGenerator[dict, None]:
        """
        Generador que emite progreso y termina con la serie anual.

        Una petición por año (27 años ≈ 27 peticiones), con la pausa que dicte
        el throttling de OPS. Los eventos son los mismos que los de la ruta de
        Scopus: `{status, progress, message}` y un `completed` final con
        `counts`.
        """
        years = list(range(year_from, year_to + 1))
        counts: dict[int, int] = {}
        failed: list[int] = []

        yield {
            "status": "running",
            "progress": 4,
            "message": "Autenticando con EPO OPS…",
        }
        await self.authenticate()

        cql_preview = build_cql(query)
        yield {
            "status": "running",
            "progress": 8,
            "message": f"Consulta CQL: {cql_preview}",
        }

        for i, year in enumerate(years):
            try:
                n = await self.count(build_cql(query, year))
                counts[year] = n
            except EPOQuotaError:
                # Si no se llegó a obtener NINGÚN año, no hay curva que
                # devolver: terminar con un resultado vacío se lee como "no
                # devolvió nada" y esconde la causa real, que es la cuota.
                if not counts:
                    yield {
                        "status": "error",
                        "message": (
                            "La cuota de EPO OPS está agotada: no se pudo "
                            "obtener ningún año. El límite se restablece por "
                            "horas; consulta el estado en «Estado de EPO» y "
                            "reintenta más tarde, o acorta el rango de años."
                        ),
                    }
                    return
                # Agotada a mitad de serie: devolver lo que hay, marcado, es
                # mejor que perderlo todo.
                yield {
                    "status": "running",
                    "progress": 8 + int((i + 1) / len(years) * 78),
                    "message": (
                        f"Cuota de EPO OPS agotada en {year}. "
                        f"Se devuelven los {len(counts)} años ya obtenidos "
                        f"({years[0]}–{years[i-1] if i else years[0]}); el resto "
                        f"falta por cuota, no por ausencia de patentes."
                    ),
                }
                failed.extend(years[i:])
                break
            except EPOError as exc:
                failed.append(year)
                yield {
                    "status": "running",
                    "progress": 8 + int((i + 1) / len(years) * 78),
                    "message": f"Año {year} no disponible ({exc}). Continuando…",
                }
                continue

            yield {
                "status": "running",
                "progress": 8 + int((i + 1) / len(years) * 78),
                "message": (
                    f"EPO OPS {year}: {n:,} patentes "
                    f"({i + 1}/{len(years)} años, {sum(counts.values()):,} acumuladas)"
                ),
            }

            if i < len(years) - 1:
                await asyncio.sleep(self._delay)

        if not counts or sum(counts.values()) == 0:
            yield {
                "status": "error",
                "message": (
                    "EPO OPS no devolvió patentes para esta consulta en el "
                    "rango solicitado."
                ),
            }
            return

        yield {
            "status": "completed",
            "progress": 86,
            "message": f"{sum(counts.values()):,} patentes en {len(counts)} años.",
            "counts": counts,
            "failed_years": failed,
            "requests_made": self.requests_made,
            "throttle": self.last_throttle,
            "throttled": self.throttled,
            "cql": cql_preview,
        }


async def check_credentials() -> dict:
    """
    Diagnóstico: ¿autentica OPS y con qué presupuesto de búsqueda?

    No imprime ni devuelve ningún fragmento de las credenciales.
    """
    if not credentials_configured():
        return {
            "configured": False,
            "authenticated": False,
            "message": "EPO_CONSUMER_KEY y/o EPO_API_KEY no están definidas.",
        }

    async with EPOClient() as epo:
        try:
            await epo.authenticate()
        except EPOAuthError as exc:
            return {
                "configured": True,
                "authenticated": False,
                "message": str(exc),
            }
        try:
            total = await epo.count(build_cql("biodiesel", 2015))
        except EPOError as exc:
            return {
                "configured": True,
                "authenticated": True,
                "search_ok": False,
                "message": f"Autenticó, pero la búsqueda falló: {exc}",
            }
        return {
            "configured": True,
            "authenticated": True,
            "search_ok": True,
            "probe": {"cql": build_cql("biodiesel", 2015), "total": total},
            "throttle": epo.last_throttle,
            "search_quota_per_min": _parse_search_quota(epo.last_throttle),
            "message": "EPO OPS operativo.",
        }
