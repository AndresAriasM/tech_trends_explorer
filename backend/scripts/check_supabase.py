#!/usr/bin/env python
"""
Verificador de configuración de Supabase.

Comprueba que backend/.env está completo y que las credenciales funcionan,
SIN imprimir jamás el valor de un secreto: sólo longitud y un prefijo corto
suficiente para distinguir una clave de otra.

Uso:
    cd backend && .venv/bin/python scripts/check_supabase.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv  # noqa: E402

OK, FAIL, WARN, SKIP = "  OK  ", " FALLA", " AVISO", " OMITE"


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))


def mask(value: str) -> str:
    """Describe un secreto sin revelarlo."""
    if not value:
        return "vacío"
    head = value[:7] if len(value) > 14 else value[:3]
    return f"{head}… ({len(value)} caracteres)"


# ── 1. Carga del .env ─────────────────────────────────────────

env_path = BACKEND_DIR / ".env"
print(f"\nProspecta — verificación de Supabase\n{'=' * 52}\n")

if not env_path.exists():
    line(FAIL, ".env", f"no existe en {env_path}")
    print("\n  Ejecuta:  cp backend/.env.example backend/.env\n")
    sys.exit(1)

load_dotenv(env_path)
line(OK, ".env", f"cargado desde {env_path.relative_to(BACKEND_DIR.parent)}")

# ── 2. Presencia de variables ─────────────────────────────────

print()
REQUIRED = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "DATABASE_URL"]
OPTIONAL = ["SUPABASE_ANON_KEY", "OPENAI_API_KEY", "SERP_API_KEY", "SCOPUS_API_KEY"]

missing: list[str] = []
for name in REQUIRED:
    value = os.getenv(name, "").strip()
    if value:
        line(OK, name, mask(value) if "KEY" in name or "URL" in name and "://" not in value else "definida")
    else:
        line(FAIL, name, "sin definir")
        missing.append(name)

for name in OPTIONAL:
    value = os.getenv(name, "").strip()
    line(OK if value else WARN, name, mask(value) if value else "sin definir (opcional por ahora)")

# ── 2b. Las claves de Supabase tienen forma reconocible ───────
# Un error fácil es pegar aquí la clave de otro servicio; se detecta
# por la forma y comparando con las demás claves del .env.

print()


def looks_like_supabase_key(value: str, kind: str) -> bool:
    """kind: 'secret' o 'publishable'. Acepta el formato nuevo y el JWT legacy."""
    return value.startswith(f"sb_{kind}_") or value.startswith("eyJ")


service_key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
if looks_like_supabase_key(service_key, "secret"):
    line(OK, "Forma de SUPABASE_SERVICE_KEY", "clave de servidor válida")
else:
    line(
        FAIL,
        "Forma de SUPABASE_SERVICE_KEY",
        "no parece de Supabase (se espera sb_secret_… o un JWT eyJ…)",
    )
    # ¿Es en realidad otra clave del mismo archivo?
    for other in ("SERP_API_KEY", "SCOPUS_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        if service_key and service_key == os.getenv(other, "").strip():
            line(FAIL, "Clave duplicada", f"su valor es idéntico al de {other}")
    print(
        "\n  Dashboard → Project Settings → API Keys → «secret» (o «service_role»).\n"
    )
    missing.append("SUPABASE_SERVICE_KEY (forma incorrecta)")

anon_key = os.getenv("SUPABASE_ANON_KEY", "").strip()
if anon_key and not looks_like_supabase_key(anon_key, "publishable"):
    line(WARN, "Forma de SUPABASE_ANON_KEY", "no parece la clave pública de Supabase")

if missing:
    print(f"\n  Corrige {len(missing)} problema(s) en backend/.env:")
    for m in missing:
        print(f"    · {m}")
    print()
    sys.exit(1)

# ── 3. Forma de los valores ───────────────────────────────────

print()
supabase_url = os.getenv("SUPABASE_URL", "").strip()
_u = urlparse(supabase_url)
if _u.scheme == "https" and "supabase" in _u.netloc:
    if _u.path not in ("", "/"):
        # Copiar la URL del endpoint REST en vez del origen es un error
        # silencioso: todas las llamadas responden PGRST125.
        line(
            FAIL,
            "Forma de SUPABASE_URL",
            f"sobra el path «{_u.path}» — debe ser sólo https://{_u.netloc}",
        )
        sys.exit(1)
    line(OK, "Forma de SUPABASE_URL", _u.netloc)
else:
    line(WARN, "Forma de SUPABASE_URL", "se esperaba https://<ref>.supabase.co")

db_url = os.getenv("DATABASE_URL", "").strip()
parsed = urlparse(db_url)
if parsed.scheme.startswith("postgres"):
    pooled = "pooler" in (parsed.hostname or "")
    line(
        OK,
        "Forma de DATABASE_URL",
        f"{parsed.hostname}:{parsed.port or 5432} "
        f"({'pooler' if pooled else 'conexión directa'})",
    )
    if "[YOUR-PASSWORD]" in db_url or "[PASSWORD]" in db_url:
        line(FAIL, "Contraseña", "DATABASE_URL aún tiene el marcador [YOUR-PASSWORD]")
        sys.exit(1)
else:
    line(FAIL, "Forma de DATABASE_URL", "se esperaba postgresql://…")
    sys.exit(1)

# ── 4. Conexión Postgres ──────────────────────────────────────

print()
try:
    import psycopg
except ImportError:
    line(SKIP, "Postgres", "psycopg no instalado — uv pip install 'psycopg[binary]'")
else:
    try:
        with psycopg.connect(db_url, connect_timeout=12) as conn:
            with conn.cursor() as cur:
                cur.execute("select current_database(), current_user, version()")
                db, user, version = cur.fetchone()
                line(OK, "Conexión Postgres", f"base «{db}» como «{user}»")
                line(OK, "Servidor", version.split(" on ")[0])

                cur.execute(
                    "select table_name from information_schema.tables "
                    "where table_schema = 'public' order by table_name"
                )
                tables = [r[0] for r in cur.fetchall()]
                if tables:
                    line(OK, f"Tablas en public ({len(tables)})", ", ".join(tables[:8]))
                else:
                    line(WARN, "Tablas en public", "ninguna todavía — falta crear el esquema")
    except Exception as exc:  # noqa: BLE001
        line(FAIL, "Conexión Postgres", f"{type(exc).__name__}: {exc}")
        print(
            "\n  Revisa: contraseña correcta, y que uses la cadena del pooler\n"
            "  si tu red no tiene IPv6 (la conexión directa es IPv6).\n"
        )
        sys.exit(1)

# ── 5. API de Supabase ────────────────────────────────────────

print()
try:
    from supabase import create_client
except ImportError:
    line(SKIP, "API Supabase", "SDK no instalado — uv pip install supabase")
else:
    try:
        client = create_client(supabase_url, os.getenv("SUPABASE_SERVICE_KEY", ""))
        # storage.list_buckets() es la llamada autenticada más barata que
        # no depende de que exista ninguna tabla todavía.
        buckets = client.storage.list_buckets()
        line(OK, "API Supabase", f"autenticada ({len(buckets)} buckets de storage)")
    except Exception as exc:  # noqa: BLE001
        line(FAIL, "API Supabase", f"{type(exc).__name__}: {exc}")
        print("\n  Suele ser la clave equivocada: hace falta la secret/service_role.\n")
        sys.exit(1)

print(f"\n{'=' * 52}\nTodo correcto. Supabase está listo.\n")
