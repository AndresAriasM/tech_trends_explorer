#!/usr/bin/env python
"""
backend/scripts/import_delphi_historico.py

Importador **de un solo uso** del ejercicio *Empleos del futuro UPB*
(2023-2024) al modelo Delphi de Prospecta. Implementa la §10 de
`docs/LOGICA_DELPHI.md`.

Es deliberadamente un script y no parte del núcleo: «la ETL del notebook
(celdas 2–12) se convierte en un importador de un solo uso, **fuera del
núcleo**». Nada de aquí se importa desde `core/` ni desde `api/`.

Las dos correcciones obligatorias de la §10, que son la razón de que este
script exista:

  1. **Normalizar los dos vocabularios de etiquetas** de la ronda 2.
     El formulario usó `"Opción 1"/"Opción 2"` en unos ítems y
     `"PRIORITARIO A 2036"/"NO PRIORITARIO A 2036"` en otros. El notebook
     sólo comparaba contra el primero, así que los del segundo grupo quedaban
     **permanentemente en blanco** y se leían como «no cambió de opinión».
     Verificado sobre `Gerente de productos digitales` (§3.2). Aquí ambos
     vocabularios colapsan al mismo valor.

  2. **`NS/NR` y vacío → `is_abstention = true`**, nunca un 6 dentro de la
     escala y nunca agrupados con el extremo bajo. En el notebook una
     abstención contaba como juicio negativo: imputación fuerte y no
     declarada.

Además, y como el documento pide:
  · el panel se cruza por **correo electrónico**, no por nombre tecleado
    (§4: `'Sandra Milena Rondón Lagos '` con espacio final dio 29
    coincidencias y no hay forma de saber cuántas se perdieron);
  · el `panelist_id` resultante es opaco (uuid), y los nombres pueden
    seudonimizarse con `--pseudonymize`;
  · los `.xlsx` **no están en el repositorio** y contienen datos personales
    de 55 personas: las rutas se pasan por argumento y nada se versiona.

──────────────────────────────────────────────────────────────────────────
FORMATO ESPERADO DE LOS .xlsx
──────────────────────────────────────────────────────────────────────────
Un export de Google Forms: una hoja ancha, una fila por respondente.

  · Una columna de identidad. Por defecto se busca, en este orden, una
    cabecera que contenga «correo», «email» o «mail»; si no hay ninguna, se
    usa `--id-column`. El cruce entre rondas se hace por ese valor
    normalizado (minúsculas, sin espacios extremos).

  · Una columna por ítem, cuya cabecera es el enunciado del rol/competencia.
    El texto de la cabecera se normaliza a un `code` estable
    (`cientifico_de_datos`) que es lo que une ronda 1 con ronda 2 y con la
    variable del estudio.

  · Columnas de metadatos que se descartan: las que empiezan por
    `Marca temporal`, `Puntuación`, `Puntos`, `Comentarios`, y las
    marcadoras de familia `Opiniones*` — salvo que se use
    `--family-markers`, en cuyo caso `Opiniones*` **define la familia** de
    todos los ítems que la preceden (§1: hoy la taxonomía está implícita en
    esos marcadores y la ETL la descarta).

  · Columnas de competencia experta, si existen: la cabecera que contenga
    «coeficiente de competencia» / «Kc» alimenta `kc`; «Ka» alimenta `ka`.
    También se recogen «Formación Académica» y «Afiliación laboral» (§5.2).

Valores admitidos en una celda de ítem:
  · un número de la escala (1–5, o «3 - Media»: se toma el primer entero);
  · `Opción 1` / `PRIORITARIO A 2036` / `Sí` / `Prioritario`  → alto;
  · `Opción 2` / `NO PRIORITARIO A 2036` / `No`               → bajo;
  · `NS/NR`, `No sabe`, `6`, vacío                             → abstención.

Los valores binarios se guardan en los extremos de la escala del estudio
(alto → `scale_max`, bajo → `scale_min`) y el import lo deja anotado: la
ronda 2 cambió de unidades y eso ya destruyó la estadística pareada (§5.3).
La matriz de la §2 sí se puede reconstruir, porque sólo necesita el lado del
umbral, y es lo que el módulo calculará.

──────────────────────────────────────────────────────────────────────────
USO
──────────────────────────────────────────────────────────────────────────
    cd backend
    # 1. Ver qué se leería, sin tocar la base (ni siquiera conecta):
    .venv/bin/python scripts/import_delphi_historico.py \
        --round-1 ~/ruta/EmpleosFuturo_R1.xlsx \
        --round-2 ~/ruta/EmpleosFuturo_R2.xlsx \
        --dry-run --report

    # 2. Ensayo contra la base, en transacción, deshaciendo al final:
    .venv/bin/python scripts/import_delphi_historico.py \
        --round-1 ... --round-2 ... --owner-id <uuid> --rollback

    # 3. Importación real:
    .venv/bin/python scripts/import_delphi_historico.py \
        --round-1 ... --round-2 ... --owner-id <uuid> --commit

Requisitos: `backend/migrations/001_delphi_prospecta.sql` aplicada, y
`openpyxl` instalado en el entorno (`uv pip install openpyxl`).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# ══════════════════════════════════════════════════════════════════════════
# 1 · Normalización de etiquetas — el bug de la §3.2
# ══════════════════════════════════════════════════════════════════════════

def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def norm(text: object) -> str:
    """Minúsculas, sin tildes, sin espacios dobles. La clave de todo cruce."""
    if text is None:
        return ""
    s = _strip_accents(str(text)).lower().strip()
    return re.sub(r"\s+", " ", s)


# Los DOS vocabularios que el formulario usó indistintamente, más los que
# aparecen sueltos. Que «Opción 1» y «PRIORITARIO A 2036» signifiquen lo
# mismo es exactamente el fallo que dejaba media ronda 2 en blanco.
HIGH_LABELS = {
    "opcion 1", "opcion1",
    "prioritario a 2036", "prioritario", "es prioritario",
    "si", "si prioritario", "alta", "alto",
}
LOW_LABELS = {
    "opcion 2", "opcion2",
    "no prioritario a 2036", "no prioritario",
    "no", "no es prioritario", "baja", "bajo",
}
ABSTENTION_LABELS = {
    "", "ns/nr", "ns / nr", "nsnr", "ns", "nr", "n/a", "na", "ninguna",
    "no sabe", "no sabe / no responde", "no sabe no responde",
    "no responde", "no aplica", "sin respuesta", "-", "--", "6",
}


@dataclass
class Cell:
    """Una celda ya interpretada. `value` es None si y sólo si hay abstención."""
    value: Optional[float]
    is_abstention: bool
    kind: str        # 'numeric' | 'binary_high' | 'binary_low' | 'abstention' | 'unparsed'
    raw: str


def parse_cell(raw: object, scale_min: float, scale_max: float) -> Cell:
    """
    Interpreta una celda de ítem. Las tres reglas de §10 viven aquí:
    los dos vocabularios colapsan, NS/NR y vacío son abstención, y el 6
    **no es un valor de la escala** sino la marca de NS/NR del notebook.
    """
    text = norm(raw)

    if text in ABSTENTION_LABELS:
        return Cell(None, True, "abstention", str(raw) if raw is not None else "")

    if text in HIGH_LABELS:
        return Cell(float(scale_max), False, "binary_high", str(raw))
    if text in LOW_LABELS:
        return Cell(float(scale_min), False, "binary_low", str(raw))

    # «3 - Media», «4 (alta)», «5»
    m = re.search(r"-?\d+(?:[.,]\d+)?", text)
    if m:
        try:
            num = float(m.group(0).replace(",", "."))
        except ValueError:
            return Cell(None, True, "unparsed", str(raw))
        # El 6 del notebook: NS/NR disfrazado de valor. Nunca entra al rango.
        if num == 6 and scale_max <= 5:
            return Cell(None, True, "abstention", str(raw))
        if num < scale_min or num > scale_max:
            return Cell(None, True, "unparsed", str(raw))
        return Cell(num, False, "numeric", str(raw))

    return Cell(None, True, "unparsed", str(raw))


# ══════════════════════════════════════════════════════════════════════════
# 2 · Lectura del .xlsx
# ══════════════════════════════════════════════════════════════════════════

# Cabeceras que Google Forms mezcla con los ítems y que el notebook podaba
# con una lista de ~200 índices literales (celda 7).
JUNK_PREFIXES = (
    "marca temporal", "timestamp", "puntuacion", "puntos", "score",
    "direccion de correo", "nombre de usuario", "id de respuesta",
)
META_HINTS = {
    "kc": ("coeficiente de competencia", "fuente conocimiento", "fuente de conocimiento", " kc"),
    "ka": ("argumentacion", "coeficiente de argumentacion", " ka"),
    "education": ("formacion academica", "nivel maximo de formacion", "nivel de formacion"),
    "affiliation": ("afiliacion laboral", "institucion", "afiliacion"),
    "country": ("pais",),
}
COMMENT_HINTS = ("comentario", "opinion", "observacion", "sugerencia")


def code_of(header: str, used: Optional[set] = None) -> str:
    """Cabecera → `code` estable y legible: «Científico de datos» → `cientifico_de_datos`."""
    base = re.sub(r"[^a-z0-9]+", "_", norm(header)).strip("_")[:90] or "item"
    if used is None:
        return base
    code, i = base, 2
    while code in used:
        code = f"{base}_{i}"
        i += 1
    used.add(code)
    return code


@dataclass
class SheetRow:
    identity: str
    items: Dict[str, Cell]              # code → celda
    meta: Dict[str, object] = field(default_factory=dict)
    comments: Dict[str, str] = field(default_factory=dict)  # code → argumento


@dataclass
class Sheet:
    path: Path
    headers: List[str]
    item_codes: List[str]
    item_names: Dict[str, str]          # code → enunciado original
    item_family: Dict[str, Optional[str]]
    rows: List[SheetRow]
    skipped_headers: List[str]
    warnings: List[str] = field(default_factory=list)


def classify_header(h: str) -> str:
    """'junk' | 'meta:<campo>' | 'comment' | 'item'"""
    n = norm(h)
    if not n:
        return "junk"
    if any(n.startswith(p) for p in JUNK_PREFIXES):
        return "junk"
    for field_name, hints in META_HINTS.items():
        if any(hint in n for hint in hints):
            return f"meta:{field_name}"
    if any(hint in n for hint in COMMENT_HINTS):
        return "comment"
    return "item"


def read_sheet(
    path: Path,
    scale_min: float,
    scale_max: float,
    id_column: Optional[str] = None,
    family_markers: bool = False,
    sheet_name: Optional[str] = None,
) -> Sheet:
    try:
        from openpyxl import load_workbook
    except ImportError:  # pragma: no cover
        sys.exit("Falta openpyxl:  uv pip install --python .venv/bin/python openpyxl")

    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb[wb.sheetnames[0]]
    it = ws.iter_rows(values_only=True)
    try:
        headers_raw = next(it)
    except StopIteration:
        sys.exit(f"{path.name}: la hoja está vacía")
    headers = [("" if h is None else str(h).strip()) for h in headers_raw]

    # ── columna de identidad ────────────────────────────────────────────
    id_idx: Optional[int] = None
    if id_column:
        for i, h in enumerate(headers):
            if norm(h) == norm(id_column):
                id_idx = i
                break
        if id_idx is None:
            sys.exit(f"{path.name}: no existe la columna de identidad «{id_column}»")
    else:
        for key in ("correo", "email", "e-mail", "mail"):
            for i, h in enumerate(headers):
                if key in norm(h):
                    id_idx = i
                    break
            if id_idx is not None:
                break
    if id_idx is None:
        sys.exit(
            f"{path.name}: no se encontró columna de correo. Pásala con --id-column.\n"
            "  (Cruzar por nombre tecleado es justo lo que la §4 prohíbe.)"
        )

    # ── clasificar el resto ─────────────────────────────────────────────
    used: set = set()
    item_codes: List[str] = []
    item_names: Dict[str, str] = {}
    item_family: Dict[str, Optional[str]] = {}
    idx_of_code: Dict[str, int] = {}
    meta_idx: Dict[str, int] = {}
    comment_idx: List[Tuple[int, str]] = []
    skipped: List[str] = []

    pending_family: List[str] = []   # ítems aún sin familia asignada
    warnings: List[str] = []

    for i, h in enumerate(headers):
        if i == id_idx:
            continue
        kind = classify_header(h)
        if kind == "junk":
            skipped.append(h)
            continue
        if kind.startswith("meta:"):
            meta_idx[kind.split(":", 1)[1]] = i
            continue
        if kind == "comment":
            # Marcador de familia Opiniones* (§1): cierra la familia anterior.
            if family_markers and norm(h).startswith("opiniones"):
                fam = h.strip()
                for code in pending_family:
                    item_family[code] = fam
                pending_family = []
            comment_idx.append((i, h))
            continue
        code = code_of(h, used)
        item_codes.append(code)
        item_names[code] = h
        item_family[code] = None
        idx_of_code[code] = i
        pending_family.append(code)

    if family_markers and pending_family:
        warnings.append(
            f"{len(pending_family)} ítems quedaron sin marcador Opiniones* detrás: sin familia."
        )

    # ── filas ───────────────────────────────────────────────────────────
    rows: List[SheetRow] = []
    seen_identities: Counter = Counter()
    for raw_row in it:
        if raw_row is None or all(v is None or str(v).strip() == "" for v in raw_row):
            continue
        ident = norm(raw_row[id_idx] if id_idx < len(raw_row) else "")
        if not ident:
            warnings.append("Una fila sin correo se descarta (no es cruzable entre rondas).")
            continue
        seen_identities[ident] += 1

        items = {
            code: parse_cell(
                raw_row[idx] if idx < len(raw_row) else None, scale_min, scale_max
            )
            for code, idx in idx_of_code.items()
        }
        meta = {
            k: (raw_row[i] if i < len(raw_row) else None)
            for k, i in meta_idx.items()
        }
        comments = {}
        for i, h in comment_idx:
            v = raw_row[i] if i < len(raw_row) else None
            if v is not None and str(v).strip():
                comments[code_of(h)] = str(v).strip()
        rows.append(SheetRow(ident, items, meta, comments))

    for ident, n in seen_identities.items():
        if n > 1:
            warnings.append(f"Correo repetido ({n} veces): se conservan todas las filas.")

    return Sheet(path, headers, item_codes, item_names, item_family,
                 rows, skipped, warnings)


# ══════════════════════════════════════════════════════════════════════════
# 3 · Competencia experta K (§5.2)
# ══════════════════════════════════════════════════════════════════════════

# «Soy especialista y tengo bastante conocimiento…» /
# «No soy especialista y tengo algún conocimiento…»
KC_SCALE: List[Tuple[Tuple[str, ...], float]] = [
    (("muy alto", "amplio conocimiento", "bastante conocimiento", "soy especialista"), 0.9),
    (("alto", "buen conocimiento"), 0.7),
    (("medio", "algun conocimiento", "algún conocimiento"), 0.5),
    (("bajo", "poco conocimiento"), 0.3),
    (("muy bajo", "ningun conocimiento", "no soy especialista"), 0.1),
]


def parse_competence(raw: object) -> Optional[float]:
    """Texto del instrumento → Kc/Ka en [0,1]. None si no se reconoce."""
    if raw is None:
        return None
    text = norm(raw)
    if not text:
        return None
    m = re.fullmatch(r"(\d+(?:[.,]\d+)?)", text)
    if m:
        v = float(m.group(1).replace(",", "."))
        if v <= 1:
            return round(v, 2)
        if v <= 10:
            return round(v / 10.0, 2)
        return None
    # «No soy especialista» debe ganar a «soy especialista»: se comprueba
    # primero la negación, recorriendo la escala de arriba abajo con cuidado.
    if "no soy especialista" in text:
        return 0.1
    for hints, score in KC_SCALE:
        if any(h in text for h in hints):
            return score
    return None


# ══════════════════════════════════════════════════════════════════════════
# 4 · Cruce de rondas y construcción del plan de importación
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Plan:
    variables: List[Tuple[str, str, Optional[str]]]     # code, name, family
    panelists: Dict[str, Dict]                          # correo → {email, kc, ka, ...}
    responses: List[Dict]                               # filas a insertar
    stats: Dict


def build_plan(sheets: Dict[int, Sheet]) -> Plan:
    """Cruza las rondas por correo y produce las `DelphiResponse` de la §10."""
    all_codes: Dict[str, Tuple[str, Optional[str]]] = {}
    for rn in sorted(sheets):
        sh = sheets[rn]
        for code in sh.item_codes:
            if code not in all_codes:
                all_codes[code] = (sh.item_names[code], sh.item_family.get(code))
            elif all_codes[code][1] is None and sh.item_family.get(code):
                all_codes[code] = (all_codes[code][0], sh.item_family[code])

    panelists: Dict[str, Dict] = {}
    responses: List[Dict] = []
    kinds: Counter = Counter()
    per_round: Dict[int, Dict] = {}

    for rn in sorted(sheets):
        sh = sheets[rn]
        r_kinds: Counter = Counter()
        for row in sh.rows:
            p = panelists.setdefault(row.identity, {
                "email": row.identity, "kc": None, "ka": None,
                "education": None, "affiliation": None, "rounds": set(),
            })
            p["rounds"].add(rn)
            # La ronda 1 es la que lleva las preguntas de competencia.
            if p["kc"] is None:
                p["kc"] = parse_competence(row.meta.get("kc"))
            if p["ka"] is None:
                p["ka"] = parse_competence(row.meta.get("ka"))
            for key in ("education", "affiliation"):
                if p[key] is None and row.meta.get(key):
                    p[key] = str(row.meta[key]).strip()

            for code, cell in row.items.items():
                kinds[cell.kind] += 1
                r_kinds[cell.kind] += 1
                responses.append({
                    "round_number": rn,
                    "email": row.identity,
                    "code": code,
                    "value": cell.value,
                    "is_abstention": cell.is_abstention,
                    # Los argumentos se recogen por familia en el formulario
                    # (columnas Opiniones*) y aquí se adjuntan a cada ítem de
                    # esa familia: es la materia prima de la devolución §5.1.
                    "argument": _argument_for(row, code, all_codes),
                })
        per_round[rn] = {
            "n_respondents": len({r.identity for r in sh.rows}),
            "n_items": len(sh.item_codes),
            "cells": dict(r_kinds),
        }

    rounds_seen = sorted(sheets)
    intersect = None
    if len(rounds_seen) >= 2:
        a = {e for e, p in panelists.items() if rounds_seen[0] in p["rounds"]}
        b = {e for e, p in panelists.items() if rounds_seen[1] in p["rounds"]}
        intersect = len(a & b)

    codes_by_round = {rn: set(sheets[rn].item_codes) for rn in rounds_seen}
    comparable = set.intersection(*codes_by_round.values()) if codes_by_round else set()

    return Plan(
        variables=[(c, n, f) for c, (n, f) in all_codes.items()],
        panelists=panelists,
        responses=responses,
        stats={
            "rounds": per_round,
            "n_panelists_total": len(panelists),
            "n_panelists_intersection": intersect,
            "n_variables": len(all_codes),
            "n_variables_comparables": len(comparable),
            "cells_by_kind": dict(kinds),
            "n_with_k": sum(1 for p in panelists.values() if p["kc"] is not None),
        },
    )


def _argument_for(row: SheetRow, code: str, all_codes: Dict) -> Optional[str]:
    """
    El argumento asociado al ítem. El formulario sólo lo recoge por familia,
    así que sin marcadores de familia no hay a qué ítem atribuirlo y se deja
    en None antes que inventar una atribución.
    """
    family = all_codes.get(code, (None, None))[1]
    if not family or not row.comments:
        return None
    key = code_of(family)
    return row.comments.get(key)


# ══════════════════════════════════════════════════════════════════════════
# 5 · Escritura
# ══════════════════════════════════════════════════════════════════════════

def write_plan(conn, plan: Plan, args) -> Dict:
    """Inserta todo. El llamador decide si hace COMMIT o ROLLBACK."""
    from psycopg.types.json import Json

    cur = conn.cursor()

    def one(sql, params=()):
        cur.execute(sql, params or None)
        return cur.fetchone()

    owner = args.owner_id
    if not owner:
        cur.execute("select id from profiles limit 2")
        rows = cur.fetchall()
        if len(rows) != 1:
            sys.exit("Hacen falta --owner-id: hay 0 o varios perfiles en la base.")
        owner = rows[0][0]

    study = one("""
        insert into delphi_studies
            (owner_id, title, description, status, horizon_year, scale_type,
             dichotomy_threshold, max_rounds, stopping_rule)
        values (%s, %s, %s, 'closed', %s, %s, %s, %s, %s)
        returning id
    """, (owner, args.title, args.description, args.horizon,
          args.scale_type, args.threshold, max(2, len(plan.stats["rounds"])),
          Json({"stability_min": 0.85, "iqr_max": 1.0})))
    sid = study[0]

    var_id: Dict[str, object] = {}
    for i, (code, name, family) in enumerate(plan.variables):
        row = one("""
            insert into delphi_variables (study_id, code, name, family, order_index)
            values (%s, %s, %s, %s, %s) returning id
        """, (sid, code, name[:300], family, i))
        var_id[code] = row[0]

    part_id: Dict[str, object] = {}
    for i, (email, p) in enumerate(sorted(plan.panelists.items())):
        display = None if args.pseudonymize else email
        stored_email = f"panelista{i + 1:03d}@importado.local" if args.pseudonymize else email
        row = one("""
            insert into delphi_participants
                (study_id, email, display_name, status, kc, ka, education, affiliation,
                 registered_at)
            values (%s, %s, %s, 'active', %s, %s, %s, %s, now())
            returning id
        """, (sid, stored_email, display, p["kc"], p["ka"],
              p["education"], p["affiliation"]))
        part_id[email] = row[0]

    round_id: Dict[int, object] = {}
    question_id: Dict[Tuple[int, str], object] = {}
    codes_per_round: Dict[int, List[str]] = defaultdict(list)
    for r in plan.responses:
        if r["code"] not in codes_per_round[r["round_number"]]:
            codes_per_round[r["round_number"]].append(r["code"])

    for rn in sorted(codes_per_round):
        row = one("""
            insert into delphi_rounds (study_id, round_number, title, status, closes_at)
            values (%s, %s, %s, 'closed', now()) returning id
        """, (sid, rn, f"Ronda {rn} (importada)"))
        round_id[rn] = row[0]
        for i, code in enumerate(codes_per_round[rn]):
            q = one("""
                insert into delphi_questions (round_id, order_index, type, title, variable_id)
                values (%s, %s, %s::question_type, %s, %s) returning id
            """, (row[0], i, args.question_type,
                  next(n for c, n, _ in plan.variables if c == code)[:300],
                  var_id[code]))
            question_id[(rn, code)] = q[0]

    sub_id: Dict[Tuple[int, str], object] = {}
    n_answers = 0
    for r in plan.responses:
        key = (r["round_number"], r["email"])
        if key not in sub_id:
            row = one("""
                insert into delphi_submissions
                    (participant_id, round_id, status, submitted_at)
                values (%s, %s, 'submitted', now()) returning id
            """, (part_id[r["email"]], round_id[r["round_number"]]))
            sub_id[key] = row[0]
        cur.execute("""
            insert into delphi_answers
                (submission_id, question_id, value, is_abstention, argument)
            values (%s, %s, %s, %s, %s)
            on conflict (submission_id, question_id) do nothing
        """, (sub_id[key], question_id[(r["round_number"], r["code"])],
              None if r["is_abstention"] else Json(float(r["value"])),
              r["is_abstention"], r["argument"]))
        n_answers += 1

    return {
        "study_id": str(sid),
        "n_variables": len(var_id),
        "n_panelists": len(part_id),
        "n_rounds": len(round_id),
        "n_submissions": len(sub_id),
        "n_answers": n_answers,
    }


# ══════════════════════════════════════════════════════════════════════════
# 6 · Informe
# ══════════════════════════════════════════════════════════════════════════

def print_report(sheets: Dict[int, Sheet], plan: Plan, args) -> None:
    print("\n" + "=" * 68)
    print("IMPORTADOR HISTÓRICO DELPHI — lectura de los .xlsx")
    print("=" * 68)

    for rn in sorted(sheets):
        sh = sheets[rn]
        st = plan.stats["rounds"][rn]
        print(f"\nRonda {rn} · {sh.path.name}")
        print(f"  respondentes .......... {st['n_respondents']}")
        print(f"  ítems detectados ...... {st['n_items']}")
        print(f"  columnas descartadas .. {len(sh.skipped_headers)}")
        cells = st["cells"]
        total = sum(cells.values()) or 1
        for kind in ("numeric", "binary_high", "binary_low", "abstention", "unparsed"):
            n = cells.get(kind, 0)
            if n:
                print(f"    {kind:14s} {n:6d}  ({100*n/total:5.1f}%)")
        for w in dict.fromkeys(sh.warnings):
            print(f"  AVISO: {w}")

    s = plan.stats
    print("\nCruce del panel (por correo, nunca por nombre — §4)")
    print(f"  panelistas distintos ........ {s['n_panelists_total']}")
    print(f"  presentes en ambas rondas ... {s['n_panelists_intersection']}"
          "   ← población de cambio (§6)")
    print(f"  con Kc reconocido ........... {s['n_with_k']}")

    print("\nVariables")
    print(f"  totales ..................... {s['n_variables']}")
    print(f"  comparables entre rondas .... {s['n_variables_comparables']}")

    hi = s["cells_by_kind"].get("binary_high", 0)
    lo = s["cells_by_kind"].get("binary_low", 0)
    ab = s["cells_by_kind"].get("abstention", 0)
    up = s["cells_by_kind"].get("unparsed", 0)
    print("\nLas dos correcciones de la §10")
    print(f"  · Vocabularios normalizados: {hi + lo} celdas binarias colapsadas")
    print("    («Opción 1» ≡ «PRIORITARIO A 2036»). En el notebook, las del")
    print("    segundo vocabulario quedaban permanentemente en blanco.")
    print(f"  · NS/NR y vacío → is_abstention: {ab} celdas.")
    print("    Ninguna se guarda como 6 ni se agrupa con el extremo bajo.")
    if up:
        print(f"  · {up} celdas no reconocidas: tratadas como abstención y")
        print("    listadas abajo para revisión manual.")
        muestras = Counter()
        for rn in sorted(sheets):
            for row in sheets[rn].rows:
                for cell in row.items.values():
                    if cell.kind == "unparsed":
                        muestras[cell.raw[:60]] += 1
        for raw, n in muestras.most_common(10):
            print(f"      {n:5d} × {raw!r}")

    print(f"\nRespuestas a insertar: {len(plan.responses)}")
    print("=" * 68 + "\n")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Importa el Delphi histórico 2023-2024 (§10 de LOGICA_DELPHI.md).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Los .xlsx no están en el repositorio: pasa las rutas por argumento.",
    )
    ap.add_argument("--round-1", dest="r1", required=True, type=Path,
                    help="xlsx de la ronda 1 (Likert 1-5)")
    ap.add_argument("--round-2", dest="r2", type=Path,
                    help="xlsx de la ronda 2 (binaria prioritario / no)")
    ap.add_argument("--extra-round", dest="extra", action="append", default=[],
                    metavar="N:RUTA", help="rondas adicionales, p. ej. 3:/ruta/r3.xlsx")
    ap.add_argument("--sheet", default=None, help="nombre de la hoja (por defecto, la primera)")
    ap.add_argument("--id-column", default=None,
                    help="cabecera de la columna de correo, si no se detecta sola")
    ap.add_argument("--family-markers", action="store_true",
                    help="usar las columnas Opiniones* como marcador de familia (§1)")

    ap.add_argument("--title", default="Empleos del futuro UPB — horizonte 2036")
    ap.add_argument("--description",
                    default="Importado del ejercicio 2023-2024 ejecutado con Google Forms.")
    ap.add_argument("--horizon", type=int, default=2036)
    ap.add_argument("--scale-type", default="likert_5")
    ap.add_argument("--scale-min", type=float, default=1.0)
    ap.add_argument("--scale-max", type=float, default=5.0)
    ap.add_argument("--threshold", type=int, default=4,
                    help="umbral de dicotomía (§2: era un literal enterrado)")
    ap.add_argument("--question-type", default="likert_5")
    ap.add_argument("--owner-id", default=None, help="uuid del perfil dueño del estudio")
    ap.add_argument("--pseudonymize", action="store_true",
                    help="no guardar correos ni nombres reales (§10, datos personales)")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="sólo leer y reportar; no conecta a la base (por defecto)")
    mode.add_argument("--rollback", action="store_true",
                      help="escribir en una transacción y deshacerla (ensayo real)")
    mode.add_argument("--commit", action="store_true", help="escribir de verdad")
    ap.add_argument("--report", action="store_true", default=True)

    args = ap.parse_args(argv)
    if not (args.rollback or args.commit):
        args.dry_run = True

    paths: Dict[int, Path] = {1: args.r1}
    if args.r2:
        paths[2] = args.r2
    for spec in args.extra:
        n, _, p = spec.partition(":")
        if not p:
            sys.exit(f"--extra-round mal formado: {spec!r} (se espera N:RUTA)")
        paths[int(n)] = Path(p)

    for p in paths.values():
        if not p.exists():
            sys.exit(f"No existe el archivo: {p}")

    sheets = {
        rn: read_sheet(p, args.scale_min, args.scale_max,
                       id_column=args.id_column,
                       family_markers=args.family_markers,
                       sheet_name=args.sheet)
        for rn, p in sorted(paths.items())
    }
    plan = build_plan(sheets)

    if args.report:
        print_report(sheets, plan, args)

    if args.dry_run:
        print("--dry-run: no se ha tocado la base. Usa --rollback para ensayar "
              "contra ella o --commit para importar.\n")
        return 0

    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
    dsn = (os.getenv("DATABASE_URL") or "").strip()
    if not dsn:
        sys.exit("DATABASE_URL no configurada (ver backend/.env.example).")

    import psycopg
    with psycopg.connect(dsn, autocommit=False, connect_timeout=20) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from information_schema.tables "
                "where table_schema='public' and table_name='delphi_variables'")
            if cur.fetchone()[0] == 0:
                sys.exit("Falta aplicar migrations/001_delphi_prospecta.sql.")

        result = write_plan(conn, plan, args)
        if args.commit:
            conn.commit()
            print("COMMIT.", result)
            print("\nPrueba de aceptación (§10): abre GET /api/delphi/studies/"
                  f"{result['study_id']}/transitions y cuenta cuántas de las celdas "
                  "que el notebook dejaba en blanco eran cambios reales de opinión.\n")
        else:
            conn.rollback()
            print("ROLLBACK (ensayo).", result)
            print("La base queda como estaba. Repite con --commit para importar.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
