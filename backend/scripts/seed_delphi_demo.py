#!/usr/bin/env python
"""
Estudio Delphi de demostración, con dos rondas ya respondidas.

Siembra por la API pública, no por SQL: así el estudio que queda es
indistinguible de uno creado desde la interfaz, y de paso se ejercita el
camino completo de endpoints.

Los datos imitan el ejercicio real de Empleos del Futuro (UPB): panel
anidado con deserción entre rondas, abstenciones declaradas y variables con
comportamientos distintos, para que las métricas de docs/LOGICA_DELPHI.md
tengan algo que mostrar:

  · una variable que CIERRA          — consenso alto y estable
  · una que CONVERGE                 — el panel se mueve hacia el acuerdo
  · una en DISENSO PERSISTENTE       — desacuerdo firme, no lo arregla otra ronda
  · una INESTABLE                    — mucho movimiento sin dirección

Uso:
    cd backend && .venv/bin/python scripts/seed_delphi_demo.py
    .venv/bin/python scripts/seed_delphi_demo.py --borrar   # elimina el demo
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import httpx

API = "http://localhost:8000/api/delphi"
TITULO = "Empleos del futuro en tecnología — Demo 2036"

# ── Panel ────────────────────────────────────────────────────────────────
# 14 personas en la ronda 1; sólo 9 siguen en la 2 (deserción del 36 %),
# que es el patrón normal y hay que poder verlo en las métricas.
PANEL = [
    ("ana.rivera@upb.edu.co",      "Ana Rivera",        0.9, 0.8, "Doctorado",      "UPB"),
    ("carlos.mejia@upb.edu.co",    "Carlos Mejía",      0.8, 0.7, "Maestría",       "UPB"),
    ("diana.torres@unal.edu.co",   "Diana Torres",      0.9, 0.9, "Doctorado",      "UNAL"),
    ("esteban.gil@eafit.edu.co",   "Esteban Gil",       0.7, 0.6, "Maestría",       "EAFIT"),
    ("felipe.ramos@upb.edu.co",    "Felipe Ramos",      0.6, 0.5, "Especialización","UPB"),
    ("gloria.sanchez@udea.edu.co", "Gloria Sánchez",    0.8, 0.8, "Doctorado",      "UdeA"),
    ("hector.lopez@itm.edu.co",    "Héctor López",      0.5, 0.6, "Maestría",       "ITM"),
    ("irene.castro@upb.edu.co",    "Irene Castro",      0.9, 0.7, "Doctorado",      "UPB"),
    ("javier.ortiz@eafit.edu.co",  "Javier Ortiz",      0.7, 0.7, "Maestría",       "EAFIT"),
    ("karen.duque@unal.edu.co",    "Karen Duque",       0.6, 0.5, "Especialización","UNAL"),
    ("luis.penagos@upb.edu.co",    "Luis Penagos",      0.8, 0.6, "Maestría",       "UPB"),
    ("marta.velez@udea.edu.co",    "Marta Vélez",       0.9, 0.8, "Doctorado",      "UdeA"),
    ("nestor.arias@itm.edu.co",    "Néstor Arias",      0.5, 0.4, "Pregrado",       "ITM"),
    ("olga.betancur@upb.edu.co",   "Olga Betancur",     0.7, 0.8, "Maestría",       "UPB"),
]
SIGUEN_EN_R2 = 9  # los 9 primeros

VARIABLES = [
    ("cientifico_datos",  "Científico de datos",              "Datos e IA"),
    ("ing_datos",         "Ingeniero de datos",               "Datos e IA"),
    ("esp_ia_generativa", "Especialista en IA generativa",    "Datos e IA"),
    ("ing_plataformas",   "Ingeniero de plataformas",         "Nube y plataformas"),
    ("ing_devops",        "Ingeniero DevOps",                 "Nube y plataformas"),
    ("sre",               "Ingeniero de confiabilidad (SRE)", "Nube y plataformas"),
    ("analista_ciber",    "Analista de ciberseguridad",       "Seguridad"),
    ("auditor_algoritmos","Auditor de algoritmos",            "Gobernanza"),
]

# Respuestas por variable: (ronda 1 para los 14, ronda 2 para los 9 primeros).
# None = abstención declarada, nunca un número dentro de la escala.
RESPUESTAS: dict[str, tuple[list[Any], list[Any]]] = {
    # CIERRA: acuerdo alto desde el principio y se mantiene
    "cientifico_datos":   ([5,5,5,4,5,5,4,5,5,4,5,5,4,5],  [5,5,5,5,5,5,4,5,5]),
    # CONVERGE: el panel estaba dividido y se mueve hacia el acuerdo
    "esp_ia_generativa":  ([3,5,2,4,3,5,2,4,3,5,2,4,3,5],  [5,5,4,5,4,5,4,5,4]),
    # DISENSO PERSISTENTE: firme desacuerdo, cada uno en su sitio
    "auditor_algoritmos": ([5,1,5,1,5,1,5,1,5,1,5,1,5,1],  [5,1,5,1,5,1,5,1,5]),
    # INESTABLE: mucho movimiento y sin dirección clara
    "sre":                ([2,4,3,5,1,4,2,5,3,1,4,2,5,3],  [5,1,5,2,4,1,5,2,1]),
    # Resto: comportamientos intermedios, con alguna abstención
    "ing_datos":          ([4,4,5,3,4,5,4,4,3,4,5,4,None,4], [4,5,5,4,4,5,4,4,4]),
    "ing_plataformas":    ([4,3,4,4,3,4,None,4,3,4,4,3,4,4], [4,4,4,4,3,4,4,4,3]),
    "ing_devops":         ([4,4,3,4,4,3,4,None,4,3,4,4,3,4], [4,4,4,4,4,3,4,4,4]),
    "analista_ciber":     ([5,4,5,5,4,5,4,5,5,4,5,4,5,5],  [5,5,5,4,5,5,5,4,5]),
}

ARGUMENTOS = {
    "esp_ia_generativa": "La demanda crece más rápido de lo que forman los programas actuales.",
    "auditor_algoritmos": "Depende por completo de que la regulación llegue a la región.",
    "sre": "El rol existe, pero en la región se confunde con DevOps.",
}


def api(cli: httpx.Client, metodo: str, ruta: str, **kw) -> Any:
    r = cli.request(metodo, f"{API}{ruta}", timeout=60, **kw)
    if r.status_code >= 400:
        raise SystemExit(f"  ✗ {metodo} {ruta} → {r.status_code}: {r.text[:300]}")
    return r.json() if r.content else None


def borrar_demo(cli: httpx.Client) -> int:
    n = 0
    for est in api(cli, "GET", "/studies") or []:
        if est.get("title") == TITULO:
            api(cli, "DELETE", f"/studies/{est['id']}")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--borrar", action="store_true", help="elimina el estudio demo y sale")
    args = ap.parse_args()

    with httpx.Client() as cli:
        estado = api(cli, "GET", "/schema-check")
        if not estado.get("ready"):
            raise SystemExit(f"El esquema no está aplicado: falta {estado.get('missing')}")

        borrados = borrar_demo(cli)
        if borrados:
            print(f"  · {borrados} estudio(s) demo previo(s) eliminado(s)")
        if args.borrar:
            print("Listo." if borrados else "No había nada que borrar.")
            return

        # ── Estudio ──────────────────────────────────────────
        est = api(cli, "POST", "/studies", json={
            "title": TITULO,
            "description": (
                "Panel de expertos sobre los empleos de tecnología con mayor "
                "demanda proyectada a 2036 en América Latina. Dos rondas, "
                "escala Likert 1–5, umbral de priorización en 4."
            ),
            "horizon_year": 2036,
            "scale_type": "likert_5",
            "dichotomy_threshold": 4,
            "max_rounds": 3,
        })
        sid = est["id"]
        print(f"  ✓ estudio creado: {sid}")

        # ── Variables ────────────────────────────────────────
        # El endpoint recibe la lista completa de una vez.
        creadas = api(cli, "POST", f"/studies/{sid}/variables", json=[
            {"code": code, "name": nombre, "family": familia, "order_index": i}
            for i, (code, nombre, familia) in enumerate(VARIABLES)
        ])
        vids = {v["code"]: v["id"] for v in creadas}
        print(f"  ✓ {len(vids)} variables en {len({f for _,_,f in VARIABLES})} familias")

        # ── Panel ────────────────────────────────────────────
        invitados = api(cli, "POST", f"/studies/{sid}/participants", json=[
            {"email": email, "display_name": nombre, "kc": kc, "ka": ka,
             "education": edu, "affiliation": afi}
            for email, nombre, kc, ka, edu, afi in PANEL
        ])
        # Se conserva el orden del panel para que la deserción sea reproducible.
        por_email = {p["email"]: p["invitation_token"] for p in invitados}
        tokens = [por_email[email] for email, *_ in PANEL]
        print(f"  ✓ {len(tokens)} panelistas invitados")

        # ── Rondas ───────────────────────────────────────────
        for numero, (titulo, instr) in enumerate([
            ("Ronda 1 · Valoración inicial",
             "Valora de 1 a 5 la relevancia de cada rol al horizonte 2036."),
            ("Ronda 2 · Re-valoración",
             "Revisa tu respuesta anterior y la del panel antes de decidir."),
        ], start=1):
            r = api(cli, "POST", f"/studies/{sid}/rounds", json={
                "round_number": numero, "title": titulo,
                "instructions": instr, "question_type": "likert_5",
            })
            api(cli, "POST", f"/rounds/{r['id']}/open")

            n_panel = len(PANEL) if numero == 1 else SIGUEN_EN_R2
            for idx in range(n_panel):
                respuestas = []
                for code, (r1, r2) in RESPUESTAS.items():
                    serie = r1 if numero == 1 else r2
                    if idx >= len(serie):
                        continue
                    val = serie[idx]
                    item: dict[str, Any] = {"variable_id": vids[code]}
                    if val is None:
                        item["is_abstention"] = True
                    else:
                        item["value"] = float(val)
                    # Un argumento por variable, del primer panelista que la toca.
                    if idx == 0 and code in ARGUMENTOS:
                        item["argument"] = ARGUMENTOS[code]
                    respuestas.append(item)
                api(cli, "POST", f"/invite/{tokens[idx]}/responses",
                    json={"answers": respuestas, "submit": True})

            api(cli, "POST", f"/rounds/{r['id']}/close")
            print(f"  ✓ {titulo}: {n_panel} entregas, ronda cerrada")

        # ── Comprobación ─────────────────────────────────────
        stats = api(cli, "GET", f"/studies/{sid}/stats")
        print("\n  Estudio listo. Ábrelo en http://localhost:3000/delphi")
        print(f"  id: {sid}")
        return


if __name__ == "__main__":
    sys.exit(main())
