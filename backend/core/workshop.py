"""
backend/core/workshop.py
Lógica de negocio para talleres prospectivos LACCEI 2032.

Gestiona sesiones, votos de expertos y síntesis de resultados.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, List, Optional

from storage.workshop import WorkshopStorage

logger = logging.getLogger(__name__)

# ── Catálogo de variables del taller ──────────────────────────────────────────
# Orden y agrupación acordados con el equipo LACCEI

WORKSHOP_VARIABLES: List[Dict] = [
    # ── Económicas / Financieras ───────────────────────────────────────────
    {
        "id": 1,
        "name": "Costos operativos - tecnológicos",
        "category": "Económica / Financiera",
        "description": (
            "Nivel de los costos de operación e infraestructura tecnológica "
            "que enfrentan las instituciones de educación en ingeniería de LACCEI."
        ),
        "search_keywords": "operational technology costs higher education Latin America engineering",
    },
    {
        "id": 2,
        "name": "Fondos en educación e innovación",
        "category": "Económica / Financiera",
        "description": (
            "Disponibilidad y suficiencia de financiamiento público y privado "
            "para educación superior e innovación en la región."
        ),
        "search_keywords": "education innovation funding Latin America Caribbean higher education",
    },
    {
        "id": 3,
        "name": "Sostenibilidad financiera",
        "category": "Económica / Financiera",
        "description": (
            "Capacidad de las instituciones de mantener operaciones a largo plazo "
            "ante presiones presupuestarias, inflación y reducción de matrícula."
        ),
        "search_keywords": "financial sustainability higher education institutions Latin America",
    },
    {
        "id": 4,
        "name": "Pobreza digital en capacidad de computación",
        "category": "Económica / Financiera",
        "description": (
            "Brecha en acceso a hardware, conectividad y capacidad de cómputo "
            "entre estudiantes e instituciones de distintos estratos económicos."
        ),
        "search_keywords": "digital poverty computing capacity access Latin America students",
    },
    # ── Social / Geopolítica ───────────────────────────────────────────────
    {
        "id": 5,
        "name": "Inseguridad en países sedes",
        "category": "Social / Geopolítica",
        "description": (
            "Impacto de la inseguridad ciudadana y violencia en la operación "
            "de congresos, movilidad académica y retención de talento en la región."
        ),
        "search_keywords": "security violence impact academic institutions Latin America Caribbean",
    },
    {
        "id": 6,
        "name": "Migración",
        "category": "Social / Geopolítica",
        "description": (
            "Efecto de la emigración de talento técnico-científico (brain drain) "
            "y la migración intra-regional en la capacidad de ingeniería local."
        ),
        "search_keywords": "brain drain migration engineers Latin America higher education",
    },
    {
        "id": 7,
        "name": "Geopolítica global",
        "category": "Social / Geopolítica",
        "description": (
            "Influencia de tensiones geopolíticas mundiales en la cooperación "
            "científica, acceso a tecnología y relaciones internacionales de LACCEI."
        ),
        "search_keywords": "geopolitics global impact scientific cooperation Latin America 2032",
    },
    {
        "id": 8,
        "name": "Problemas sociales de la región",
        "category": "Social / Geopolítica",
        "description": (
            "Desigualdad social, conflictos internos y problemáticas estructurales "
            "que condicionan el desarrollo de la ingeniería en Latinoamérica."
        ),
        "search_keywords": "social problems inequality Latin America engineering education impact",
    },
    # ── Digital / Tecnológica ─────────────────────────────────────────────
    {
        "id": 9,
        "name": "Brechas por uso de IA",
        "category": "Digital / Tecnológica",
        "description": (
            "Desigualdad en el acceso, comprensión y aprovechamiento de "
            "herramientas de inteligencia artificial entre instituciones y estudiantes."
        ),
        "search_keywords": "AI artificial intelligence gap inequality higher education Latin America",
    },
    {
        "id": 10,
        "name": "Barreras lingüísticas",
        "category": "Digital / Tecnológica",
        "description": (
            "Obstáculos derivados del predominio del inglés en publicaciones "
            "científicas, tecnología y colaboración internacional para la región."
        ),
        "search_keywords": "language barrier Spanish Portuguese engineering research Latin America",
    },
    # ── Institucional / Colaboración ──────────────────────────────────────
    {
        "id": 11,
        "name": "Colaboración entre países - regiones",
        "category": "Institucional / Colaboración",
        "description": (
            "Grado de articulación y proyectos conjuntos entre instituciones "
            "de distintos países en investigación, movilidad y programas académicos."
        ),
        "search_keywords": "international collaboration engineering higher education Latin America partnerships",
    },
    {
        "id": 12,
        "name": "Colaboración regional",
        "category": "Institucional / Colaboración",
        "description": (
            "Efectividad de redes regionales como LACCEI para coordinar "
            "esfuerzos conjuntos de investigación y formación en ingeniería."
        ),
        "search_keywords": "LACCEI regional collaboration Latin America Caribbean engineering network",
    },
    {
        "id": 13,
        "name": "Noción de altruismo - voluntariado",
        "category": "Institucional / Colaboración",
        "description": (
            "Cultura de contribución voluntaria y servicio dentro de la comunidad "
            "académica de ingeniería; sostenibilidad de redes basadas en voluntarios."
        ),
        "search_keywords": "volunteerism altruism academic community engineering professional associations",
    },
    {
        "id": 14,
        "name": "Gobernanza institucional",
        "category": "Institucional / Colaboración",
        "description": (
            "Calidad de los marcos de gobernanza de universidades y redes regionales; "
            "transparencia, agilidad y capacidad de adaptación institucional."
        ),
        "search_keywords": "institutional governance higher education Latin America reform",
    },
    {
        "id": 15,
        "name": "Prioridades de miembros LACCEI",
        "category": "Institucional / Colaboración",
        "description": (
            "Alineación y divergencias entre las prioridades estratégicas de las "
            "instituciones miembro de LACCEI en la región."
        ),
        "search_keywords": "LACCEI member priorities strategic engineering education Latin America",
    },
    # ── Académica / Investigación ─────────────────────────────────────────
    {
        "id": 16,
        "name": "Exigencias en investigación de profesores",
        "category": "Académica / Investigación",
        "description": (
            "Presión sobre los docentes-investigadores para cumplir métricas de "
            "publicación, patentes e indicadores de productividad científica."
        ),
        "search_keywords": "faculty research pressure publish productivity metrics Latin America universities",
    },
    {
        "id": 17,
        "name": "Igualdad, equidad e inclusión en ingeniería",
        "category": "Académica / Investigación",
        "description": (
            "Avances y barreras para lograr diversidad de género, origen y "
            "capacidades en programas de ingeniería de la región."
        ),
        "search_keywords": "diversity equity inclusion DEI engineering education Latin America women STEM",
    },
    {
        "id": 18,
        "name": "Transformación mercado laboral de la ingeniería",
        "category": "Académica / Investigación",
        "description": (
            "Impacto de la automatización, IA y nuevos perfiles laborales en "
            "la demanda y competencias del ingeniero latinoamericano hacia 2032."
        ),
        "search_keywords": "engineering labor market transformation automation AI jobs Latin America 2032",
    },
    # ── Regulatoria / Normativa ───────────────────────────────────────────
    {
        "id": 19,
        "name": "Normativas internacionales sobre IA y tecnologías emergentes",
        "category": "Regulatoria / Normativa",
        "description": (
            "Evolución del marco regulatorio global sobre IA, datos y tecnologías "
            "emergentes y su impacto en instituciones educativas de la región."
        ),
        "search_keywords": "AI regulation emerging technology policy international impact education 2032",
    },
    {
        "id": 20,
        "name": "Brechas en acceso a la educación",
        "category": "Regulatoria / Normativa",
        "description": (
            "Desigualdades persistentes en el acceso a educación superior en "
            "ingeniería por razones geográficas, económicas o de conectividad."
        ),
        "search_keywords": "access higher education gaps inequality Latin America engineering enrollment",
    },
]

RATING_LABELS = {1: "Malo", 2: "Regular", 3: "Bueno", 4: "Excelente"}


# ── WorkshopService ────────────────────────────────────────────────────────────


class WorkshopService:
    """Orquesta sesiones, votos y cálculo de síntesis."""

    def __init__(self, db: WorkshopStorage):
        self.db = db

    # ── Sesiones ──────────────────────────────────────────────────────────

    def create_session(self, title: str, description: str = "", gforms_url: str = "") -> Dict:
        if not title.strip():
            raise ValueError("El título de la sesión no puede estar vacío")
        return self.db.create_session(title.strip(), description.strip(), gforms_url.strip())

    def get_session(self, session_id: str) -> Optional[Dict]:
        return self.db.get_session_by_id(session_id)

    def get_session_by_code(self, code: str) -> Optional[Dict]:
        return self.db.get_session_by_code(code)

    def list_sessions(self) -> List[Dict]:
        return self.db.list_sessions()

    def close_session(self, session_id: str) -> bool:
        return self.db.close_session(session_id)

    # ── Variables ─────────────────────────────────────────────────────────

    def get_variables(self) -> List[Dict]:
        """Retorna el catálogo completo de variables (sin info de búsqueda)."""
        return [
            {
                "id":          v["id"],
                "name":        v["name"],
                "category":    v["category"],
                "description": v["description"],
            }
            for v in WORKSHOP_VARIABLES
        ]

    # ── Votos ─────────────────────────────────────────────────────────────

    def register_expert_votes(
        self,
        session_id:   str,
        expert_name:  str,
        votes:        List[Dict],   # [{variable_id, rating_current, rating_2032, justification?}]
        allow_revote: bool = False,
    ) -> Dict:
        """
        Guarda todos los votos de un experto en una sesión.
        Genera un expert_id estable basado en sesión + nombre (normalizado).
        """
        session = self.db.get_session_by_id(session_id)
        if not session:
            raise ValueError("Sesión no encontrada")
        if session.get("status") == "closed":
            raise ValueError("La sesión está cerrada")

        expert_id = f"exp_{uuid.uuid5(uuid.NAMESPACE_DNS, session_id + expert_name.lower().strip()).hex[:12]}"

        if self.db.expert_has_voted(session_id, expert_id):
            if not allow_revote:
                raise ValueError("Este experto ya registró sus votos en esta sesión")
            self.db.delete_expert_votes(session_id, expert_id)

        # Validar y persistir
        var_map = {v["id"]: v for v in WORKSHOP_VARIABLES}
        saved_ids = []
        for vote in votes:
            var_id  = int(vote["variable_id"])
            rating_c = int(vote["rating_current"])
            rating_f = int(vote["rating_2032"])

            if var_id not in var_map:
                raise ValueError(f"Variable ID inválida: {var_id}")
            if not (1 <= rating_c <= 4) or not (1 <= rating_f <= 4):
                raise ValueError(f"Rating fuera de rango (1-4) en variable {var_id}")

            vid = self.db.save_vote(
                session_id=session_id,
                variable_id=var_id,
                variable_name=var_map[var_id]["name"],
                expert_id=expert_id,
                expert_name=expert_name,
                rating_current=rating_c,
                rating_2032=rating_f,
                justification=vote.get("justification", ""),
            )
            saved_ids.append(vid)

        return {"expert_id": expert_id, "votes_saved": len(saved_ids)}

    # ── Síntesis ──────────────────────────────────────────────────────────

    def compute_synthesis(self, session_id: str) -> Dict:
        """
        Calcula estadísticas de consenso por variable y genera el resumen general.

        Retorna un dict con:
          - variables: lista de stats por variable
          - overall:   métricas globales de la sesión
          - divergences: las 3 variables con mayor disenso
        """
        all_votes = self.db.get_votes_for_session(session_id)

        # Separar votos humanos de votos AI
        human_votes = [v for v in all_votes if not v.get("is_ai_vote")]
        ai_votes    = {v["variable_id"]: v for v in all_votes if v.get("is_ai_vote")}

        # Expertos únicos
        expert_ids = list({v["expert_id"] for v in human_votes})

        # Agrupar por variable
        by_var: Dict[int, List[Dict]] = defaultdict(list)
        for v in human_votes:
            by_var[int(v["variable_id"])].append(v)

        var_stats = []
        for var in WORKSHOP_VARIABLES:
            vid   = var["id"]
            votes = by_var.get(vid, [])

            if not votes:
                var_stats.append({
                    "variable_id":   vid,
                    "variable_name": var["name"],
                    "category":      var["category"],
                    "n_responses":   0,
                    "avg_current":   None,
                    "avg_2032":      None,
                    "std_current":   None,
                    "std_2032":      None,
                    "label_current": None,
                    "label_2032":    None,
                    "consensus_score": None,
                    "distribution_current": {},
                    "distribution_2032":    {},
                    "ai_vote": ai_votes.get(vid),
                    "ai_divergence_current": None,
                    "ai_divergence_2032":    None,
                })
                continue

            rc = [int(v["rating_current"]) for v in votes]
            rf = [int(v["rating_2032"])    for v in votes]

            avg_c = round(mean(rc), 2)
            avg_f = round(mean(rf), 2)
            std_c = round(stdev(rc), 2) if len(rc) > 1 else 0.0
            std_f = round(stdev(rf), 2) if len(rf) > 1 else 0.0

            # Consenso: 1 = total acuerdo, 0 = máximo disenso
            consensus = round(1 - (std_c / 1.5), 2)   # stdev máx ≈ 1.5
            consensus = max(0.0, min(1.0, consensus))

            # Distribución de ratings
            dist_c = {RATING_LABELS[i]: rc.count(i) for i in range(1, 5)}
            dist_f = {RATING_LABELS[i]: rf.count(i) for i in range(1, 5)}

            # Comparación con voto AI
            ai_v = ai_votes.get(vid)
            ai_div_c = ai_div_f = None
            if ai_v:
                ai_div_c = round(abs(avg_c - int(ai_v["rating_current"])), 2)
                ai_div_f = round(abs(avg_f - int(ai_v["rating_2032"])),    2)

            var_stats.append({
                "variable_id":   vid,
                "variable_name": var["name"],
                "category":      var["category"],
                "n_responses":   len(votes),
                "avg_current":   avg_c,
                "avg_2032":      avg_f,
                "std_current":   std_c,
                "std_2032":      std_f,
                "label_current": _avg_to_label(avg_c),
                "label_2032":    _avg_to_label(avg_f),
                "consensus_score": consensus,
                "distribution_current": dist_c,
                "distribution_2032":    dist_f,
                "ai_vote":       ai_v,
                "ai_divergence_current": ai_div_c,
                "ai_divergence_2032":    ai_div_f,
            })

        # Top 3 variables con mayor disenso (mayor std)
        divergences = sorted(
            [s for s in var_stats if s["std_current"] is not None],
            key=lambda x: x["std_current"],
            reverse=True,
        )[:3]

        # Métricas globales
        answered = [s for s in var_stats if s["n_responses"] > 0]
        overall_avg_c = round(mean([s["avg_current"] for s in answered]), 2) if answered else None
        overall_avg_f = round(mean([s["avg_2032"]    for s in answered]), 2) if answered else None

        return {
            "session_id":      session_id,
            "n_experts":       len(expert_ids),
            "n_variables":     len(WORKSHOP_VARIABLES),
            "variables":       var_stats,
            "divergences":     divergences,
            "overall": {
                "avg_current": overall_avg_c,
                "avg_2032":    overall_avg_f,
                "label_current": _avg_to_label(overall_avg_c) if overall_avg_c else None,
                "label_2032":    _avg_to_label(overall_avg_f) if overall_avg_f else None,
            },
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _avg_to_label(avg: Optional[float]) -> Optional[str]:
    if avg is None:
        return None
    if avg < 1.5:
        return "Malo"
    if avg < 2.5:
        return "Regular"
    if avg < 3.5:
        return "Bueno"
    return "Excelente"
