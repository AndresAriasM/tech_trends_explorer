"""
backend/core/ai_voter.py

Genera votos sintéticos de IA para cada variable del taller prospectivo.

Flujo por variable:
  1. Busca evidencia reciente en SerpAPI (si está configurado)
  2. Construye un prompt con contexto LACCEI + evidencia
  3. Llama a OpenAI GPT-4o para obtener rating_current, rating_2032 y justificación
  4. Persiste el voto con is_ai_vote=True via WorkshopStorage

Se lanza como BackgroundTask desde el endpoint POST /api/workshop/{session_id}/ai-vote
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Dependencias opcionales ────────────────────────────────────────────────────

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai no instalado — pip install openai")

try:
    from serpapi.google_search import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False
    logger.warning("serpapi no instalado — pip install google-search-results")


# ── Constantes ─────────────────────────────────────────────────────────────────

RATING_MAP = {1: "Malo", 2: "Regular", 3: "Bueno", 4: "Excelente"}
AI_EXPERT_ID   = "AI_SYNTHETIC"
AI_EXPERT_NAME = "Experto Sintético IA"

SYSTEM_PROMPT = """Eres un experto en educación en ingeniería, innovación tecnológica y prospectiva \
para América Latina y el Caribe (región LACCEI).

Tu tarea es evaluar variables clave del ecosistema de educación en ingeniería considerando:
- El estado ACTUAL de cada variable en la región (2026)
- Una proyección al año 2032

Responde SIEMPRE en el siguiente formato JSON (sin texto adicional):
{
  "rating_current": <entero 1-4>,
  "rating_2032": <entero 1-4>,
  "justification": "<párrafo de 2-3 oraciones en español>"
}

Escala de ratings:
  1 = Malo      (situación problemática, barreras significativas)
  2 = Regular   (desafíos notables, avance limitado)
  3 = Bueno     (condiciones favorables, progreso moderado)
  4 = Excelente (situación óptima, avance sólido)

Basa tu evaluación en evidencia real: tendencias geopolíticas, datos de UNESCO, BID, OCDE, \
informes de mercado laboral, adopción tecnológica, publicaciones académicas y noticias recientes \
proporcionadas en el contexto."""


# ── Clase principal ────────────────────────────────────────────────────────────


class AIVoter:
    """Genera votos sintéticos de IA para variables del taller prospectivo."""

    def __init__(
        self,
        openai_api_key:  Optional[str] = None,
        serpapi_api_key: Optional[str] = None,
        model:           str           = "gpt-4o",
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("Instala openai: pip install openai")

        self.openai_api_key  = openai_api_key  or os.getenv("OPENAI_API_KEY")
        self.serpapi_api_key = serpapi_api_key or os.getenv("SERP_API_KEY")
        self.model           = model

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY no configurada")

        self.client = OpenAI(api_key=self.openai_api_key)

    # ── API pública ────────────────────────────────────────────────────────

    async def vote_all_variables(
        self,
        session_id: str,
        variables:  List[Dict],
        storage,                   # WorkshopStorage — evitamos import circular
    ) -> Dict:
        """
        Genera votos para todas las variables de una sesión.
        Corre en background; actualiza DynamoDB en tiempo real.

        Retorna resumen: {success, voted, failed, errors}
        """
        voted  = 0
        failed = 0
        errors = []

        for var in variables:
            try:
                await self._vote_variable(session_id, var, storage)
                voted += 1
                # Pequeña pausa para no saturar la API de OpenAI
                await asyncio.sleep(0.5)
            except Exception as e:
                failed += 1
                errors.append({"variable_id": var["id"], "error": str(e)})
                logger.error("Error votando variable %s: %s", var["id"], e)

        # Marcar sesión como "AI votes ready"
        try:
            storage.mark_ai_votes_ready(session_id)
        except Exception as e:
            logger.warning("No se pudo marcar ai_votes_ready: %s", e)

        logger.info(
            "AIVoter completado — sesión %s: %d votadas, %d fallidas",
            session_id, voted, failed,
        )
        return {"success": failed == 0, "voted": voted, "failed": failed, "errors": errors}

    # ── Internos ───────────────────────────────────────────────────────────

    async def _vote_variable(
        self,
        session_id: str,
        var:        Dict,
        storage,
    ) -> None:
        """Genera y persiste el voto AI para una variable."""

        # 1. Buscar evidencia (I/O bloqueante → thread pool)
        evidence = await asyncio.to_thread(self._fetch_evidence, var)

        # 2. Construir prompt
        user_prompt = self._build_prompt(var, evidence)

        # 3. Llamar a OpenAI (bloqueante → thread pool)
        result = await asyncio.to_thread(self._call_openai, user_prompt)

        # 4. Persistir voto
        storage.save_vote(
            session_id=session_id,
            variable_id=var["id"],
            variable_name=var["name"],
            expert_id=AI_EXPERT_ID,
            expert_name=AI_EXPERT_NAME,
            rating_current=result["rating_current"],
            rating_2032=result["rating_2032"],
            justification=result["justification"],
            is_ai_vote=True,
            ai_evidence=evidence[:1500] if evidence else "",
        )
        logger.info(
            "AI voto guardado — var=%s current=%s 2032=%s",
            var["id"], result["rating_current"], result["rating_2032"],
        )

    def _fetch_evidence(self, var: Dict) -> str:
        """Busca snippets de noticias recientes con SerpAPI para enriquecer el prompt."""
        if not SERPAPI_AVAILABLE or not self.serpapi_api_key:
            return ""

        keywords = var.get("search_keywords", var["name"])
        try:
            results = GoogleSearch({
                "q":       keywords,
                "hl":      "es",
                "num":     5,
                "api_key": self.serpapi_api_key,
            }).get_dict()

            snippets = []
            for r in results.get("organic_results", [])[:5]:
                title   = r.get("title",   "")
                snippet = r.get("snippet", "")
                if snippet:
                    snippets.append(f"• {title}: {snippet}")

            return "\n".join(snippets) if snippets else ""
        except Exception as e:
            logger.warning("SerpAPI error para '%s': %s", var["name"], e)
            return ""

    def _build_prompt(self, var: Dict, evidence: str) -> str:
        prompt = f"""
VARIABLE A EVALUAR: {var["name"]}
CATEGORÍA: {var["category"]}
DESCRIPCIÓN: {var["description"]}

CONTEXTO LACCEI:
La red LACCEI (Latin American and Caribbean Consortium of Engineering Institutions) agrupa \
a universidades de ingeniería de América Latina y el Caribe. Este taller prospectivo evalúa \
el entorno que condicionará la educación en ingeniería en la región hacia 2032.
"""
        if evidence:
            prompt += f"\nEVIDENCIA RECIENTE (noticias/reportes 2024-2026):\n{evidence}\n"
        else:
            prompt += "\nNota: No se encontró evidencia web reciente. Basa tu evaluación en conocimiento general actualizado.\n"

        prompt += "\nGenera tu evaluación siguiendo estrictamente el formato JSON indicado."
        return prompt

    def _call_openai(self, user_prompt: str) -> Dict:
        """Llama a OpenAI y parsea la respuesta JSON."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        rating_c = int(data.get("rating_current", 2))
        rating_f = int(data.get("rating_2032",    2))
        justif   = str(data.get("justification",  "Sin justificación generada."))

        # Sanitizar rangos
        rating_c = max(1, min(4, rating_c))
        rating_f = max(1, min(4, rating_f))

        return {
            "rating_current": rating_c,
            "rating_2032":    rating_f,
            "justification":  justif,
        }


# ── Factory helper ─────────────────────────────────────────────────────────────

def get_ai_voter() -> AIVoter:
    return AIVoter(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        serpapi_api_key=os.getenv("SERP_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    )
