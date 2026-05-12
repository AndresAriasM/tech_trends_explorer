"""
backend/storage/workshop.py
DynamoDB CRUD para sesiones de taller prospectivo y votos de expertos.

Tablas:
  - laccei-workshop-sessions  (PK: session_id)
  - laccei-workshop-votes     (PK: vote_id, SK: session_id)
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from boto3.dynamodb.conditions import Attr
    from botocore.exceptions import ClientError
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

# Nombres de tablas
SESSIONS_TABLE = "laccei-workshop-sessions"
VOTES_TABLE    = "laccei-workshop-votes"


def _gen_id(prefix: str) -> str:
    ts  = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{uid}"


def _gen_code() -> str:
    """Código corto de 6 letras mayúsculas para que los expertos entren a la sala."""
    import random, string
    return "".join(random.choices(string.ascii_uppercase, k=6))


class WorkshopStorage:
    """Capa de persistencia para el módulo de talleres prospectivos."""

    def __init__(self, region_name: str = "us-east-2"):
        if not BOTO3_AVAILABLE:
            raise RuntimeError("boto3 no instalado")
        ddb = boto3.resource("dynamodb", region_name=region_name)
        self._sessions = ddb.Table(SESSIONS_TABLE)
        self._votes    = ddb.Table(VOTES_TABLE)
        self._ensure_tables()

    # ── Inicialización ─────────────────────────────────────────────────────

    def _ensure_tables(self):
        """Crea las tablas si no existen (idempotente)."""
        client = boto3.client(
            "dynamodb",
            region_name=self._sessions.meta.client.meta.region_name,
        )
        existing = {t for t in client.list_tables()["TableNames"]}

        if SESSIONS_TABLE not in existing:
            client.create_table(
                TableName=SESSIONS_TABLE,
                AttributeDefinitions=[
                    {"AttributeName": "session_id", "AttributeType": "S"},
                ],
                KeySchema=[{"AttributeName": "session_id", "KeyType": "HASH"}],
                BillingMode="PAY_PER_REQUEST",
            )
            logger.info("Tabla %s creada", SESSIONS_TABLE)

        if VOTES_TABLE not in existing:
            client.create_table(
                TableName=VOTES_TABLE,
                AttributeDefinitions=[
                    {"AttributeName": "vote_id",    "AttributeType": "S"},
                    {"AttributeName": "session_id", "AttributeType": "S"},
                ],
                KeySchema=[
                    {"AttributeName": "vote_id",    "KeyType": "HASH"},
                    {"AttributeName": "session_id", "KeyType": "RANGE"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            logger.info("Tabla %s creada", VOTES_TABLE)

    # ── Sesiones ───────────────────────────────────────────────────────────

    def create_session(self, title: str, description: str = "", gforms_url: str = "") -> Dict:
        """Crea una sesión de taller. Retorna el item completo con session_id y code."""
        session_id = _gen_id("ws")
        code       = _gen_code()
        now        = datetime.now(timezone.utc).isoformat()
        item = {
            "session_id":     session_id,
            "code":           code,
            "title":          title,
            "description":    description,
            "gforms_url":     gforms_url,   # URL del Google Form de votación
            "status":         "active",     # active | closed
            "created_at":     now,
            "closed_at":      None,
            "ai_votes_ready": False,
        }
        self._sessions.put_item(Item=item)
        logger.info("Sesión creada: %s (code=%s)", session_id, code)
        return item

    def update_gforms_url(self, session_id: str, gforms_url: str) -> bool:
        try:
            self._sessions.update_item(
                Key={"session_id": session_id},
                UpdateExpression="SET gforms_url = :u",
                ExpressionAttributeValues={":u": gforms_url},
            )
            return True
        except Exception as e:
            logger.error("Error actualizando gforms_url: %s", e)
            return False

    def get_session_by_id(self, session_id: str) -> Optional[Dict]:
        resp = self._sessions.get_item(Key={"session_id": session_id})
        return resp.get("Item")

    def get_session_by_code(self, code: str) -> Optional[Dict]:
        resp = self._sessions.scan(
            FilterExpression=Attr("code").eq(code.upper())
        )
        items = resp.get("Items", [])
        return items[0] if items else None

    def list_sessions(self) -> List[Dict]:
        resp  = self._sessions.scan()
        items = resp.get("Items", [])
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return items

    def close_session(self, session_id: str) -> bool:
        try:
            self._sessions.update_item(
                Key={"session_id": session_id},
                UpdateExpression="SET #s = :s, closed_at = :ca",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={
                    ":s":  "closed",
                    ":ca": datetime.now(timezone.utc).isoformat(),
                },
            )
            return True
        except Exception as e:
            logger.error("Error cerrando sesión %s: %s", session_id, e)
            return False

    def mark_ai_votes_ready(self, session_id: str) -> None:
        self._sessions.update_item(
            Key={"session_id": session_id},
            UpdateExpression="SET ai_votes_ready = :v",
            ExpressionAttributeValues={":v": True},
        )

    # ── Votos ──────────────────────────────────────────────────────────────

    def save_vote(
        self,
        session_id:      str,
        variable_id:     int,
        variable_name:   str,
        expert_id:       str,
        expert_name:     str,
        rating_current:  int,   # 1=Malo 2=Regular 3=Bueno 4=Excelente
        rating_2032:     int,
        justification:   str   = "",
        is_ai_vote:      bool  = False,
        ai_evidence:     str   = "",
    ) -> str:
        vote_id = _gen_id("vote")
        item = {
            "vote_id":       vote_id,
            "session_id":    session_id,
            "variable_id":   variable_id,
            "variable_name": variable_name,
            "expert_id":     expert_id,
            "expert_name":   expert_name,
            "rating_current": rating_current,
            "rating_2032":    rating_2032,
            "justification":  justification,
            "is_ai_vote":     is_ai_vote,
            "ai_evidence":    ai_evidence,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        }
        self._votes.put_item(Item=item)
        return vote_id

    def get_votes_for_session(self, session_id: str) -> List[Dict]:
        resp  = self._votes.scan(
            FilterExpression=Attr("session_id").eq(session_id)
        )
        items = resp.get("Items", [])
        items.sort(key=lambda x: (x.get("variable_id", 0), x.get("timestamp", "")))
        return items

    def get_votes_by_expert(self, session_id: str, expert_id: str) -> List[Dict]:
        resp = self._votes.scan(
            FilterExpression=(
                Attr("session_id").eq(session_id) & Attr("expert_id").eq(expert_id)
            )
        )
        return resp.get("Items", [])

    def expert_has_voted(self, session_id: str, expert_id: str) -> bool:
        votes = self.get_votes_by_expert(session_id, expert_id)
        return len(votes) > 0

    def delete_expert_votes(self, session_id: str, expert_id: str) -> int:
        """Elimina todos los votos de un experto en una sesión (para re-votar)."""
        votes = self.get_votes_by_expert(session_id, expert_id)
        for v in votes:
            self._votes.delete_item(
                Key={"vote_id": v["vote_id"], "session_id": session_id}
            )
        return len(votes)


# ── Singleton ──────────────────────────────────────────────────────────────

_instance: Optional[WorkshopStorage] = None


def get_workshop_db() -> WorkshopStorage:
    global _instance
    if _instance is None:
        import os
        _instance = WorkshopStorage(
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-2")
        )
    return _instance
