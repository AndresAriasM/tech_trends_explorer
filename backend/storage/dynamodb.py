"""
DynamoDB storage layer — ported from src/data_storage.py
All Streamlit dependencies removed; errors are raised as exceptions or logged.
"""

import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import boto3
    from boto3.dynamodb.conditions import Attr
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.error("boto3 not installed. Run: pip install boto3")


class DynamoDBStorage:
    """
    Manages persistent storage for tech-trends data using AWS DynamoDB.
    Table names:
      - tech-trends-analyses   (PK: analysis_id, SK: timestamp)
      - tech-trends-categories (PK: category_id)
    """

    ANALYSES_TABLE = "tech-trends-analyses"
    CATEGORIES_TABLE = "tech-trends-categories"

    def __init__(
        self,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        if not BOTO3_AVAILABLE:
            raise RuntimeError("boto3 is not installed")

        session_kwargs: Dict[str, Any] = {}
        if region_name:
            session_kwargs["region_name"] = region_name
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.dynamodb = boto3.resource("dynamodb", **session_kwargs)
        self.analyses_table = self.dynamodb.Table(self.ANALYSES_TABLE)
        self.categories_table = self.dynamodb.Table(self.CATEGORIES_TABLE)
        self._verify_connection()

    # ── Connection ────────────────────────────────────────────

    def _verify_connection(self):
        try:
            _ = self.analyses_table.table_status
            _ = self.categories_table.table_status
            self._ensure_default_category()
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "ResourceNotFoundException":
                raise RuntimeError(
                    "DynamoDB tables not found. Required: "
                    f"{self.ANALYSES_TABLE}, {self.CATEGORIES_TABLE}"
                ) from e
            raise
        except NoCredentialsError as e:
            raise RuntimeError("AWS credentials not configured") from e

    def _ensure_default_category(self):
        try:
            resp = self.categories_table.get_item(Key={"category_id": "default"})
            if "Item" not in resp:
                self.categories_table.put_item(
                    Item={
                        "category_id": "default",
                        "name": "Sin categoría",
                        "description": "Categoría por defecto",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
        except Exception as e:
            logger.warning("Could not ensure default category: %s", e)

    # ── ID helpers ────────────────────────────────────────────

    @staticmethod
    def generate_id(prefix: str = "analysis") -> str:
        ts = int(time.time() * 1000)
        uid = uuid.uuid4().hex[:8]
        return f"{prefix}_{ts}_{uid}"

    # ── Data sanitization ─────────────────────────────────────

    @staticmethod
    def _sanitize(data: Any) -> Any:
        """Remove NaN / Inf floats recursively."""
        if isinstance(data, dict):
            return {k: DynamoDBStorage._sanitize(v) for k, v in data.items()}
        if isinstance(data, list):
            return [DynamoDBStorage._sanitize(i) for i in data]
        if isinstance(data, float):
            return 0.0 if (math.isnan(data) or math.isinf(data)) else data
        if hasattr(data, "item"):  # numpy scalar
            try:
                v = float(data.item())
                return 0.0 if (math.isnan(v) or math.isinf(v)) else v
            except Exception:
                return 0.0
        return data

    @staticmethod
    def to_decimal(obj: Any) -> Any:
        """Convert floats → Decimal for DynamoDB."""
        if isinstance(obj, dict):
            return {k: DynamoDBStorage.to_decimal(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [DynamoDBStorage.to_decimal(i) for i in obj]
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return Decimal("0")
            try:
                return Decimal(str(round(obj, 6)))
            except Exception:
                return Decimal("0")
        if isinstance(obj, (int, str, bool)) or obj is None:
            return obj
        if hasattr(obj, "item"):  # numpy scalar
            try:
                v = float(obj.item())
                if math.isnan(v) or math.isinf(v):
                    return Decimal("0")
                return Decimal(str(round(v, 6)))
            except Exception:
                return Decimal("0")
        return str(obj)

    @staticmethod
    def from_decimal(obj: Any) -> Any:
        """Convert Decimal → float when reading from DynamoDB."""
        if isinstance(obj, dict):
            return {k: DynamoDBStorage.from_decimal(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [DynamoDBStorage.from_decimal(i) for i in obj]
        if isinstance(obj, Decimal):
            return float(obj)
        return obj

    # ── Analyses ──────────────────────────────────────────────

    def put_analysis(self, data: Dict, category_id: str = "default") -> Optional[str]:
        """Save an analysis item. Returns the generated analysis_id."""
        try:
            analysis_id = self.generate_id("analysis")
            timestamp = datetime.now(timezone.utc).isoformat()
            item = {
                "analysis_id": analysis_id,
                "timestamp": timestamp,
                "category_id": category_id,
                **self._sanitize(data),
            }
            self.analyses_table.put_item(Item=self.to_decimal(item))
            logger.info("Analysis saved: %s", analysis_id)
            return analysis_id
        except Exception as e:
            logger.error("Error saving analysis: %s", e)
            return None

    def put_analysis_raw(self, item: Dict) -> bool:
        """Save a pre-built item dict directly (used by HypeCycleStorage)."""
        try:
            self.analyses_table.put_item(Item=self.to_decimal(item))
            return True
        except Exception as e:
            logger.error("Error saving raw analysis: %s", e)
            return False

    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict]:
        try:
            resp = self.analyses_table.scan(
                FilterExpression=Attr("analysis_id").eq(analysis_id)
            )
            items = resp.get("Items", [])
            return self.from_decimal(items[0]) if items else None
        except Exception as e:
            logger.error("Error fetching analysis %s: %s", analysis_id, e)
            return None

    def get_analyses_by_category(self, category_id: str) -> List[Dict]:
        try:
            items: List[Dict] = []
            resp = self.analyses_table.scan(
                FilterExpression=Attr("category_id").eq(category_id)
            )
            items.extend(resp.get("Items", []))
            while "LastEvaluatedKey" in resp:
                resp = self.analyses_table.scan(
                    FilterExpression=Attr("category_id").eq(category_id),
                    ExclusiveStartKey=resp["LastEvaluatedKey"],
                )
                items.extend(resp.get("Items", []))
            return [self.from_decimal(i) for i in items]
        except Exception as e:
            logger.error("Error fetching analyses for category %s: %s", category_id, e)
            return []

    def get_all_analyses(self) -> List[Dict]:
        try:
            items: List[Dict] = []
            resp = self.analyses_table.scan()
            items.extend(resp.get("Items", []))
            while "LastEvaluatedKey" in resp:
                resp = self.analyses_table.scan(
                    ExclusiveStartKey=resp["LastEvaluatedKey"]
                )
                items.extend(resp.get("Items", []))
            converted = [self.from_decimal(i) for i in items]
            converted.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return converted
        except Exception as e:
            logger.error("Error fetching all analyses: %s", e)
            return []

    def delete_analysis(self, analysis_id: str, timestamp: str) -> bool:
        try:
            self.analyses_table.delete_item(
                Key={"analysis_id": analysis_id, "timestamp": timestamp}
            )
            return True
        except Exception as e:
            logger.error("Error deleting analysis %s: %s", analysis_id, e)
            return False

    # ── Categories ────────────────────────────────────────────

    def get_all_categories(self) -> List[Dict]:
        try:
            items: List[Dict] = []
            resp = self.categories_table.scan()
            items.extend(resp.get("Items", []))
            while "LastEvaluatedKey" in resp:
                resp = self.categories_table.scan(
                    ExclusiveStartKey=resp["LastEvaluatedKey"]
                )
                items.extend(resp.get("Items", []))
            items.sort(key=lambda x: x.get("name", ""))
            return items
        except Exception as e:
            logger.error("Error fetching categories: %s", e)
            return [
                {
                    "category_id": "default",
                    "name": "Sin categoría",
                    "description": "Categoría por defecto",
                    "created_at": "",
                }
            ]

    def get_category_by_id(self, category_id: str) -> Optional[Dict]:
        try:
            resp = self.categories_table.get_item(Key={"category_id": category_id})
            return resp.get("Item")
        except Exception as e:
            logger.error("Error fetching category %s: %s", category_id, e)
            return None

    def create_category(self, name: str, description: str = "") -> Optional[str]:
        try:
            category_id = self.generate_id("cat")
            self.categories_table.put_item(
                Item={
                    "category_id": category_id,
                    "name": name,
                    "description": description,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            return category_id
        except Exception as e:
            logger.error("Error creating category: %s", e)
            return None

    def update_category(
        self,
        category_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> tuple[bool, str]:
        try:
            if not self.get_category_by_id(category_id):
                return False, "La categoría no existe"

            parts, values = [], {}
            if name is not None:
                parts.append("name = :name")
                values[":name"] = name
            if description is not None:
                parts.append("description = :description")
                values[":description"] = description
            if not parts:
                return False, "No hay cambios"

            self.categories_table.update_item(
                Key={"category_id": category_id},
                UpdateExpression="SET " + ", ".join(parts),
                ExpressionAttributeValues=values,
            )
            return True, "Categoría actualizada"
        except Exception as e:
            return False, f"Error: {e}"

    def delete_category(self, category_id: str) -> tuple[bool, str]:
        if category_id == "default":
            return False, "No se puede eliminar la categoría por defecto"
        try:
            if not self.get_category_by_id(category_id):
                return False, "La categoría no existe"
            self.categories_table.delete_item(Key={"category_id": category_id})
            return True, "Categoría eliminada"
        except Exception as e:
            return False, f"Error: {e}"

    def count_analyses_in_category(self, category_id: str) -> int:
        return len(self.get_analyses_by_category(category_id))


# ── Module-level singleton factory ────────────────────────────

_instance: Optional[DynamoDBStorage] = None


def get_db() -> DynamoDBStorage:
    """
    Return a module-level DynamoDBStorage singleton.
    Reads credentials from environment variables via boto3 defaults
    (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION).
    """
    global _instance
    if _instance is None:
        import os
        _instance = DynamoDBStorage(
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-2")
        )
    return _instance
