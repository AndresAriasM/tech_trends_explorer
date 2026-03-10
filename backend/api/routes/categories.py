"""
Category CRUD routes.
GET    /api/categories
POST   /api/categories
DELETE /api/categories/{category_id}
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime, timezone

router = APIRouter()

# ── Models ───────────────────────────────────────────────────

class CategoryCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    parent_category_id: Optional[str] = None

class CategoryResponse(BaseModel):
    category_id: str
    name: str
    description: str
    created_at: str
    parent_category_id: Optional[str] = None
    technologies: list = []

# ── Routes ───────────────────────────────────────────────────

@router.get("", response_model=list[CategoryResponse])
async def list_categories():
    """List all categories from DynamoDB."""
    # TODO: wire to storage.dynamodb
    # from storage.dynamodb import DynamoDBStorage
    # db = DynamoDBStorage()
    # return db.get_categories()
    return [
        {
            "category_id": "default",
            "name": "Sin categoría",
            "description": "Categoría por defecto",
            "created_at": "2025-01-01T00:00:00Z",
            "technologies": [],
        }
    ]


@router.post("", response_model=CategoryResponse, status_code=201)
async def create_category(payload: CategoryCreate):
    """Create a new category."""
    # TODO: persist to DynamoDB
    new_cat = {
        "category_id": f"cat_{uuid.uuid4().hex[:12]}",
        "name": payload.name,
        "description": payload.description or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_category_id": payload.parent_category_id,
        "technologies": [],
    }
    return new_cat


@router.delete("/{category_id}", status_code=204)
async def delete_category(category_id: str):
    """Delete a category (not default)."""
    if category_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete default category")
    # TODO: remove from DynamoDB
    return None
