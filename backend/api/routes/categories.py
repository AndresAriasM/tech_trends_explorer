"""
Category CRUD routes — wired to DynamoDB via CategoryService.
GET    /api/categories
POST   /api/categories
PATCH  /api/categories/{category_id}
DELETE /api/categories/{category_id}
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ── DynamoDB dependency ───────────────────────────────────────

def get_category_service():
    from storage.dynamodb import get_db
    from core.categories import CategoryService
    return CategoryService(get_db())


# ── Models ────────────────────────────────────────────────────

class CategoryCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    parent_category_id: Optional[str] = None


class CategoryUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class CategoryResponse(BaseModel):
    category_id: str
    name: str
    description: str
    created_at: str
    parent_category_id: Optional[str] = None
    technologies: list = []


# ── Routes ────────────────────────────────────────────────────

@router.get("", response_model=list[CategoryResponse])
async def list_categories(service=Depends(get_category_service)):
    """List all categories from DynamoDB."""
    try:
        cats = service.list_categories()
        # Normalize fields not stored in DynamoDB (technologies list)
        for cat in cats:
            cat.setdefault("description", "")
            cat.setdefault("created_at", "")
            cat.setdefault("technologies", [])
        return cats
    except Exception as e:
        logger.error("Error listing categories: %s", e)
        raise HTTPException(status_code=500, detail="Error fetching categories")


@router.post("", response_model=CategoryResponse, status_code=201)
async def create_category(
    payload: CategoryCreate,
    service=Depends(get_category_service),
):
    """Create a new category in DynamoDB."""
    try:
        cat_id = service.create_category(payload.name, payload.description or "")
        if not cat_id:
            raise HTTPException(status_code=500, detail="Failed to create category")
        cat = service.get_category(cat_id)
        cat.setdefault("technologies", [])
        cat.setdefault("parent_category_id", payload.parent_category_id)
        return cat
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating category: %s", e)
        raise HTTPException(status_code=500, detail="Error creating category")


@router.patch("/{category_id}", response_model=CategoryResponse)
async def update_category(
    category_id: str,
    payload: CategoryUpdate,
    service=Depends(get_category_service),
):
    """Update an existing category."""
    try:
        ok, msg = service.update_category(
            category_id,
            name=payload.name,
            description=payload.description,
        )
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        cat = service.get_category(category_id)
        cat.setdefault("technologies", [])
        return cat
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating category %s: %s", category_id, e)
        raise HTTPException(status_code=500, detail="Error updating category")


@router.delete("/{category_id}", status_code=204)
async def delete_category(
    category_id: str,
    service=Depends(get_category_service),
):
    """Delete a category (cannot delete 'default')."""
    try:
        ok, msg = service.delete_category(category_id)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting category %s: %s", category_id, e)
        raise HTTPException(status_code=500, detail="Error deleting category")


@router.get("/{category_id}/usage")
async def category_usage(
    category_id: str,
    service=Depends(get_category_service),
):
    """Return how many analyses reference this category."""
    try:
        count = service.usage_count(category_id)
        return {"category_id": category_id, "analysis_count": count}
    except Exception as e:
        logger.error("Error counting usage for %s: %s", category_id, e)
        raise HTTPException(status_code=500, detail="Error counting usage")
