"""
Category business logic — pure CRUD on top of DynamoDBStorage.
Ported from src/category_admin.py (only the data layer, no Streamlit UI).
"""

from typing import Dict, List, Optional, Tuple
from storage.dynamodb import DynamoDBStorage


class CategoryService:
    """
    Thin service layer for category management.
    All UI logic has been stripped; this is pure business logic.
    """

    def __init__(self, db: DynamoDBStorage):
        self.db = db

    def list_categories(self) -> List[Dict]:
        """Return all categories sorted by name."""
        return self.db.get_all_categories()

    def get_category(self, category_id: str) -> Optional[Dict]:
        """Return a single category by ID, or None."""
        return self.db.get_category_by_id(category_id)

    def create_category(self, name: str, description: str = "") -> Optional[str]:
        """
        Create a new category.
        Returns the new category_id or None on failure.
        """
        name = name.strip()
        if not name:
            raise ValueError("Category name cannot be empty")
        return self.db.create_category(name, description)

    def update_category(
        self,
        category_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Update name and/or description of an existing category."""
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Category name cannot be empty")
        return self.db.update_category(category_id, name=name, description=description)

    def delete_category(self, category_id: str) -> Tuple[bool, str]:
        """
        Delete a category.
        Refuses to delete the 'default' category.
        Does NOT cascade-delete analyses in that category.
        """
        return self.db.delete_category(category_id)

    def usage_count(self, category_id: str) -> int:
        """Return how many analyses reference this category."""
        return self.db.count_analyses_in_category(category_id)
