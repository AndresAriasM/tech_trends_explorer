"""
One-shot script: translate technology_name fields of the 5 mora queries
from Spanish to English so the Hype Cycle chart renders in English for
the paper submission.

Run with:
    cd backend && python -m scripts.translate_mora_names
"""

import sys
from pathlib import Path

# Make `backend/` importable when invoked as a module
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Load backend/.env so AWS credentials are available outside the FastAPI process
from dotenv import load_dotenv  # noqa: E402
load_dotenv(BACKEND_DIR / ".env")

from storage.dynamodb import get_db  # noqa: E402

# Spanish → English translation map.
# Keys must match the exact current technology_name values.
TRANSLATIONS = {
    "M1: Bebidas funcionales de mora":
        "M1: Functional beverages from blackberry",
    "M2: Bebidas funcionales + pasteurización":
        "M2: Functional beverages + pasteurization",
    "M3: Compuestos bioactivos":
        "M3: Bioactive compounds",
    "M4: Vida útil + pasteurización en frío":
        "M4: Shelf life + cold pasteurization",
    "M5: Vida útil postcosecha":
        "M5: Postharvest shelf life",
}

CATEGORY_ID = "cat_1776749725868_f7a182e4"  # "Chain Blackberry"


def main() -> int:
    db = get_db()
    items = db.get_analyses_by_category(CATEGORY_ID)
    hc_items = [i for i in items if i.get("analysis_type") == "hype_cycle"]
    print(f"Found {len(hc_items)} Hype Cycle items in category {CATEGORY_ID}")

    updated = 0
    for item in hc_items:
        current_name = item.get("technology_name", "")
        new_name = TRANSLATIONS.get(current_name)
        if not new_name:
            print(f"  [skip] {current_name!r} → no translation in map")
            continue

        item["technology_name"] = new_name
        ok = db.put_analysis_raw(item)
        if ok:
            updated += 1
            print(f"  [ok]   {current_name!r} → {new_name!r}")
        else:
            print(f"  [fail] {current_name!r} — put_analysis_raw returned False")

    print(f"\nTotal updated: {updated}/{len(hc_items)}")
    return 0 if updated == len(TRANSLATIONS) else 1


if __name__ == "__main__":
    sys.exit(main())
