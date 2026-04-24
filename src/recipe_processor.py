import ast
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from src.config import RECIPES_PATH, CLEANED_RECIPES_PATH, DATA_DIR


@dataclass
class Recipe:
    id: str
    name: str
    ingredients: list[dict[str, Any]]  # [{name: str, grams: float}]
    instructions: str
    tags: list[str]
    servings: int
    prep_time_min: int
    estimated_cost_usd: float = 0.0


# ── Ingredient name parser ────────────────────────────────────────────────────

_LEADING_QTY_RE = re.compile(
    r"^\s*\d+\s*(?:\d+\s*/\s*\d+|\d*\s*/\s*\d+)?\s*"
)
_LEADING_UNIT_RE = re.compile(
    r"^(?:cups?|tbsps?|tablespoons?|tsps?|teaspoons?|oz|ounces?|lbs?|pounds?|"
    r"grams?|g\b|kgs?|ml\b|liters?|litres?|cans?|packages?|pkgs?|"
    r"cloves?|slices?|pieces?|stalks?|sprigs?|bunches?|heads?|"
    r"pinch(?:es)?|dash(?:es)?|handfuls?|sticks?|quarts?|pints?)\s*",
    re.IGNORECASE,
)


def parse_ingredient_name(raw: str) -> str:
    """Strip quantity, unit, and preparation notes from a raw RecipeNLG ingredient string."""
    s = re.sub(r"\(.*?\)", "", raw)      # remove parentheticals like "(optional)"
    s = re.sub(r"[^\w\s/]", " ", s)     # punctuation -> space, keep / for fractions
    s = _LEADING_QTY_RE.sub("", s)      # strip leading quantity (1, 1/2, 1 1/2)
    s = _LEADING_UNIT_RE.sub("", s)     # strip leading unit word
    s = re.sub(r"[^a-zA-Z\s]", " ", s)  # strip any remaining non-alpha
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


# ── Heuristic tagger ──────────────────────────────────────────────────────────

_MEAT_FISH = {
    "chicken", "beef", "pork", "lamb", "turkey", "salmon", "tuna", "shrimp",
    "bacon", "ham", "sausage", "anchovy", "cod", "tilapia", "crab", "lobster",
    "veal", "duck", "venison", "bison", "prawn", "sardine", "mackerel", "herring",
    "halibut", "trout", "mussels", "clam", "oyster", "scallop", "squid",
}
_DAIRY_EGG = {
    "milk", "cheese", "butter", "cream", "yogurt", "egg", "ghee", "whey",
    "mozzarella", "parmesan", "cheddar", "ricotta", "brie", "feta", "gouda",
    "buttermilk", "casein",
}
_GLUTEN = {
    "flour", "bread", "pasta", "wheat", "barley", "rye", "breadcrumb",
    "noodle", "tortilla", "couscous", "semolina", "spelt", "farro",
    "cracker", "biscuit", "muffin", "cake", "cookie",
}


def tag_recipe_heuristic(recipe: "Recipe", macros: dict) -> list[str]:
    """Infer dietary tags from parsed ingredient names and computed macros."""
    words: set[str] = set()
    for ing in recipe.ingredients:
        words.update(ing["name"].lower().split())

    tags: list[str] = []
    has_meat = bool(words & _MEAT_FISH)
    has_dairy = bool(words & _DAIRY_EGG)
    has_gluten = bool(words & _GLUTEN)

    if not has_meat:
        tags.append("vegetarian")
    if not has_meat and not has_dairy:
        tags.append("vegan")
    if not has_gluten:
        tags.append("gluten-free")
    if macros.get("carbs", 999) < 50 and not has_gluten:
        tags.append("keto")

    servings = max(recipe.servings or 2, 1)
    cost_per_serving = macros.get("cost_usd", 0) / servings
    if 0 < cost_per_serving < 4.0:
        tags.append("budget")

    if macros.get("protein", 0) > 25:
        tags.append("high-protein")

    return tags


# ── TheMealDB loader ──────────────────────────────────────────────────────────

_THEMEALDB_CACHE = DATA_DIR / "themealdb_recipes.json"
_THEMEALDB_BASE = "https://www.themealdb.com/api/json/v1/1"


def load_themealdb_recipes() -> list["Recipe"]:
    """
    Fetch recipes from TheMealDB free API (no key required).
    Results are cached locally in data/themealdb_recipes.json.
    Returns a list of Recipe objects with cuisine area as a tag.
    """
    if _THEMEALDB_CACHE.exists():
        raw_meals = json.loads(_THEMEALDB_CACHE.read_text())
    else:
        areas_resp = requests.get(f"{_THEMEALDB_BASE}/list.php?a=list", timeout=10)
        areas = [a["strArea"] for a in areas_resp.json().get("meals", [])]
        raw_meals = []
        for area in areas:
            resp = requests.get(f"{_THEMEALDB_BASE}/filter.php?a={area}", timeout=10)
            meal_list = resp.json().get("meals") or []
            for stub in meal_list:
                detail_resp = requests.get(
                    f"{_THEMEALDB_BASE}/lookup.php?i={stub['idMeal']}", timeout=10
                )
                detail = (detail_resp.json().get("meals") or [{}])[0]
                detail["_area"] = area
                raw_meals.append(detail)
                time.sleep(0.1)
        _THEMEALDB_CACHE.write_text(json.dumps(raw_meals, indent=2))
        print(f"TheMealDB: fetched {len(raw_meals)} recipes across {len(areas)} areas")

    recipes: list[Recipe] = []
    for i, m in enumerate(raw_meals):
        area = m.get("_area", "")
        category = m.get("strCategory", "")
        tags_raw = [t.strip() for t in (m.get("strTags") or "").split(",") if t.strip()]
        tags = list({area.lower(), category.lower()} | {t.lower() for t in tags_raw} - {""})

        ingredients = []
        for idx in range(1, 21):
            ing_name = (m.get(f"strIngredient{idx}") or "").strip()
            if not ing_name:
                break
            ingredients.append({"name": ing_name.lower(), "grams": 100.0})

        recipes.append(
            Recipe(
                id=f"mdb_{m.get('idMeal', i):>08}",
                name=m.get("strMeal", "").strip(),
                ingredients=ingredients,
                instructions=(m.get("strInstructions") or "").strip(),
                tags=tags,
                servings=2,
                prep_time_min=30,
                estimated_cost_usd=0.0,
            )
        )
    return recipes


# ── Core loaders ──────────────────────────────────────────────────────────────

def load_recipes(path: Path = RECIPES_PATH) -> list[Recipe]:
    raw = json.loads(path.read_text())
    return [
        Recipe(
            id=r["id"],
            name=r["name"],
            ingredients=r["ingredients"],
            instructions=r.get("instructions", ""),
            tags=r.get("tags", []),
            servings=r.get("servings", 2),
            prep_time_min=r.get("prep_time_min", 30),
            estimated_cost_usd=r.get("estimated_cost_usd", 0.0),
        )
        for r in raw
    ]


def _parse_directions(directions_raw) -> str:
    if isinstance(directions_raw, list):
        return " ".join(directions_raw)
    if isinstance(directions_raw, str):
        try:
            parsed = ast.literal_eval(directions_raw)
            if isinstance(parsed, list):
                return " ".join(str(s) for s in parsed)
        except (ValueError, SyntaxError):
            pass
        return directions_raw
    return ""


def load_cleaned_recipes(path: Path = CLEANED_RECIPES_PATH) -> list[Recipe]:
    raw = json.loads(path.read_text())
    recipes = []
    for i, r in enumerate(raw):
        ingredients = [
            {"name": parse_ingredient_name(ing), "grams": 100.0}
            for ing in r.get("ingredients", [])
            if ing.strip()
        ]
        instructions = _parse_directions(r.get("directions", ""))
        recipes.append(
            Recipe(
                id=f"nlg_{i:06d}",
                name=r.get("title", "").strip(),
                ingredients=ingredients,
                instructions=instructions,
                tags=[],
                servings=2,
                prep_time_min=30,
                estimated_cost_usd=0.0,
            )
        )
    return recipes


# ── Macro computation ─────────────────────────────────────────────────────────

def _cache_lookup(name: str, nutrition_cache: dict[str, dict]) -> dict | None:
    """Exact match, then progressive word-drop from the right until a cache hit."""
    if name in nutrition_cache:
        return nutrition_cache[name]
    words = name.split()
    for drop in range(1, len(words)):
        key = " ".join(words[:-drop])
        if key in nutrition_cache:
            return nutrition_cache[key]
    return None


def compute_recipe_macros(
    recipe: Recipe,
    nutrition_cache: dict[str, dict],
) -> dict[str, float]:
    totals = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0, "cost_usd": 0.0}
    for ing in recipe.ingredients:
        name = ing["name"].lower().strip()
        grams = ing.get("grams", 100.0)
        scale = grams / 100.0
        facts = _cache_lookup(name, nutrition_cache)
        if facts is None:
            continue
        totals["calories"] += facts.get("calories_per_100g", 0.0) * scale
        totals["protein"] += facts.get("protein_per_100g", 0.0) * scale
        totals["fat"] += facts.get("fat_per_100g", 0.0) * scale
        totals["carbs"] += facts.get("carbs_per_100g", 0.0) * scale
        totals["cost_usd"] += facts.get("price_per_100g", 0.0) * scale
    return totals


def recipe_to_document(recipe: Recipe) -> str:
    ing_names = ", ".join(i["name"] for i in recipe.ingredients[:10])
    tags_part = f" Tags: {', '.join(recipe.tags)}." if recipe.tags else ""
    return (
        f"{recipe.name}: {ing_names}."
        f"{tags_part}"
        f" Serves {recipe.servings}, {recipe.prep_time_min} min."
    )


# ── Indexing ──────────────────────────────────────────────────────────────────

def build_chroma_index(
    recipes: list[Recipe],
    collection,
    embedding_model,
    nutrition_cache: dict[str, dict],
    batch_size: int = 256,
) -> None:
    # Compute macros first so heuristic tagger has real numbers to work with.
    # Tags are injected before document text is built so embeddings include them.
    enriched: list[tuple[Recipe, dict]] = []
    for recipe in recipes:
        macros = compute_recipe_macros(recipe, nutrition_cache)
        if not recipe.tags:
            recipe.tags = tag_recipe_heuristic(recipe, macros)
        enriched.append((recipe, macros))

    documents = [recipe_to_document(r) for r, _ in enriched]
    embeddings = embedding_model.encode(
        documents, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    for (recipe, macros), doc, emb in tqdm(
        zip(enriched, documents, embeddings), total=len(enriched), desc="Upserting to ChromaDB"
    ):
        collection.upsert(
            ids=[recipe.id],
            embeddings=[emb],
            documents=[doc],
            metadatas=[{
                "recipe_id": recipe.id,
                "name": recipe.name,
                "calories": round(macros["calories"], 1),
                "protein": round(macros["protein"], 1),
                "carbs": round(macros["carbs"], 1),
                "fat": round(macros["fat"], 1),
                "cost_usd": round(macros["cost_usd"] or recipe.estimated_cost_usd, 2),
                "tags": "|".join(recipe.tags),
                "ingredients": "|".join(i["name"] for i in recipe.ingredients[:20]),
                "prep_time_min": recipe.prep_time_min,
            }],
        )
