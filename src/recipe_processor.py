import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import RECIPES_PATH, CLEANED_RECIPES_PATH


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
            {"name": ing.strip(), "grams": 100.0}
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


def compute_recipe_macros(
    recipe: Recipe,
    nutrition_cache: dict[str, dict],
) -> dict[str, float]:
    totals = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0, "cost_usd": 0.0}
    for ing in recipe.ingredients:
        name = ing["name"].lower().strip()
        grams = ing.get("grams", 100.0)
        scale = grams / 100.0
        facts = nutrition_cache.get(name)
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
