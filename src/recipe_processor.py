"""
Recipe loading, macro computation, text preparation, and ChromaDB indexing.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config import RECIPES_PATH
from src.data_pipeline import NutritionFacts


@dataclass
class Recipe:
    id: str
    name: str
    ingredients: list[dict]   # [{"name": str, "grams": float}, ...]
    instructions: str
    tags: list[str]
    servings: int
    prep_time_min: int
    estimated_cost_usd: float = 0.0


@dataclass
class MacroProfile:
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    cost_usd: float


def load_recipes(path: Path = RECIPES_PATH) -> list[Recipe]:
    """Load recipes from JSON file."""
    with open(path) as f:
        raw = json.load(f)
    return [Recipe(**r) for r in raw]


def recipe_to_document(recipe: Recipe) -> str:
    """
    Convert a recipe to a single string for embedding.
    Natural language form produces better semantic search results than raw JSON.
    """
    ingredient_names = ", ".join(i["name"] for i in recipe.ingredients)
    tags = ", ".join(recipe.tags) if recipe.tags else "none"
    return (
        f"{recipe.name}. "
        f"Ingredients: {ingredient_names}. "
        f"Tags: {tags}. "
        f"Prep time: {recipe.prep_time_min} minutes. "
        f"Estimated cost: ${recipe.estimated_cost_usd:.2f} per serving."
    )


def compute_recipe_macros(
    recipe: Recipe,
    nutrition_cache: dict[str, NutritionFacts],
) -> MacroProfile:
    """
    Compute total macros for a recipe by summing per-ingredient contributions.
    Falls back to the recipe's pre-computed cost if ingredient lookup fails.
    """
    calories = protein = carbs = fat = cost = 0.0

    for ing in recipe.ingredients:
        name = ing["name"].lower()
        grams = ing.get("grams", 100.0)
        ratio = grams / 100.0

        nf = nutrition_cache.get(name)
        if nf is None:
            # Try partial match
            for key, val in nutrition_cache.items():
                if key in name or name in key:
                    nf = val
                    break

        if nf:
            calories += nf.calories_per_100g * ratio
            protein  += nf.protein_per_100g  * ratio
            carbs    += nf.carbs_per_100g    * ratio
            fat      += nf.fat_per_100g      * ratio
            cost     += nf.price_per_100g    * ratio

    if cost == 0.0:
        cost = recipe.estimated_cost_usd

    return MacroProfile(
        calories=round(calories, 1),
        protein_g=round(protein, 1),
        carbs_g=round(carbs, 1),
        fat_g=round(fat, 1),
        cost_usd=round(cost, 2),
    )


def build_chroma_index(
    recipes: list[Recipe],
    collection: Any,
    embedding_model: Any,
    nutrition_cache: dict | None = None,
) -> None:
    """
    Embed all recipes and upsert into a ChromaDB collection.
    Metadata stored: name, macros, cost, tags, ingredients list.
    """
    if nutrition_cache is None:
        nutrition_cache = {}

    documents, ids, metadatas = [], [], []

    for recipe in tqdm(recipes, desc="Indexing recipes"):
        doc = recipe_to_document(recipe)
        macros = compute_recipe_macros(recipe, nutrition_cache)

        metadata = {
            "recipe_id":    recipe.id,
            "name":         recipe.name,
            "calories":     macros.calories,
            "protein":      macros.protein_g,
            "carbs":        macros.carbs_g,
            "fat":          macros.fat_g,
            "cost_usd":     macros.cost_usd,
            "tags":         "|".join(recipe.tags),
            "ingredients":  "|".join(i["name"] for i in recipe.ingredients),
            "prep_time":    recipe.prep_time_min,
        }
        documents.append(doc)
        ids.append(recipe.id)
        metadatas.append(metadata)

    # Embed in one batch for efficiency
    embeddings = embedding_model.encode(documents, show_progress_bar=True).tolist()

    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"Indexed {len(recipes)} recipes into ChromaDB.")
