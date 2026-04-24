#!/usr/bin/env python
"""
Embed all recipes and upsert them into ChromaDB.

Usage:
    python ingest.py                              # index both sources
    python ingest.py --source sample              # 65 curated recipes (fast)
    python ingest.py --source cleaned             # 9,999 RecipeNLG recipes
    python ingest.py --source themealdb           # ~300 TheMealDB global recipes
    python ingest.py --reset --source both        # drop collection and rebuild
    python ingest.py --reset --source both --prefetch  # also expand USDA cache
"""
import argparse
import os
from collections import Counter

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL
from src.data_pipeline import load_cache, batch_fetch_ingredients
from src.recipe_processor import (
    load_recipes, load_cleaned_recipes, load_themealdb_recipes,
    build_chroma_index, Recipe,
)
from src.retriever import get_or_create_collection


def _merge_dedup(lists: list[list[Recipe]]) -> list[Recipe]:
    seen: set[str] = set()
    merged: list[Recipe] = []
    for recipes in lists:
        for r in recipes:
            if r.id not in seen:
                seen.add(r.id)
                merged.append(r)
    return merged


def _prefetch_common_ingredients(recipes: list[Recipe], existing_cache: dict, top_n: int = 100) -> None:
    """Batch-fetch the most common parsed ingredient names not yet in the USDA cache."""
    if not os.getenv("USDA_API_KEY"):
        print("Skipping prefetch: no USDA_API_KEY set.")
        return
    freq = Counter(
        ing["name"]
        for r in recipes
        for ing in r.ingredients
        if ing["name"]
    )
    to_fetch = [name for name, _ in freq.most_common(top_n * 2) if name not in existing_cache][:top_n]
    if not to_fetch:
        print("Prefetch: all top ingredients already cached.")
        return
    print(f"Prefetching {len(to_fetch)} common ingredients into USDA cache...")
    batch_fetch_ingredients(to_fetch, api_key=os.getenv("USDA_API_KEY", ""))
    print("Prefetch complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection")
    parser.add_argument(
        "--source",
        choices=["sample", "cleaned", "themealdb", "both", "all"],
        default="both",
        help="Which recipe corpus to index (both = sample+cleaned, all = sample+cleaned+themealdb)",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Pre-fetch top-100 common ingredient names into USDA cache before indexing",
    )
    args = parser.parse_args()

    client, collection = get_or_create_collection()

    if args.reset:
        from src.config import CHROMA_COLLECTION_NAME
        client.delete_collection(CHROMA_COLLECTION_NAME)
        _, collection = get_or_create_collection()
        print("Collection reset.")

    if args.source == "sample":
        recipes = load_recipes()
    elif args.source == "cleaned":
        recipes = load_cleaned_recipes()
    elif args.source == "themealdb":
        recipes = load_themealdb_recipes()
    elif args.source == "both":
        recipes = _merge_dedup([load_recipes(), load_cleaned_recipes()])
    else:  # all
        recipes = _merge_dedup([load_recipes(), load_cleaned_recipes(), load_themealdb_recipes()])

    nutrition_cache = load_cache()

    if args.prefetch:
        _prefetch_common_ingredients(recipes, nutrition_cache)
        nutrition_cache = load_cache()  # reload after prefetch

    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Indexing {len(recipes)} recipes ({args.source}) ...")
    build_chroma_index(recipes, collection, model, nutrition_cache)
    print(f"Done. {collection.count()} recipes indexed.")


if __name__ == "__main__":
    main()
