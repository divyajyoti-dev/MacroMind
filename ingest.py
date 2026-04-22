#!/usr/bin/env python
"""
Embed all recipes and upsert them into ChromaDB.

Usage:
    python ingest.py                         # index both sources
    python ingest.py --source sample         # 65 curated recipes (fast)
    python ingest.py --source cleaned        # 9,999 RecipeNLG recipes
    python ingest.py --reset --source both   # drop collection and rebuild
"""
import argparse

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL
from src.data_pipeline import load_cache
from src.recipe_processor import load_recipes, load_cleaned_recipes, build_chroma_index, Recipe
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection")
    parser.add_argument(
        "--source",
        choices=["sample", "cleaned", "both"],
        default="both",
        help="Which recipe corpus to index",
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
    else:
        recipes = _merge_dedup([load_recipes(), load_cleaned_recipes()])

    nutrition_cache = load_cache()
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Indexing {len(recipes)} recipes ({args.source}) ...")
    build_chroma_index(recipes, collection, model, nutrition_cache)
    print(f"Done. {collection.count()} recipes indexed.")


if __name__ == "__main__":
    main()
