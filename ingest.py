"""
MacroMind — Ingest Pipeline
============================
Builds the ChromaDB vector index from data/recipes/sample_recipes.json
using the team's src/ modules.

Usage:
    python ingest.py           # index recipes
    python ingest.py --reset   # drop & rebuild
"""
import argparse
import json
import logging

from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    USDA_CACHE_PATH,
)
from src.recipe_processor import load_recipes, build_chroma_index
from src.retriever import get_or_create_collection
from src.data_pipeline import NutritionFacts, load_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def run_ingestion(reset: bool = False) -> None:
    import chromadb

    # Load or reset collection
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    if reset and CHROMA_COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        log.info("Dropped collection '%s'", CHROMA_COLLECTION_NAME)

    collection = get_or_create_collection(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)

    # Load recipes
    recipes = load_recipes()
    log.info("Loaded %d recipes", len(recipes))

    # Load USDA nutrition cache
    raw_cache = load_cache(USDA_CACHE_PATH)
    nutrition_cache = {k: NutritionFacts.from_dict(v) for k, v in raw_cache.items()}
    log.info("Nutrition cache: %d ingredients", len(nutrition_cache))

    # Load embedding model
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Build index
    build_chroma_index(recipes, collection, model, nutrition_cache)
    log.info("Done. %d recipes indexed.", collection.count())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()
    run_ingestion(reset=args.reset)
