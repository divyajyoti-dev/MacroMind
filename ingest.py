"""
MacroMind — Ingest Pipeline
============================
Loads recipes_sample.csv, estimates macros for each recipe by matching
ingredient NER tokens against the USDA common nutrition lookup, embeds each
recipe (title + ingredients text) using sentence-transformers, and stores
everything in ChromaDB.

Usage:
    python ingest.py                  # uses paths from config.py
    python ingest.py --reset          # drops & recreates the ChromaDB collection

Pipeline steps:
    1. Load recipes CSV  →  list of recipe dicts
    2. For each recipe   →  parse NER tokens, look up macros per ingredient,
                           estimate total recipe macros (sum across all servings)
    3. Build embedding text: "{title}. Ingredients: {ingredients_text}"
    4. Batch-embed with sentence-transformers (all-MiniLM-L6-v2)
    5. Upsert into ChromaDB with metadata (macros, tags, ingredient list)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

import config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── USDA Lookup ───────────────────────────────────────────────────────────────

def load_usda_lookup(path: str) -> dict[str, dict]:
    """Load the simplified USDA JSON mapping ingredient names → macros per 100g."""
    with open(path, "r") as f:
        data = json.load(f)
    # Build a lowercased key map for fuzzy matching
    return {k.lower(): v for k, v in data.items()}


def find_ingredient_macros(
    ingredient_token: str,
    usda: dict[str, dict],
) -> dict[str, float] | None:
    """
    Try to match an ingredient token against the USDA lookup.

    Strategy:
      1. Exact match.
      2. Check if any USDA key is a substring of the token.
      3. Check if the token is a substring of any USDA key.

    Returns macros dict or None if no match found.
    """
    token = _normalise(ingredient_token)
    if token in usda:
        return usda[token]
    for key, macros in usda.items():
        if key in token or token in key:
            return macros
    return None


def estimate_recipe_macros(
    ner_tokens: list[str],
    usda: dict[str, dict],
    assumed_grams_per_ingredient: float = 100.0,
) -> dict[str, float]:
    """
    Sum macros for each matched NER ingredient.

    We assume ~100g per ingredient as a crude default when no quantity is parsed.
    For a production system, you'd parse quantities from the ingredient strings.
    Each USDA entry is per-100g, so values scale linearly.

    Returns dict with keys: calories, protein, carbs, fat.
    """
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    matched = 0
    for token in ner_tokens:
        macros = find_ingredient_macros(token, usda)
        if macros:
            scale = assumed_grams_per_ingredient / 100.0
            totals["calories"] += macros["calories"] * scale
            totals["protein"]  += macros["protein"]  * scale
            totals["carbs"]    += macros["carbs"]     * scale
            totals["fat"]      += macros["fat"]       * scale
            matched += 1
    # Round to 1 decimal place
    return {k: round(v, 1) for k, v in totals.items()}


# ── Dietary Tag Detection ─────────────────────────────────────────────────────

_MEAT_TOKENS = {
    "chicken", "beef", "turkey", "pork", "lamb", "salmon", "shrimp",
    "tuna", "tilapia", "bacon", "sausage", "ham",
}
_ANIMAL_TOKENS = _MEAT_TOKENS | {"eggs", "egg", "milk", "cheese", "yogurt", "butter"}
_GLUTEN_TOKENS = {"flour", "pasta", "bread", "wheat", "soy sauce", "breadcrumbs"}

def detect_tags(ner_tokens: list[str]) -> list[str]:
    """
    Infer dietary tags from ingredient list.

    Returns subset of: ["vegetarian", "vegan", "gluten-free"]
    """
    tokens_lower = {_normalise(t) for t in ner_tokens}
    tags = []
    has_meat   = bool(tokens_lower & _MEAT_TOKENS)
    has_animal = bool(tokens_lower & _ANIMAL_TOKENS)
    has_gluten = any(g in t for t in tokens_lower for g in _GLUTEN_TOKENS)

    if not has_meat:
        tags.append("vegetarian")
    if not has_animal:
        tags.append("vegan")
    if not has_gluten:
        tags.append("gluten-free")
    return tags


# ── Embedding Text Builder ────────────────────────────────────────────────────

def build_embedding_text(title: str, ingredients: str, ner: list[str]) -> str:
    """
    Create a rich text representation for embedding.

    Combines title + pipe-separated ingredients + NER tokens so the embedding
    captures both culinary phrasing and canonical ingredient names.
    """
    ner_str = ", ".join(ner) if ner else ""
    ingredients_clean = ingredients.replace("|", ", ")
    return f"{title}. Ingredients: {ingredients_clean}. Key items: {ner_str}"


# ── Main Ingestion ────────────────────────────────────────────────────────────

def load_recipes(csv_path: str) -> list[dict[str, Any]]:
    """Parse the recipes CSV into a list of standardised dicts."""
    df = pd.read_csv(csv_path)
    required_cols = {"title", "ingredients", "directions", "NER"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    recipes = []
    for _, row in df.iterrows():
        ner_raw = str(row.get("NER", ""))
        ner_tokens = [t.strip() for t in ner_raw.split("|") if t.strip()]
        recipes.append({
            "title":       str(row["title"]),
            "ingredients": str(row["ingredients"]),
            "directions":  str(row["directions"]),
            "ner":         ner_tokens,
        })
    log.info("Loaded %d recipes from %s", len(recipes), csv_path)
    return recipes


def run_ingestion(reset: bool = False) -> None:
    t_start = time.time()

    # 1. Load data
    usda = load_usda_lookup(config.USDA_JSON)
    log.info("USDA lookup loaded: %d ingredients", len(usda))

    recipes = load_recipes(config.RECIPES_CSV)

    # 2. Estimate macros + tags for each recipe
    for r in recipes:
        r["estimated_macros"] = estimate_recipe_macros(r["ner"], usda)
        r["tags"] = detect_tags(r["ner"])

    # 3. Init ChromaDB
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    if reset and config.COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(config.COLLECTION_NAME)
        log.info("Dropped existing collection '%s'", config.COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Load embedding model
    log.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    # 5. Build texts for embedding
    texts = [
        build_embedding_text(r["title"], r["ingredients"], r["ner"])
        for r in recipes
    ]

    # 6. Embed in one batch
    log.info("Embedding %d recipes …", len(texts))
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    # 7. Upsert into ChromaDB
    ids        = [f"recipe_{i}" for i in range(len(recipes))]
    documents  = texts
    metadatas  = []
    for r in recipes:
        macros = r["estimated_macros"]
        meta = {
            "title":      r["title"],
            "ingredients": r["ingredients"],
            "ner":        "|".join(r["ner"]),
            "tags":       "|".join(r["tags"]),
            "calories":   macros["calories"],
            "protein":    macros["protein"],
            "carbs":      macros["carbs"],
            "fat":        macros["fat"],
            "estimated_cost": 0.0,   # placeholder — extend with price API
        }
        metadatas.append(meta)

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    elapsed = time.time() - t_start
    log.info(
        "Ingestion complete. %d recipes indexed in ChromaDB in %.1fs",
        len(recipes),
        elapsed,
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MacroMind data ingestion pipeline")
    p.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the ChromaDB collection before ingesting.",
    )
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    args = _parse_args()
    run_ingestion(reset=args.reset)
