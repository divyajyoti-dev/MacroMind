"""
ChromaDB wrapper for semantic recipe retrieval.
"""
from dataclasses import dataclass
from typing import Any

import chromadb

from src.config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME


@dataclass
class SearchResult:
    recipe_id: str
    name: str
    score: float        # cosine distance (lower = more similar)
    metadata: dict


def get_or_create_collection(
    persist_dir: str = CHROMA_DB_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> chromadb.Collection:
    """Return (or create) a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def collection_is_populated(collection: chromadb.Collection) -> bool:
    return collection.count() > 0


def semantic_search(
    query: str,
    collection: chromadb.Collection,
    embedding_model: Any,
    n_results: int = 20,
) -> list[SearchResult]:
    """
    Encode the query and retrieve the top-n closest recipes from ChromaDB.
    Returns SearchResult objects sorted by cosine distance (ascending).
    """
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    search_results = []
    for rid, meta, dist in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        search_results.append(
            SearchResult(
                recipe_id=rid,
                name=meta.get("name", rid),
                score=dist,
                metadata=meta,
            )
        )
    return search_results


def build_query_text(
    macro_targets: dict,
    available_ingredients: list[str],
    dietary_tags: list[str],
    budget_usd: float | None = None,
) -> str:
    """
    Construct a natural-language query string from user constraints.
    This text form works better for semantic search than structured inputs.
    """
    parts = []
    if macro_targets.get("calories"):
        parts.append(f"{int(macro_targets['calories'])} calorie meal")
    if macro_targets.get("protein"):
        parts.append(f"high protein {int(macro_targets['protein'])}g protein")
    if dietary_tags:
        parts.append(" ".join(dietary_tags))
    if available_ingredients:
        parts.append("with " + ", ".join(available_ingredients[:5]))
    if budget_usd:
        parts.append(f"budget under ${budget_usd:.0f}")
    return " ".join(parts) if parts else "healthy balanced meal"
