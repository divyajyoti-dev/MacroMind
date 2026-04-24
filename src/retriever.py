from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from src.config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME


@dataclass
class SearchResult:
    recipe_id: str
    name: str
    score: float
    metadata: dict


@dataclass
class UserConstraints:
    calories: float = 2000.0
    protein_g: float = 100.0
    carbs_g: float = 250.0
    fat_g: float = 65.0
    budget_usd: float = 12.0
    available_ingredients: list[str] = None
    dietary_tags: list[str] = None
    allergy_tags: list[str] = None
    cultural_dietary: list[str] = None
    w_macro: float = 0.50
    w_budget: float = 0.30
    w_waste: float = 0.20

    def __post_init__(self):
        if self.available_ingredients is None:
            self.available_ingredients = []
        if self.dietary_tags is None:
            self.dietary_tags = []
        if self.allergy_tags is None:
            self.allergy_tags = []
        if self.cultural_dietary is None:
            self.cultural_dietary = []


def get_or_create_collection(
    persist_dir: Path = CHROMA_DB_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> tuple[chromadb.PersistentClient, chromadb.Collection]:
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def build_query_text(constraints: UserConstraints) -> str:
    parts = []

    if constraints.dietary_tags:
        parts.extend(constraints.dietary_tags)

    parts.append(f"{int(constraints.calories)} calorie meal")

    if constraints.protein_g >= 100:
        parts.append(f"high protein {int(constraints.protein_g)}g")

    if constraints.carbs_g < 50:
        parts.append("low carb keto")

    if constraints.budget_usd < 8:
        parts.append("budget cheap")

    if constraints.available_ingredients:
        parts.append("with " + " ".join(constraints.available_ingredients[:5]))

    return " ".join(parts)


def semantic_search(
    query: str,
    collection,
    embedding_model,
    n_results: int = 20,
) -> list[SearchResult]:
    query_embedding = embedding_model.encode(
        [query], normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    search_results = []
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for rid, dist, meta in zip(ids, distances, metadatas):
        search_results.append(
            SearchResult(
                recipe_id=rid,
                name=meta.get("name", ""),
                score=dist,
                metadata=meta,
            )
        )

    search_results.sort(key=lambda r: r.score)
    return search_results
