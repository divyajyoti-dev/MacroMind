from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME


@dataclass
class SearchResult:
    recipe_id: str
    name: str
    score: float
    metadata: dict


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
