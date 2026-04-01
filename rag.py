"""
MacroMind — RAG Pipeline
========================
Handles retrieval, reranking, and LLM generation for all three experiment variants.

Variants:
    "baseline"   → skip retrieval entirely; Gemini invents meals from constraints.
    "rag"        → retrieve top-20 from ChromaDB, pass raw results to Gemini.
    "rag-rerank" → retrieve top-20, rerank with scoring.py, pass top-5 to Gemini.

Public entry point:
    result = run_pipeline(user_constraints, variant="rag-rerank")

Returns a PipelineResult dataclass containing the LLM response text, the
recipes used, token counts, latency, and per-variant metadata.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

import config
from prompts import (
    SYSTEM_MEAL_PLANNER,
    BASELINE_USER_PROMPT,
    RAG_USER_PROMPT,
    RAG_RERANK_USER_PROMPT,
    format_recipe_block,
)
from scoring import rerank

log = logging.getLogger(__name__)


# ── Result Container ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Returned by run_pipeline(); consumed by app.py and eval.py."""
    variant:           str
    response_text:     str
    recipes_used:      list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0
    latency_seconds:   float = 0.0
    retrieval_count:   int = 0   # how many docs came back from ChromaDB
    rerank_applied:    bool = False
    error:             str | None = None

    @property
    def estimated_cost_usd(self) -> float:
        input_cost  = (self.prompt_tokens     / 1000) * config.GROQ_INPUT_COST_PER_1K
        output_cost = (self.completion_tokens / 1000) * config.GROQ_OUTPUT_COST_PER_1K
        return round(input_cost + output_cost, 5)


# ── Lazy singletons (loaded once per process) ─────────────────────────────────

_embed_model: SentenceTransformer | None = None
_chroma_collection = None
_groq_client: Groq | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        log.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embed_model


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        _chroma_collection = client.get_collection(config.COLLECTION_NAME)
    return _chroma_collection


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at console.groq.com and paste it in the sidebar."
            )
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


# ── Query Builder ─────────────────────────────────────────────────────────────

def build_query_string(user_constraints: dict[str, Any]) -> str:
    """
    Build a keyword-rich semantic search query from user constraints.

    Example output:
      "high protein low carb dinner chicken broccoli gluten-free"
    """
    parts = []

    # Macro profile
    protein = user_constraints.get("protein", 0) or 0
    carbs   = user_constraints.get("carbs", 0)   or 0
    fat     = user_constraints.get("fat", 0)     or 0

    if protein > 100:
        parts.append("high protein")
    elif protein < 50:
        parts.append("low protein")

    if carbs < 100:
        parts.append("low carb")
    elif carbs > 250:
        parts.append("high carb")

    if fat < 50:
        parts.append("low fat")

    # Dietary restrictions
    restrictions = user_constraints.get("dietary_restrictions", []) or []
    if isinstance(restrictions, str):
        restrictions = [r.strip() for r in restrictions.split(",") if r.strip()]
    parts.extend(restrictions)

    # Available ingredients (add the top 4 most informative ones)
    available = user_constraints.get("available_ingredients", []) or []
    if isinstance(available, str):
        available = [a.strip() for a in available.split(",") if a.strip()]
    parts.extend(available[:4])

    query = " ".join(parts) if parts else "healthy balanced meal"
    log.debug("Built query string: %r", query)
    return query


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_candidates(
    query: str,
    top_k: int = 20,
    dietary_restrictions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Embed the query and retrieve top_k nearest recipes from ChromaDB.

    Returns a list of recipe dicts with keys:
        title, ingredients, ner, tags, estimated_macros, estimated_cost
    """
    model      = _get_embed_model()
    collection = _get_collection()

    query_embedding = model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "documents", "distances"],
    )

    candidates = []
    for meta, doc, dist in zip(
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        ner_tokens = [t for t in meta.get("ner", "").split("|") if t]
        recipe = {
            "title":       meta.get("title", "Unknown"),
            "ingredients": meta.get("ingredients", ""),
            "ner":         ner_tokens,
            "tags":        [t for t in meta.get("tags", "").split("|") if t],
            "estimated_macros": {
                "calories": meta.get("calories", 0),
                "protein":  meta.get("protein",  0),
                "carbs":    meta.get("carbs",    0),
                "fat":      meta.get("fat",      0),
            },
            "estimated_cost": meta.get("estimated_cost", 0.0),
            "cosine_distance": round(float(dist), 4),
        }
        candidates.append(recipe)

    # Filter by dietary restrictions if provided
    if dietary_restrictions:
        restrictions_lower = [r.lower().strip() for r in dietary_restrictions]
        filtered = _filter_by_restrictions(candidates, restrictions_lower)
        if filtered:  # only apply filter if it doesn't wipe all results
            candidates = filtered

    log.info("Retrieved %d candidates from ChromaDB", len(candidates))
    return candidates


def _filter_by_restrictions(
    candidates: list[dict],
    restrictions: list[str],
) -> list[dict]:
    """Filter out recipes whose tags don't include all required restrictions."""
    result = []
    for recipe in candidates:
        recipe_tags = [t.lower() for t in recipe.get("tags", [])]
        if all(r in recipe_tags for r in restrictions):
            result.append(recipe)
    return result


# ── LLM Call ──────────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
    """
    Call Groq (free tier) and return (response_text, usage_dict).
    usage_dict has keys: prompt_tokens, completion_tokens, total_tokens.
    """
    client = _get_groq()
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.4,
    )
    text  = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens":     response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens":      response.usage.total_tokens,
    }
    return text, usage


# ── Constraint Formatter ──────────────────────────────────────────────────────

def _fmt_constraints(user_constraints: dict[str, Any]) -> dict[str, str]:
    """Convert raw constraint values to clean display strings for prompts."""
    budget = user_constraints.get("budget")
    budget_str = f"${float(budget):.2f}" if budget else "no budget specified"

    restrictions = user_constraints.get("dietary_restrictions", [])
    if isinstance(restrictions, list):
        restrictions_str = ", ".join(restrictions) if restrictions else "none"
    else:
        restrictions_str = str(restrictions) if restrictions else "none"

    available = user_constraints.get("available_ingredients", [])
    if isinstance(available, list):
        available_str = ", ".join(available) if available else "nothing on hand"
    else:
        available_str = str(available) if available else "nothing on hand"

    return {
        "calories":              str(user_constraints.get("calories", 2000)),
        "protein":               str(user_constraints.get("protein", 150)),
        "carbs":                 str(user_constraints.get("carbs", 200)),
        "fat":                   str(user_constraints.get("fat", 65)),
        "budget":                budget_str,
        "dietary_restrictions":  restrictions_str,
        "available_ingredients": available_str,
    }


# ── Variant Runners ───────────────────────────────────────────────────────────

def _run_baseline(user_constraints: dict[str, Any]) -> PipelineResult:
    fmt = _fmt_constraints(user_constraints)
    user_prompt = BASELINE_USER_PROMPT.format(**fmt)

    t0 = time.time()
    text, usage = call_llm(SYSTEM_MEAL_PLANNER, user_prompt)
    latency = time.time() - t0

    return PipelineResult(
        variant="baseline",
        response_text=text,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        latency_seconds=round(latency, 2),
    )


def _run_rag(user_constraints: dict[str, Any]) -> PipelineResult:
    query      = build_query_string(user_constraints)
    candidates = retrieve_candidates(
        query,
        top_k=config.TOP_K_RETRIEVAL,
        dietary_restrictions=user_constraints.get("dietary_restrictions"),
    )

    fmt = _fmt_constraints(user_constraints)
    recipe_block = format_recipe_block(candidates, include_score=False)
    user_prompt  = RAG_USER_PROMPT.format(recipes=recipe_block, **fmt)

    t0 = time.time()
    text, usage = call_llm(SYSTEM_MEAL_PLANNER, user_prompt)
    latency = time.time() - t0

    return PipelineResult(
        variant="rag",
        response_text=text,
        recipes_used=candidates,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        latency_seconds=round(latency, 2),
        retrieval_count=len(candidates),
    )


def _run_rag_rerank(user_constraints: dict[str, Any]) -> PipelineResult:
    query      = build_query_string(user_constraints)
    candidates = retrieve_candidates(
        query,
        top_k=config.TOP_K_RETRIEVAL,
        dietary_restrictions=user_constraints.get("dietary_restrictions"),
    )

    weights = {
        "w1": config.W1_MACRO_DEVIATION,
        "w2": config.W2_BUDGET_OVERSHOOT,
        "w3": config.W3_INGREDIENT_OVERLAP,
    }
    top_recipes = rerank(candidates, user_constraints, weights, top_k=config.TOP_K_RERANK)

    fmt = _fmt_constraints(user_constraints)
    recipe_block = format_recipe_block(top_recipes, include_score=True)
    user_prompt  = RAG_RERANK_USER_PROMPT.format(recipes=recipe_block, **fmt)

    t0 = time.time()
    text, usage = call_llm(SYSTEM_MEAL_PLANNER, user_prompt)
    latency = time.time() - t0

    return PipelineResult(
        variant="rag-rerank",
        response_text=text,
        recipes_used=top_recipes,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        latency_seconds=round(latency, 2),
        retrieval_count=len(candidates),
        rerank_applied=True,
    )


# ── Public Entry Point ────────────────────────────────────────────────────────

def run_pipeline(
    user_constraints: dict[str, Any],
    variant: str | None = None,
) -> PipelineResult:
    """
    Run the full MacroMind pipeline for the given variant.

    Args:
        user_constraints: Dict with keys:
            calories, protein, carbs, fat, budget,
            dietary_restrictions (list[str]), available_ingredients (list[str])
        variant: "baseline" | "rag" | "rag-rerank"
                 Defaults to config.VARIANT if not specified.

    Returns:
        PipelineResult — response text + metadata.
    """
    if variant is None:
        variant = config.VARIANT

    log.info("Running pipeline: variant=%s", variant)

    try:
        if variant == "baseline":
            return _run_baseline(user_constraints)
        elif variant == "rag":
            return _run_rag(user_constraints)
        elif variant == "rag-rerank":
            return _run_rag_rerank(user_constraints)
        else:
            raise ValueError(f"Unknown variant: {variant!r}. Choose from baseline | rag | rag-rerank")
    except Exception as exc:
        log.error("Pipeline error [%s]: %s", variant, exc)
        return PipelineResult(
            variant=variant,
            response_text="",
            error=str(exc),
        )
