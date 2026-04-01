"""
MacroMind Scoring Module
========================
Reranking scoring function for RAG candidates.

Formula:
    score = -W1 * macro_deviation - W2 * budget_overshoot + W3 * ingredient_overlap

Where:
    macro_deviation   = normalised sum of absolute differences between recipe macros
                        and user target macros (each dimension divided by target so
                        the scale is unit-free and dimensions are comparable).
    budget_overshoot  = max(0, estimated_cost - budget) / budget  (0 if no budget set)
    ingredient_overlap = fraction of recipe NER ingredients already in user's pantry

All components are in [0, ∞) before weighting, so a perfect recipe scores near 0
(no deviation, no overshoot) plus a positive ingredient-overlap bonus.
"""

from __future__ import annotations

import re
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def score_recipe(
    recipe: dict[str, Any],
    user_constraints: dict[str, Any],
    weights: dict[str, float],
) -> float:
    """
    Compute a scalar score for a single recipe given user constraints.

    Args:
        recipe: Dict with keys:
            - "estimated_macros": {"calories", "protein", "carbs", "fat"}
            - "ner": list[str] of ingredient name tokens
            - "estimated_cost": float (optional, 0 if unknown)
        user_constraints: Dict with keys:
            - "calories", "protein", "carbs", "fat": numeric targets
            - "budget": float or None
            - "available_ingredients": list[str]
        weights: Dict with keys "w1", "w2", "w3".

    Returns:
        float — higher is better.
    """
    macro_dev = _macro_deviation(recipe, user_constraints)
    budget_os = _budget_overshoot(recipe, user_constraints)
    ingr_ov   = _ingredient_overlap(recipe, user_constraints)

    score = (
        -weights.get("w1", 1.0) * macro_dev
        - weights.get("w2", 0.5) * budget_os
        + weights.get("w3", 0.3) * ingr_ov
    )
    return score


def rerank(
    candidates: list[dict[str, Any]],
    user_constraints: dict[str, Any],
    weights: dict[str, float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Score and sort candidates, returning the top_k highest-scoring recipes.
    Mutates each candidate dict by adding a "score" key.

    Args:
        candidates: List of recipe dicts (from ChromaDB query).
        user_constraints: User's targets and pantry.
        weights: Scoring weights dict.
        top_k: How many top results to return.

    Returns:
        List of up to top_k recipe dicts sorted descending by score.
    """
    for recipe in candidates:
        recipe["score"] = score_recipe(recipe, user_constraints, weights)
    ranked = sorted(candidates, key=lambda r: r["score"], reverse=True)
    return ranked[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Component functions (exposed for unit testing / debugging)
# ─────────────────────────────────────────────────────────────────────────────

def _macro_deviation(
    recipe: dict[str, Any],
    user_constraints: dict[str, Any],
) -> float:
    """
    Normalised macro deviation.

    For each macro dimension d in {calories, protein, carbs, fat}:
        deviation_d = |recipe_d - target_d| / max(target_d, 1)

    Returns the average across the four dimensions (a value ≥ 0; perfect = 0).
    """
    macros = recipe.get("estimated_macros", {})
    dims = ["calories", "protein", "carbs", "fat"]
    deviations = []
    for dim in dims:
        target = float(user_constraints.get(dim, 0) or 0)
        actual = float(macros.get(dim, 0) or 0)
        deviations.append(abs(actual - target) / max(target, 1.0))
    return sum(deviations) / len(deviations)


def _budget_overshoot(
    recipe: dict[str, Any],
    user_constraints: dict[str, Any],
) -> float:
    """
    Fractional budget overshoot.

    Returns max(0, (cost - budget) / budget), or 0 if no budget is set.
    """
    budget = user_constraints.get("budget")
    if not budget:
        return 0.0
    try:
        budget = float(budget)
    except (TypeError, ValueError):
        return 0.0
    if budget <= 0:
        return 0.0
    cost = float(recipe.get("estimated_cost", 0) or 0)
    return max(0.0, (cost - budget) / budget)


def _ingredient_overlap(
    recipe: dict[str, Any],
    user_constraints: dict[str, Any],
) -> float:
    """
    Fraction of recipe NER ingredients already in the user's available pantry.

    Matching is case-insensitive substring matching:
        pantry item "chicken" matches recipe token "chicken breast".

    Returns a value in [0, 1].
    """
    ner: list[str] = recipe.get("ner", [])
    if not ner:
        return 0.0

    available: list[str] = user_constraints.get("available_ingredients", [])
    available_lower = [_normalise(a) for a in available]

    matched = 0
    for ingredient in ner:
        ing_norm = _normalise(ingredient)
        for pantry_item in available_lower:
            if pantry_item in ing_norm or ing_norm in pantry_item:
                matched += 1
                break
    return matched / len(ner)


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
