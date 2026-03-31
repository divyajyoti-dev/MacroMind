"""
Constraint-aware reranker.
Scores retrieved recipes against user constraints and returns top-k.
"""
import difflib
from dataclasses import dataclass

from src.retriever import SearchResult


@dataclass
class UserConstraints:
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    budget_usd: float
    available_ingredients: list[str]
    dietary_tags: list[str]
    # Scoring weights (must sum to 1.0)
    w_macro: float  = 0.50
    w_budget: float = 0.30
    w_waste: float  = 0.20


@dataclass
class RankedResult:
    recipe_id: str
    name: str
    score: float            # higher is better
    macro_deviation: float  # MAPE across 4 macros, [0, 1]
    cost_overshoot: float   # fraction over budget (0 if under budget)
    waste_fraction: float   # fraction of ingredients not available
    metadata: dict


def macro_deviation_score(metadata: dict, constraints: UserConstraints) -> float:
    """
    Mean absolute percentage error (MAPE) between recipe macros and targets.
    Clamped to [0, 1].
    """
    targets = {
        "calories": constraints.calories,
        "protein":  constraints.protein_g,
        "carbs":    constraints.carbs_g,
        "fat":      constraints.fat_g,
    }
    recipe = {
        "calories": metadata.get("calories", 0),
        "protein":  metadata.get("protein", 0),
        "carbs":    metadata.get("carbs", 0),
        "fat":      metadata.get("fat", 0),
    }
    errors = []
    for key, target in targets.items():
        if target > 0:
            err = abs(recipe[key] - target) / target
            errors.append(min(err, 1.0))

    return sum(errors) / len(errors) if errors else 0.5


def budget_overshoot_score(metadata: dict, constraints: UserConstraints) -> float:
    """Fraction by which the recipe exceeds the per-meal budget (0 if within budget)."""
    cost = metadata.get("cost_usd", 0.0)
    if constraints.budget_usd <= 0:
        return 0.0
    excess = max(0.0, cost - constraints.budget_usd)
    return min(excess / constraints.budget_usd, 1.0)


def ingredient_waste_fraction(metadata: dict, constraints: UserConstraints) -> float:
    """
    Fraction of recipe ingredients NOT in the user's available_ingredients list.
    Uses fuzzy matching so 'chicken' matches 'chicken breast'.
    """
    recipe_ings = [i.strip().lower() for i in metadata.get("ingredients", "").split("|") if i]
    if not recipe_ings:
        return 0.0
    if not constraints.available_ingredients:
        return 0.0

    available_lower = [a.lower() for a in constraints.available_ingredients]
    missing = 0
    for ing in recipe_ings:
        matches = difflib.get_close_matches(ing, available_lower, n=1, cutoff=0.6)
        if not matches:
            # Also check substring containment
            if not any(ing in a or a in ing for a in available_lower):
                missing += 1

    return missing / len(recipe_ings)


def dietary_tag_penalty(metadata: dict, constraints: UserConstraints) -> float:
    """Returns 1.0 if the recipe violates any required dietary tag, else 0.0."""
    if not constraints.dietary_tags:
        return 0.0
    recipe_tags = set(t.strip().lower() for t in metadata.get("tags", "").split("|"))
    for required_tag in constraints.dietary_tags:
        if required_tag.lower() not in recipe_tags:
            return 1.0
    return 0.0


def score_recipe(metadata: dict, constraints: UserConstraints) -> tuple[float, dict]:
    """
    Compute a scalar score for a recipe given user constraints.
    Returns (score, breakdown) where score is higher = better.
    The breakdown dict is useful for debugging and the evaluation notebook.
    """
    md = macro_deviation_score(metadata, constraints)
    bs = budget_overshoot_score(metadata, constraints)
    wf = ingredient_waste_fraction(metadata, constraints)
    dp = dietary_tag_penalty(metadata, constraints)

    # Hard penalty for dietary violations (overrides numeric score)
    if dp > 0:
        return -999.0, {"macro_dev": md, "budget": bs, "waste": wf, "tag_violation": True}

    penalty = (
        constraints.w_macro  * md +
        constraints.w_budget * bs +
        constraints.w_waste  * wf
    )
    score = 1.0 - penalty

    return score, {
        "macro_dev": round(md, 4),
        "budget_overshoot": round(bs, 4),
        "waste_fraction": round(wf, 4),
        "tag_violation": False,
    }


def rerank(
    search_results: list[SearchResult],
    constraints: UserConstraints,
    top_k: int = 5,
) -> list[RankedResult]:
    """
    Score and re-sort retrieved recipes by constraint alignment.
    Returns top-k RankedResult objects.
    """
    ranked = []
    for sr in search_results:
        score, breakdown = score_recipe(sr.metadata, constraints)
        ranked.append(
            RankedResult(
                recipe_id=sr.recipe_id,
                name=sr.name,
                score=score,
                macro_deviation=breakdown["macro_dev"],
                cost_overshoot=breakdown.get("budget_overshoot", 0),
                waste_fraction=breakdown.get("waste_fraction", 0),
                metadata=sr.metadata,
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_k]
