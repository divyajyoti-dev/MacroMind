import difflib
from dataclasses import dataclass

from src.retriever import UserConstraints, SearchResult


@dataclass
class RankedResult:
    recipe_id: str
    name: str
    score: float
    macro_deviation: float
    cost_overshoot: float
    waste_fraction: float
    metadata: dict


def macro_deviation_score(metadata: dict, constraints: UserConstraints) -> float:
    targets = {
        "calories": constraints.calories,
        "protein": constraints.protein_g,
        "carbs": constraints.carbs_g,
        "fat": constraints.fat_g,
    }
    deviations = []
    for macro, target in targets.items():
        if target == 0:
            continue
        recipe_val = float(metadata.get(macro, 0.0))
        deviations.append(abs(recipe_val - target) / target)

    if not deviations:
        return 0.0
    return min(sum(deviations) / len(deviations), 1.0)


def budget_overshoot_score(metadata: dict, constraints: UserConstraints) -> float:
    if constraints.budget_usd <= 0:
        return 0.0
    cost = float(metadata.get("cost_usd", 0.0))
    overshoot = max(0.0, cost - constraints.budget_usd) / constraints.budget_usd
    return min(overshoot, 1.0)


def ingredient_waste_fraction(metadata: dict, constraints: UserConstraints) -> float:
    if not constraints.available_ingredients:
        return 0.0
    recipe_ings = [i.strip().lower() for i in metadata.get("ingredients", "").split("|") if i.strip()]
    if not recipe_ings:
        return 0.0
    available = [a.lower() for a in constraints.available_ingredients]
    missing = 0
    for ing in recipe_ings:
        matches = difflib.get_close_matches(ing, available, n=1, cutoff=0.6)
        if not matches:
            missing += 1
    return missing / len(recipe_ings)


def dietary_tag_penalty(metadata: dict, constraints: UserConstraints) -> bool:
    if not constraints.dietary_tags:
        return False
    recipe_tags = {t.strip().lower() for t in metadata.get("tags", "").split("|") if t.strip()}
    for required in constraints.dietary_tags:
        if required.lower() not in recipe_tags:
            return True
    return False


def score_recipe(
    metadata: dict,
    constraints: UserConstraints,
) -> tuple[float, dict]:
    macro_dev = macro_deviation_score(metadata, constraints)
    budget_os = budget_overshoot_score(metadata, constraints)
    waste_frac = ingredient_waste_fraction(metadata, constraints)
    has_penalty = dietary_tag_penalty(metadata, constraints)

    score = 1.0 - (
        constraints.w_macro * macro_dev
        + constraints.w_budget * budget_os
        + constraints.w_waste * waste_frac
    )
    if has_penalty:
        score = -999.0

    components = {
        "macro_deviation": macro_dev,
        "cost_overshoot": budget_os,
        "waste_fraction": waste_frac,
        "dietary_penalty": has_penalty,
    }
    return score, components


def rerank(
    search_results: list[SearchResult],
    constraints: UserConstraints,
    top_k: int = 5,
) -> list[RankedResult]:
    ranked = []
    for r in search_results:
        score, components = score_recipe(r.metadata, constraints)
        ranked.append(
            RankedResult(
                recipe_id=r.recipe_id,
                name=r.name,
                score=score,
                macro_deviation=components["macro_deviation"],
                cost_overshoot=components["cost_overshoot"],
                waste_fraction=components["waste_fraction"],
                metadata=r.metadata,
            )
        )
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[:top_k]
