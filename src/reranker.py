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
