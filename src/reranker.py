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


_ALLERGY_KEYWORDS: dict[str, set[str]] = {
    "tree nuts": {"almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "macadamia", "chestnut", "pine nut", "brazil nut"},
    "peanuts": {"peanut", "groundnut"},
    "dairy": {"milk", "cream", "butter", "cheese", "yogurt", "whey", "lactose", "casein", "mozzarella", "parmesan", "cheddar", "ricotta", "brie", "feta", "gouda", "buttermilk", "ghee"},
    "eggs": {"egg", "eggs", "yolk"},
    "shellfish": {"shrimp", "prawn", "crab", "lobster", "scallop", "clam", "oyster", "mussel", "crawfish", "crayfish"},
    "fish": {"salmon", "tuna", "cod", "tilapia", "sardine", "mackerel", "herring", "halibut", "trout", "anchovy"},
    "soy": {"soy", "tofu", "tempeh", "edamame", "soybean", "miso", "tamari"},
    "wheat/gluten": {"flour", "wheat", "bread", "pasta", "barley", "rye", "noodle", "breadcrumb", "semolina", "spelt", "farro", "couscous"},
    "sesame": {"sesame", "tahini"},
}


_CULTURAL_EXCLUSIONS: dict[str, set[str]] = {
    "halal": {"pork", "lard", "bacon", "ham", "prosciutto", "wine", "beer", "liqueur", "brandy", "rum", "vodka", "sake"},
    "kosher": {"pork", "lard", "bacon", "ham", "shrimp", "prawn", "crab", "lobster", "scallop", "clam", "oyster", "mussel"},
    "jain": {
        "chicken", "beef", "pork", "lamb", "turkey", "salmon", "tuna", "shrimp", "bacon", "ham", "sausage",
        "fish", "seafood", "crab", "lobster",
        "potato", "onion", "garlic", "carrot", "beet", "beetroot", "turnip", "radish",
    },
    "hindu vegetarian (no beef)": {"beef", "veal", "pork", "lard", "bacon", "ham", "bison"},
    "buddhist (no meat)": {
        "chicken", "beef", "pork", "lamb", "turkey", "salmon", "tuna", "shrimp", "bacon", "ham",
        "sausage", "fish", "seafood", "crab", "lobster", "anchovies",
    },
}

# kosher also forbids mixing meat + dairy in the same recipe
_KOSHER_MEAT = {"chicken", "beef", "pork", "lamb", "turkey", "veal", "duck", "bison"}
_KOSHER_DAIRY = {"milk", "cream", "butter", "cheese", "yogurt", "whey", "casein", "ghee"}


def cultural_dietary_penalty(metadata: dict, constraints: UserConstraints) -> bool:
    if not constraints.cultural_dietary:
        return False
    ingredient_words = set(metadata.get("ingredients", "").lower().replace("|", " ").split())
    for filter_name in constraints.cultural_dietary:
        exclusions = _CULTURAL_EXCLUSIONS.get(filter_name.lower(), set())
        if ingredient_words & exclusions:
            return True
        if filter_name.lower() == "kosher":
            has_meat = bool(ingredient_words & _KOSHER_MEAT)
            has_dairy = bool(ingredient_words & _KOSHER_DAIRY)
            if has_meat and has_dairy:
                return True
    return False


def allergy_ingredient_penalty(metadata: dict, constraints: UserConstraints) -> bool:
    if not constraints.allergy_tags:
        return False
    ingredient_text = metadata.get("ingredients", "").lower()
    ingredient_words = set(ingredient_text.replace("|", " ").split())
    for allergen in constraints.allergy_tags:
        keywords = _ALLERGY_KEYWORDS.get(allergen.lower(), set())
        if ingredient_words & keywords:
            return True
    return False


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
    has_dietary_penalty = dietary_tag_penalty(metadata, constraints)
    has_allergy_penalty = allergy_ingredient_penalty(metadata, constraints)
    has_cultural_penalty = cultural_dietary_penalty(metadata, constraints)

    score = 1.0 - (
        constraints.w_macro * macro_dev
        + constraints.w_budget * budget_os
        + constraints.w_waste * waste_frac
    )
    if has_dietary_penalty or has_allergy_penalty or has_cultural_penalty:
        score = -999.0

    components = {
        "macro_deviation": macro_dev,
        "cost_overshoot": budget_os,
        "waste_fraction": waste_frac,
        "dietary_penalty": has_dietary_penalty or has_allergy_penalty or has_cultural_penalty,
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
