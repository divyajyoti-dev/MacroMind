"""
USDA FoodData Central integration.
Handles API calls, caching, and nutrition schema normalization.
"""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

from src.config import USDA_API_KEY, USDA_NUTRIENT_IDS, PRICE_PER_100G, USDA_CACHE_PATH

USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
USDA_FOOD_URL   = "https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"


@dataclass
class NutritionFacts:
    fdc_id: int
    name: str
    calories_per_100g: float
    protein_per_100g: float
    fat_per_100g: float
    carbs_per_100g: float
    price_per_100g: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NutritionFacts":
        return cls(**d)


def search_ingredient(query: str, api_key: str = USDA_API_KEY) -> dict | None:
    """Return the first USDA FoodData Central search result for an ingredient."""
    params = {
        "query":    query,
        "api_key":  api_key,
        "pageSize": 5,
        "dataType": ["SR Legacy", "Survey (FNDDS)"],
    }
    resp = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    foods = data.get("foods", [])
    return foods[0] if foods else None


def get_nutrition(fdc_id: int, api_key: str = USDA_API_KEY) -> NutritionFacts | None:
    """Fetch full nutrition facts for a specific FDC ID."""
    url = USDA_FOOD_URL.format(fdc_id=fdc_id)
    resp = requests.get(url, params={"api_key": api_key}, timeout=10)
    resp.raise_for_status()
    food = resp.json()

    macros = {v: 0.0 for v in ["calories", "protein", "fat", "carbs"]}
    for nutrient in food.get("foodNutrients", []):
        n = nutrient.get("nutrient", {})
        nid = n.get("id")
        for macro_name, macro_id in USDA_NUTRIENT_IDS.items():
            if nid == macro_id:
                macros[macro_name] = nutrient.get("amount", 0.0)

    name = food.get("description", "").lower()
    return NutritionFacts(
        fdc_id=fdc_id,
        name=name,
        calories_per_100g=macros["calories"],
        protein_per_100g=macros["protein"],
        fat_per_100g=macros["fat"],
        carbs_per_100g=macros["carbs"],
        price_per_100g=_estimate_price(name),
    )


def _estimate_price(name: str) -> float:
    """Look up price from the hardcoded table using fuzzy matching."""
    name_lower = name.lower()
    for key, price in PRICE_PER_100G.items():
        if key in name_lower or name_lower in key:
            return price
    return 0.20  # default fallback


def load_cache(path: Path = USDA_CACHE_PATH) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(data: dict, path: Path = USDA_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def batch_fetch_ingredients(
    names: list[str],
    cache_path: Path = USDA_CACHE_PATH,
    api_key: str = USDA_API_KEY,
    sleep_between: float = 0.3,
) -> dict[str, NutritionFacts]:
    """
    Fetch nutrition facts for a list of ingredient names.
    Results are cached to avoid repeated API calls.
    Returns a dict mapping ingredient name -> NutritionFacts.
    """
    cache = load_cache(cache_path)
    results: dict[str, NutritionFacts] = {}

    # Hydrate already-cached entries
    for name in names:
        if name in cache:
            results[name] = NutritionFacts.from_dict(cache[name])

    # Fetch missing entries
    missing = [n for n in names if n not in cache]
    for name in missing:
        print(f"  Fetching USDA data for: {name}")
        result = search_ingredient(name, api_key)
        if result:
            nf = get_nutrition(result["fdcId"], api_key)
            if nf:
                results[name] = nf
                cache[name] = nf.to_dict()
        time.sleep(sleep_between)

    if missing:
        save_cache(cache, cache_path)

    return results
