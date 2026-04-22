import json
import time
import difflib
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

from src.config import USDA_NUTRIENT_IDS, PRICE_PER_100G, USDA_CACHE_PATH

USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
USDA_FOOD_URL = "https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"


@dataclass
class NutritionFacts:
    fdc_id: int
    name: str
    calories_per_100g: float
    protein_per_100g: float
    fat_per_100g: float
    carbs_per_100g: float
    price_per_100g: float


def load_cache(path: Path = USDA_CACHE_PATH) -> dict[str, dict]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(data: dict[str, dict], path: Path = USDA_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _estimate_price(name: str) -> float:
    name_lower = name.lower()
    matches = difflib.get_close_matches(name_lower, PRICE_PER_100G.keys(), n=1, cutoff=0.5)
    if matches:
        return PRICE_PER_100G[matches[0]]
    # Try substring match as fallback
    for key, price in PRICE_PER_100G.items():
        if key in name_lower or name_lower in key:
            return price
    return 0.0


_DATATYPE_WEIGHTS = {
    "Foundation": 1.0,
    "SR Legacy": 0.7,
    "Survey (FNDDS)": 0.6,
    "Branded": 0.3,
}


def _tokenize(text: str) -> set[str]:
    import re
    return set(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def score_usda_match(query: str, candidate: dict) -> float:
    query_tokens = _tokenize(query)
    desc = candidate.get("description", "")
    desc_tokens = _tokenize(desc)
    overlap = len(query_tokens & desc_tokens) / max(len(query_tokens), 1)

    data_type = candidate.get("dataType", "Branded")
    type_weight = _DATATYPE_WEIGHTS.get(data_type, 0.3)

    return 0.6 * overlap + 0.4 * type_weight


def search_ingredient(query: str, api_key: str) -> list[dict]:
    params = {"query": query, "api_key": api_key, "pageSize": 5}
    resp = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("foods", [])


def get_nutrition(fdc_id: int, api_key: str) -> dict[str, float]:
    url = USDA_FOOD_URL.format(fdc_id=fdc_id)
    resp = requests.get(url, params={"api_key": api_key}, timeout=10)
    resp.raise_for_status()
    nutrients = resp.json().get("foodNutrients", [])
    result = {}
    for n in nutrients:
        nutrient = n.get("nutrient", {})
        nid = nutrient.get("id")
        for macro, target_id in USDA_NUTRIENT_IDS.items():
            if nid == target_id:
                result[macro] = n.get("amount", 0.0)
    return result


def _fetch_one(name: str, api_key: str) -> NutritionFacts | None:
    try:
        candidates = search_ingredient(name, api_key)
        if not candidates:
            return None
        best = max(candidates, key=lambda c: score_usda_match(name, c))
        fdc_id = best["fdcId"]
        nutrients = get_nutrition(fdc_id, api_key)
        return NutritionFacts(
            fdc_id=fdc_id,
            name=best.get("description", name),
            calories_per_100g=nutrients.get("calories", 0.0),
            protein_per_100g=nutrients.get("protein", 0.0),
            fat_per_100g=nutrients.get("fat", 0.0),
            carbs_per_100g=nutrients.get("carbs", 0.0),
            price_per_100g=_estimate_price(name),
        )
    except Exception:
        return None


def batch_fetch_ingredients(
    names: list[str],
    cache_path: Path = USDA_CACHE_PATH,
    api_key: str = "",
) -> dict[str, NutritionFacts]:
    cache = load_cache(cache_path)
    results: dict[str, NutritionFacts] = {}

    for name in names:
        key = name.lower().strip()
        if key in cache:
            results[name] = NutritionFacts(**cache[key])
            continue
        if not api_key:
            continue
        facts = _fetch_one(name, api_key)
        if facts:
            cache[key] = asdict(facts)
            results[name] = facts
        time.sleep(0.3)

    save_cache(cache, cache_path)
    return results
