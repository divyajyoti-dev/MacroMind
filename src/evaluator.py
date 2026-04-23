from dataclasses import dataclass

import pandas as pd
from rouge_score import rouge_scorer

from src.reranker import rerank, RankedResult, ingredient_waste_fraction
from src.retriever import UserConstraints


@dataclass
class MacroDeviation:
    calories_pct: float
    protein_pct: float
    carbs_pct: float
    fat_pct: float
    mean_pct: float


def macro_deviation(predicted: dict, target: UserConstraints) -> MacroDeviation:
    def pct_error(pred_val, target_val):
        if target_val == 0:
            return 0.0
        return abs(pred_val - target_val) / target_val * 100.0

    cal_e = pct_error(predicted.get("calories", 0.0), target.calories)
    pro_e = pct_error(predicted.get("protein", 0.0), target.protein_g)
    carb_e = pct_error(predicted.get("carbs", 0.0), target.carbs_g)
    fat_e = pct_error(predicted.get("fat", 0.0), target.fat_g)
    mean_e = (cal_e + pro_e + carb_e + fat_e) / 4.0

    return MacroDeviation(
        calories_pct=cal_e,
        protein_pct=pro_e,
        carbs_pct=carb_e,
        fat_pct=fat_e,
        mean_pct=mean_e,
    )


def total_day_macros(selected_recipes: list[dict]) -> dict:
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "cost_usd": 0.0}
    for meta in selected_recipes:
        totals["calories"] += float(meta.get("calories", 0.0))
        totals["protein"] += float(meta.get("protein", 0.0))
        totals["carbs"] += float(meta.get("carbs", 0.0))
        totals["fat"] += float(meta.get("fat", 0.0))
        totals["cost_usd"] += float(meta.get("cost_usd", 0.0))
    return totals


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / k


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def evaluate_variant(
    variant_name: str,
    ranked_results: list[RankedResult],
    constraints: UserConstraints,
    relevant_ids: list[str],
    k: int = 5,
) -> dict:
    top_k = ranked_results[:k]
    retrieved_ids = [r.recipe_id for r in top_k]
    metadatas = [r.metadata for r in top_k]

    day_totals = total_day_macros(metadatas)
    macro_dev = macro_deviation(day_totals, constraints)

    p_at_k = precision_at_k(retrieved_ids, relevant_ids, k)

    avg_waste = sum(
        ingredient_waste_fraction(m, constraints) for m in metadatas
    ) / max(len(metadatas), 1)

    total_cost = day_totals["cost_usd"]
    within_budget = total_cost <= constraints.budget_usd

    return {
        "variant": variant_name,
        "precision_at_k": round(p_at_k, 4),
        "macro_dev_mean_pct": round(macro_dev.mean_pct, 2),
        "total_cost_usd": round(total_cost, 2),
        "within_budget": within_budget,
        "waste_fraction": round(avg_waste, 4),
    }


def build_results_dataframe(all_results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(all_results)
