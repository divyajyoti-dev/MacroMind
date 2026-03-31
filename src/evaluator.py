"""
Evaluation metrics for MacroMind system variants.
Used by Divya's evaluation notebook.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from src.reranker import UserConstraints, RankedResult


@dataclass
class MacroDeviation:
    calories_pct: float
    protein_pct: float
    carbs_pct: float
    fat_pct: float
    mean_pct: float


def macro_deviation(predicted: dict, target: dict) -> MacroDeviation:
    """
    Per-macro absolute percentage deviation between predicted and target.
    predicted / target keys: 'calories', 'protein', 'carbs', 'fat'
    """
    def pct_err(pred_val, tgt_val):
        if tgt_val == 0:
            return 0.0
        return abs(pred_val - tgt_val) / tgt_val * 100

    cal  = pct_err(predicted.get("calories", 0), target.get("calories", 0))
    prot = pct_err(predicted.get("protein",  0), target.get("protein",  0))
    carb = pct_err(predicted.get("carbs",    0), target.get("carbs",    0))
    fat  = pct_err(predicted.get("fat",      0), target.get("fat",      0))

    return MacroDeviation(
        calories_pct=round(cal,  2),
        protein_pct= round(prot, 2),
        carbs_pct=   round(carb, 2),
        fat_pct=     round(fat,  2),
        mean_pct=    round(np.mean([cal, prot, carb, fat]), 2),
    )


def total_day_macros(selected_recipes: list[dict]) -> dict:
    """Sum macros across a list of recipe metadata dicts."""
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "cost_usd": 0.0}
    for r in selected_recipes:
        totals["calories"] += r.get("calories", 0)
        totals["protein"]  += r.get("protein",  0)
        totals["carbs"]    += r.get("carbs",    0)
        totals["fat"]      += r.get("fat",      0)
        totals["cost_usd"] += r.get("cost_usd", 0)
    return {k: round(v, 1) for k, v in totals.items()}


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of top-k retrieved items that are in the relevant set."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 between a reference plan and a generated plan."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, hypothesis)
    return round(score["rougeL"].fmeasure, 4)


def evaluate_variant(
    variant_name: str,
    ranked_results: list[RankedResult],
    constraints: UserConstraints,
    relevant_ids: list[str],
    k: int = 5,
) -> dict:
    """
    Compute all metrics for a single system variant on a single test case.
    Returns a flat dict suitable for building a DataFrame row.
    """
    top_k_ids = [r.recipe_id for r in ranked_results[:k]]
    top_k_metadata = [r.metadata for r in ranked_results[:k]]

    day_macros = total_day_macros(top_k_metadata)
    target = {
        "calories": constraints.calories,
        "protein":  constraints.protein_g,
        "carbs":    constraints.carbs_g,
        "fat":      constraints.fat_g,
    }
    dev = macro_deviation(day_macros, target)
    p_at_k = precision_at_k(top_k_ids, relevant_ids, k)
    total_cost = day_macros.get("cost_usd", 0)
    budget_ok = total_cost <= constraints.budget_usd

    return {
        "variant":           variant_name,
        "precision_at_k":    round(p_at_k, 4),
        "macro_dev_mean_pct":dev.mean_pct,
        "macro_dev_cal_pct": dev.calories_pct,
        "macro_dev_prot_pct":dev.protein_pct,
        "macro_dev_carb_pct":dev.carbs_pct,
        "macro_dev_fat_pct": dev.fat_pct,
        "total_cost_usd":    total_cost,
        "within_budget":     budget_ok,
        "waste_fraction":    round(np.mean([r.waste_fraction for r in ranked_results[:k]]), 4)
                             if ranked_results else 0.0,
    }


def build_results_dataframe(
    all_results: list[dict],
) -> pd.DataFrame:
    """Convert a list of evaluate_variant results into a tidy DataFrame."""
    return pd.DataFrame(all_results)
