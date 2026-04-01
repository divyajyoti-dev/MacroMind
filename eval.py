"""
MacroMind — Evaluation Module
==============================
Tracks and compares metrics across the three pipeline variants.

Metrics tracked per run:
    - variant name
    - macro accuracy (how close the planned macros are to user targets)
    - constraint violations (count of dietary/budget rules broken)
    - latency (seconds)
    - token usage (prompt + completion + total)
    - estimated cost (USD)

Usage:
    # Log a result from any pipeline variant:
    from eval import log_result, compare_variants, print_summary

    result = run_pipeline(constraints, variant="rag-rerank")
    log_result(result, constraints)

    # After running all three variants, compare:
    compare_variants()

Results are appended to eval_log.jsonl (configurable in config.py).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import config
from rag import PipelineResult

log = logging.getLogger(__name__)


# ── Metric Computation ────────────────────────────────────────────────────────

def compute_macro_accuracy(
    result: PipelineResult,
    user_constraints: dict[str, Any],
) -> dict[str, float]:
    """
    Parse macro values from the LLM response text and compare to targets.

    We look for patterns like "Calories: 1850 kcal", "Protein: 145g", etc.
    Returns a dict with 'accuracy' (0–1) and per-dimension deviations.

    Note: This is a best-effort heuristic parser. For production, you'd want
    the LLM to output structured JSON.
    """
    import re

    text = result.response_text.lower()

    def extract_value(pattern: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    targets = {
        "calories": float(user_constraints.get("calories", 2000) or 2000),
        "protein":  float(user_constraints.get("protein",  150)  or 150),
        "carbs":    float(user_constraints.get("carbs",    200)  or 200),
        "fat":      float(user_constraints.get("fat",      65)   or 65),
    }

    parsed = {
        "calories": extract_value(r"(?:total\s+)?calories[:\s]+(\d+(?:\.\d+)?)"),
        "protein":  extract_value(r"protein[:\s]+(\d+(?:\.\d+)?)"),
        "carbs":    extract_value(r"(?:carbs?|carbohydrates?)[:\s]+(\d+(?:\.\d+)?)"),
        "fat":      extract_value(r"fat[:\s]+(\d+(?:\.\d+)?)"),
    }

    deviations = {}
    accuracy_scores = []
    for dim, target in targets.items():
        actual = parsed.get(dim)
        if actual is not None and target > 0:
            rel_error = abs(actual - target) / target
            deviations[f"{dim}_deviation_pct"] = round(rel_error * 100, 1)
            accuracy_scores.append(max(0.0, 1.0 - rel_error))
        else:
            deviations[f"{dim}_deviation_pct"] = None

    overall = round(sum(s for s in accuracy_scores) / len(accuracy_scores), 3) if accuracy_scores else None

    return {
        "macro_accuracy": overall,
        "parsed_macros": parsed,
        "target_macros": targets,
        **deviations,
    }


def count_constraint_violations(
    result: PipelineResult,
    user_constraints: dict[str, Any],
) -> int:
    """
    Count how many dietary restriction keywords appear in the LLM's response
    text in a context suggesting a violation.

    This is a lightweight heuristic — for production you'd use a structured
    LLM output or a dedicated classifier.
    """
    restrictions = user_constraints.get("dietary_restrictions", []) or []
    if isinstance(restrictions, str):
        restrictions = [r.strip() for r in restrictions.split(",") if r.strip()]

    text = result.response_text.lower()
    violations = 0
    violation_phrases = [
        f"contains {r.lower()}" for r in restrictions
    ] + [
        f"includes {r.lower()}" for r in restrictions
    ]
    for phrase in violation_phrases:
        if phrase in text:
            violations += 1
    return violations


# ── Logging ───────────────────────────────────────────────────────────────────

def log_result(
    result: PipelineResult,
    user_constraints: dict[str, Any],
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute metrics for a pipeline result and append a JSON record to
    eval_log.jsonl.

    Returns the full metrics dict so callers can display it immediately.
    """
    macro_metrics = compute_macro_accuracy(result, user_constraints)
    violations    = count_constraint_violations(result, user_constraints)

    record = {
        "variant":             result.variant,
        "latency_seconds":     result.latency_seconds,
        "prompt_tokens":       result.prompt_tokens,
        "completion_tokens":   result.completion_tokens,
        "total_tokens":        result.total_tokens,
        "estimated_cost_usd":  result.estimated_cost_usd,
        "retrieval_count":     result.retrieval_count,
        "rerank_applied":      result.rerank_applied,
        "constraint_violations": violations,
        "error":               result.error,
        **macro_metrics,
    }

    log_path = Path(config.EVAL_LOG_PATH)
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")

    log.info(
        "[eval] variant=%s  latency=%.2fs  tokens=%d  cost=$%.5f  accuracy=%s  violations=%d",
        result.variant,
        result.latency_seconds,
        result.total_tokens,
        result.estimated_cost_usd,
        macro_metrics.get("macro_accuracy"),
        violations,
    )
    return record


# ── Comparison / Reporting ────────────────────────────────────────────────────

def load_eval_log() -> list[dict[str, Any]]:
    """Load all records from eval_log.jsonl."""
    log_path = Path(config.EVAL_LOG_PATH)
    if not log_path.exists():
        return []
    records = []
    with log_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def compare_variants(records: list[dict] | None = None) -> dict[str, dict]:
    """
    Aggregate metrics per variant from eval log.

    Returns a dict keyed by variant name with average metrics.
    """
    if records is None:
        records = load_eval_log()

    if not records:
        log.warning("No eval records found.")
        return {}

    from collections import defaultdict
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        buckets[r["variant"]].append(r)

    summary = {}
    numeric_keys = [
        "latency_seconds", "total_tokens", "estimated_cost_usd",
        "macro_accuracy", "constraint_violations",
    ]
    for variant, recs in buckets.items():
        agg = {"n": len(recs)}
        for key in numeric_keys:
            vals = [r[key] for r in recs if r.get(key) is not None]
            agg[f"avg_{key}"] = round(sum(vals) / len(vals), 4) if vals else None
        summary[variant] = agg

    return summary


def print_summary(records: list[dict] | None = None) -> None:
    """Print a human-readable comparison table to stdout."""
    summary = compare_variants(records)
    if not summary:
        print("No evaluation data yet. Run all three variants and log results.")
        return

    header = f"{'Variant':<15} {'N':>4} {'Avg Latency':>12} {'Avg Tokens':>11} {'Avg Cost $':>11} {'Macro Acc':>10} {'Violations':>11}"
    print("\n" + "=" * len(header))
    print("MacroMind Variant Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for variant, agg in sorted(summary.items()):
        def fmt(v, fmt_str):
            return format(v, fmt_str) if v is not None else "  N/A"
        print(
            f"{variant:<15} "
            f"{agg['n']:>4} "
            f"{fmt(agg['avg_latency_seconds'], '>11.2f')} "
            f"{fmt(agg['avg_total_tokens'], '>11.0f')} "
            f"{fmt(agg['avg_estimated_cost_usd'], '>11.5f')} "
            f"{fmt(agg['avg_macro_accuracy'], '>10.3f')} "
            f"{fmt(agg['avg_constraint_violations'], '>11.1f')}"
        )
    print("=" * len(header) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MacroMind evaluation reporter")
    parser.add_argument("--summary", action="store_true", help="Print variant comparison table")
    parser.add_argument("--log",     action="store_true", help="Print all raw log entries")
    args = parser.parse_args()

    if args.log:
        for record in load_eval_log():
            print(json.dumps(record, indent=2))
    else:
        print_summary()
