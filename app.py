"""
MacroMind — Streamlit Frontend
===============================
Single-page Streamlit app for the MacroMind meal planning assistant.

Sections:
  1. Sidebar  — experiment variant selector + ingest button
  2. Main form — user enters macro targets, budget, restrictions, pantry
  3. Results   — LLM meal plan, retrieved recipes (expandable), metrics

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

# ── Page Config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="MacroMind",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Helper: lazy imports after page config ─────────────────────────────────
def _import_pipeline():
    """Import rag and eval lazily so Streamlit page config runs first."""
    import rag
    import eval as ev
    return rag, ev


# ── Sidebar ────────────────────────────────────────────────────────────────

def sidebar() -> dict:
    st.sidebar.title("⚙️ MacroMind Settings")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Experiment Variant")
    variant = st.sidebar.radio(
        "Pipeline mode",
        options=["rag-rerank", "rag", "baseline"],
        index=0,
        help=(
            "**rag-rerank**: Retrieve + rerank recipes, then ask Gemini to plan.\n\n"
            "**rag**: Retrieve recipes (no reranking), then ask Gemini.\n\n"
            "**baseline**: No retrieval — Gemini invents meals from scratch."
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Ingestion")
    st.sidebar.caption(
        "Run this once after cloning the repo (or when you update recipes_sample.csv)."
    )

    col1, col2 = st.sidebar.columns(2)
    run_ingest  = col1.button("Ingest Data", use_container_width=True)
    reset_ingest = col2.button("Reset DB",   use_container_width=True)

    if run_ingest or reset_ingest:
        _run_ingest(reset=reset_ingest)

    st.sidebar.markdown("---")
    st.sidebar.subheader("API Key")
    st.sidebar.caption("Free key at **console.groq.com** — no billing needed.")
    api_key = st.sidebar.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Stored only in memory for this session.",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        import config, rag
        config.GROQ_API_KEY = api_key
        rag._groq_client = None  # force reinitialise with new key

    st.sidebar.markdown("---")
    st.sidebar.caption("MacroMind v1.0 · Powered by Groq (Llama 3.3) + ChromaDB")

    return {"variant": variant}


def _run_ingest(reset: bool = False) -> None:
    with st.sidebar:
        with st.spinner("Running ingestion pipeline…"):
            try:
                import ingest
                ingest.run_ingestion(reset=reset)
                st.success("Ingestion complete!")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")


# ── Main Form ──────────────────────────────────────────────────────────────

def render_form() -> dict | None:
    st.title("🥗 MacroMind — AI Meal Planner")
    st.caption(
        "Enter your daily nutrition targets, dietary needs, and what you already have "
        "in your kitchen. MacroMind will recommend a personalised meal plan."
    )

    with st.form("meal_plan_form"):
        # ── Macro targets ──────────────────────────────────────────────
        st.subheader("Daily Macro Targets")
        c1, c2, c3, c4 = st.columns(4)
        calories = c1.number_input("Calories (kcal)", min_value=500,  max_value=6000, value=2000, step=50)
        protein  = c2.number_input("Protein (g)",     min_value=10,   max_value=500,  value=150,  step=5)
        carbs    = c3.number_input("Carbs (g)",        min_value=0,    max_value=700,  value=200,  step=5)
        fat      = c4.number_input("Fat (g)",          min_value=0,    max_value=300,  value=65,   step=5)

        # ── Budget ─────────────────────────────────────────────────────
        st.subheader("Budget (optional)")
        budget_enabled = st.checkbox("Set a daily food budget", value=False)
        budget = None
        if budget_enabled:
            budget = st.number_input(
                "Daily budget ($)", min_value=1.0, max_value=500.0, value=20.0, step=1.0
            )

        # ── Dietary restrictions ───────────────────────────────────────
        st.subheader("Dietary Restrictions")
        restriction_options = ["vegetarian", "vegan", "gluten-free", "dairy-free", "nut-free", "low-sodium"]
        restrictions = st.multiselect(
            "Select all that apply",
            options=restriction_options,
            default=[],
        )

        # ── Available ingredients ──────────────────────────────────────
        st.subheader("Available Ingredients")
        st.caption("What do you already have at home? (comma-separated)")
        available_raw = st.text_area(
            "Pantry / fridge items",
            placeholder="e.g. chicken breast, broccoli, olive oil, garlic, brown rice",
            height=80,
        )

        submitted = st.form_submit_button("🍽️ Generate My Meal Plan", use_container_width=True)

    if not submitted:
        return None

    available = [a.strip() for a in available_raw.split(",") if a.strip()]

    return {
        "calories":               calories,
        "protein":                protein,
        "carbs":                  carbs,
        "fat":                    fat,
        "budget":                 budget,
        "dietary_restrictions":   restrictions,
        "available_ingredients":  available,
    }


# ── Results Display ────────────────────────────────────────────────────────

def render_results(result, metrics: dict, constraints: dict) -> None:
    if result.error:
        st.error(f"Pipeline error: {result.error}")
        st.info(
            "Common fixes:\n"
            "- Get a free Groq API key at console.groq.com and paste it in the sidebar.\n"
            "- Run 'Ingest Data' if you haven't yet indexed the recipes."
        )
        return

    # ── Meal Plan ──────────────────────────────────────────────────────
    st.subheader("🍽️ Your Meal Plan")
    st.markdown(result.response_text)

    # ── Retrieved Recipes (collapsible) ───────────────────────────────
    if result.recipes_used:
        with st.expander(f"📚 Recipes Considered ({len(result.recipes_used)} total)", expanded=False):
            for i, r in enumerate(result.recipes_used, 1):
                macros = r.get("estimated_macros", {})
                score  = r.get("score")
                score_str = f"  ·  Score: `{score:.3f}`" if score is not None else ""
                st.markdown(
                    f"**{i}. {r['title']}**{score_str}  \n"
                    f"Cal: `{macros.get('calories', '?')}` · "
                    f"P: `{macros.get('protein', '?')}g` · "
                    f"C: `{macros.get('carbs', '?')}g` · "
                    f"F: `{macros.get('fat', '?')}g`  \n"
                    f"*Ingredients:* {r.get('ingredients', '').replace('|', ', ')}"
                )
                if i < len(result.recipes_used):
                    st.divider()

    # ── Metrics ────────────────────────────────────────────────────────
    st.subheader("📊 Run Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Variant",          result.variant)
    m2.metric("Latency",          f"{result.latency_seconds:.2f}s")
    m3.metric("Tokens Used",      f"{result.total_tokens:,}")
    m4.metric("Est. Cost",        f"${result.estimated_cost_usd:.5f}")
    m5.metric("Retrieved",        f"{result.retrieval_count} recipes")

    macro_acc = metrics.get("macro_accuracy")
    if macro_acc is not None:
        violations = metrics.get("constraint_violations", 0)
        ma1, ma2 = st.columns(2)
        ma1.metric(
            "Macro Accuracy",
            f"{macro_acc:.1%}",
            help="How closely the planned macros match your targets (heuristic parse of LLM output).",
        )
        ma2.metric(
            "Constraint Violations",
            violations,
            help="Detected dietary restriction violations in the LLM output (heuristic).",
        )


# ── Variant Comparison ─────────────────────────────────────────────────────

def render_comparison() -> None:
    import eval as ev
    records = ev.load_eval_log()
    if not records:
        return

    st.subheader("📈 Variant Comparison (cumulative)")
    summary = ev.compare_variants(records)
    if not summary:
        return

    rows = []
    for variant, agg in sorted(summary.items()):
        rows.append({
            "Variant":        variant,
            "Runs":           agg["n"],
            "Avg Latency (s)":agg.get("avg_latency_seconds"),
            "Avg Tokens":     agg.get("avg_total_tokens"),
            "Avg Cost ($)":   agg.get("avg_estimated_cost_usd"),
            "Avg Macro Acc":  agg.get("avg_macro_accuracy"),
            "Avg Violations": agg.get("avg_constraint_violations"),
        })
    st.dataframe(rows, use_container_width=True)


# ── App Entry Point ────────────────────────────────────────────────────────

def main() -> None:
    settings    = sidebar()
    constraints = render_form()

    if constraints is None:
        # No submission yet — show onboarding info
        st.info(
            "👈 **Getting started:**\n"
            "1. Add your OpenAI API key in the sidebar.\n"
            "2. Click **Ingest Data** once to build the recipe index.\n"
            "3. Fill in your macro targets above and click **Generate My Meal Plan**."
        )
        render_comparison()
        return

    rag, ev = _import_pipeline()

    with st.spinner("Building your meal plan…"):
        result = rag.run_pipeline(constraints, variant=settings["variant"])

    metrics = ev.log_result(result, constraints)
    render_results(result, metrics, constraints)
    render_comparison()


if __name__ == "__main__":
    main()
