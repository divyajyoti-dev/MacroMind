"""
MacroMind — Streamlit App
==========================
Wired to the team's src/ modules:
  src/retriever.py    — ChromaDB semantic search
  src/reranker.py     — constraint-aware scoring
  src/llm_generator.py — Groq (default) or Gemini meal plan generation
  src/evaluator.py    — metrics logging

Run:
    streamlit run app.py
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="MacroMind",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Lazy singletons ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embed_model():
    from sentence_transformers import SentenceTransformer
    from src.config import EMBEDDING_MODEL
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Connecting to recipe database…")
def _load_collection():
    from src.config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME
    from src.retriever import get_or_create_collection
    return get_or_create_collection(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def sidebar() -> dict:
    st.sidebar.title("⚙️ MacroMind Settings")

    # Variant
    st.sidebar.markdown("---")
    st.sidebar.subheader("Experiment Variant")
    variant = st.sidebar.radio(
        "Pipeline mode",
        ["rag-rerank", "rag", "baseline"],
        index=0,
        help=(
            "**rag-rerank**: Retrieve + rerank, then generate.\n\n"
            "**rag**: Retrieve (no rerank), then generate.\n\n"
            "**baseline**: LLM generates from scratch."
        ),
    )

    # Ingest
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Ingestion")
    st.sidebar.caption("Run once to build the recipe index.")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Ingest Data", use_container_width=True):
        _run_ingest(reset=False)
    if col2.button("Reset DB", use_container_width=True):
        _run_ingest(reset=True)

    # API Key
    st.sidebar.markdown("---")
    st.sidebar.subheader("API Key")
    st.sidebar.caption("Free key at **console.groq.com** — no billing needed.")
    api_key = st.sidebar.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.sidebar.markdown("---")
    st.sidebar.caption("MacroMind · Groq (Llama 3.3) + ChromaDB · INFO 290 @ Berkeley")

    return {"variant": variant, "api_key": api_key}


def _run_ingest(reset: bool = False) -> None:
    with st.sidebar:
        with st.spinner("Indexing recipes…"):
            try:
                import ingest
                # Clear cached collection so it re-connects after reset
                _load_collection.clear()
                ingest.run_ingestion(reset=reset)
                st.success("Done!")
            except Exception as e:
                st.error(f"Ingest failed: {e}")


# ── Form ───────────────────────────────────────────────────────────────────────

def render_form() -> dict | None:
    st.title("🥗 MacroMind — AI Meal Planner")
    st.caption(
        "Enter your daily targets, dietary needs, and what's in your kitchen. "
        "MacroMind retrieves real recipes and builds a personalised meal plan."
    )

    with st.form("meal_plan_form"):
        st.subheader("Daily Macro Targets")
        c1, c2, c3, c4 = st.columns(4)
        calories = c1.number_input("Calories (kcal)", 500, 6000, 2000, 50)
        protein  = c2.number_input("Protein (g)",     10,  500,  150,  5)
        carbs    = c3.number_input("Carbs (g)",        0,  700,  200,  5)
        fat      = c4.number_input("Fat (g)",          0,  300,   65,  5)

        st.subheader("Budget (optional)")
        budget_on = st.checkbox("Set a daily food budget")
        budget = st.number_input("Daily budget ($)", 1.0, 200.0, 20.0, 1.0) if budget_on else 9999.0

        st.subheader("Dietary Restrictions")
        restrictions = st.multiselect(
            "Select all that apply",
            ["vegetarian", "vegan", "gluten-free", "dairy-free", "high-protein", "meal-prep"],
        )

        st.subheader("Available Ingredients")
        st.caption("What's already in your kitchen? (comma-separated)")
        available_raw = st.text_area(
            "Pantry / fridge", placeholder="e.g. chicken breast, broccoli, olive oil, garlic", height=80
        )

        submitted = st.form_submit_button("🍽️ Generate My Meal Plan", use_container_width=True)

    if not submitted:
        return None

    return {
        "calories":              float(calories),
        "protein_g":             float(protein),
        "carbs_g":               float(carbs),
        "fat_g":                 float(fat),
        "budget_usd":            float(budget),
        "dietary_tags":          restrictions,
        "available_ingredients": [a.strip() for a in available_raw.split(",") if a.strip()],
    }


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(constraints_dict: dict, variant: str, api_key: str) -> dict:
    from src.reranker import UserConstraints, rerank
    from src.retriever import semantic_search, build_query_text
    from src.llm_generator import (
        generate_meal_plan_groq,
        generate_baseline_plan_groq,
    )

    constraints = UserConstraints(
        calories=constraints_dict["calories"],
        protein_g=constraints_dict["protein_g"],
        carbs_g=constraints_dict["carbs_g"],
        fat_g=constraints_dict["fat_g"],
        budget_usd=constraints_dict["budget_usd"],
        available_ingredients=constraints_dict["available_ingredients"],
        dietary_tags=constraints_dict["dietary_tags"],
    )

    t0 = time.time()
    candidates = []
    ranked = []
    response_text = ""

    if variant == "baseline":
        response_text = generate_baseline_plan_groq(constraints, api_key=api_key)

    else:
        # Retrieve
        collection   = _load_collection()
        embed_model  = _load_embed_model()

        if collection.count() == 0:
            return {"error": "Recipe index is empty. Click 'Ingest Data' in the sidebar first."}

        query = build_query_text(
            macro_targets={
                "calories": constraints.calories,
                "protein":  constraints.protein_g,
                "carbs":    constraints.carbs_g,
                "fat":      constraints.fat_g,
            },
            available_ingredients=constraints.available_ingredients,
            dietary_tags=constraints.dietary_tags,
            budget_usd=constraints.budget_usd if constraints.budget_usd < 9999 else None,
        )
        candidates = semantic_search(query, collection, embed_model, n_results=20)

        if variant == "rag":
            # Pass top-5 by cosine similarity (already sorted ascending = most similar first)
            ranked = [
                type("R", (), {"recipe_id": r.recipe_id, "name": r.name,
                               "score": None, "metadata": r.metadata})()
                for r in candidates[:5]
            ]
        else:  # rag-rerank
            ranked = rerank(candidates, constraints, top_k=5)

        response_text = generate_meal_plan_groq(constraints, ranked, api_key=api_key)

    latency = round(time.time() - t0, 2)

    return {
        "response_text": response_text,
        "candidates":    candidates,
        "ranked":        ranked,
        "latency":       latency,
        "variant":       variant,
        "error":         None,
    }


# ── Results ────────────────────────────────────────────────────────────────────

def render_results(result: dict, constraints_dict: dict) -> None:
    if result.get("error"):
        st.error(result["error"])
        st.info("Get a free Groq API key at console.groq.com and paste it in the sidebar.")
        return

    st.subheader("🍽️ Your Meal Plan")
    st.markdown(result["response_text"])

    ranked = result.get("ranked", [])
    if ranked:
        with st.expander(f"📚 Top Recipes Used ({len(ranked)})", expanded=False):
            for i, r in enumerate(ranked, 1):
                m = r.metadata
                score_str = f"  ·  Score: `{r.score:.3f}`" if getattr(r, "score", None) is not None else ""
                st.markdown(
                    f"**{i}. {r.name}**{score_str}  \n"
                    f"Cal: `{m.get('calories', '?'):.0f}` · "
                    f"P: `{m.get('protein', '?'):.0f}g` · "
                    f"C: `{m.get('carbs', '?'):.0f}g` · "
                    f"F: `{m.get('fat', '?'):.0f}g` · "
                    f"Cost: `${m.get('cost_usd', 0):.2f}`  \n"
                    f"*Ingredients:* {m.get('ingredients', '').replace('|', ', ')}"
                )
                if i < len(ranked):
                    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Variant",  result["variant"])
    m2.metric("Latency",  f"{result['latency']}s")
    m3.metric("Retrieved", f"{len(result.get('candidates', []))} recipes")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    settings    = sidebar()
    constraints = render_form()

    if constraints is None:
        st.info(
            "👈 **Getting started:**\n"
            "1. Get a free Groq API key at console.groq.com\n"
            "2. Paste it in the sidebar\n"
            "3. Click **Ingest Data** once\n"
            "4. Fill in your targets and click **Generate My Meal Plan**"
        )
        return

    if not settings["api_key"]:
        st.warning("Please add your Groq API key in the sidebar.")
        return

    with st.spinner("Building your meal plan…"):
        result = run_pipeline(constraints, settings["variant"], settings["api_key"])

    render_results(result, constraints)


if __name__ == "__main__":
    main()
