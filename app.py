import time

import streamlit as st
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL
from src.retriever import get_or_create_collection, semantic_search, UserConstraints, build_query_text
from src.reranker import rerank, RankedResult
from src.data_pipeline import load_cache
from src.llm_generator import generate_baseline_plan_groq, generate_meal_plan_groq
from src.recipe_processor import load_recipes, load_cleaned_recipes, build_chroma_index

st.set_page_config(page_title="MacroMind", page_icon="🥗", layout="wide")


@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_collection():
    _, collection = get_or_create_collection()
    return collection


def run_pipeline(constraints: UserConstraints, variant: str, api_key: str) -> dict:
    result = {"response_text": "", "candidates": [], "ranked": [], "latency": 0.0, "error": None}
    t0 = time.time()
    try:
        model = load_model()
        collection = load_collection()

        if variant == "baseline":
            result["response_text"] = generate_baseline_plan_groq(constraints, api_key)

        elif variant == "rag":
            query = build_query_text(constraints)
            search_results = semantic_search(query, collection, model, n_results=5)
            result["candidates"] = search_results
            result["ranked"] = [
                RankedResult(r.recipe_id, r.name, 1 - r.score, 0.0, 0.0, 0.0, r.metadata)
                for r in search_results
            ]
            result["response_text"] = generate_meal_plan_groq(constraints, result["ranked"], api_key)

        elif variant == "rag-rerank":
            query = build_query_text(constraints)
            search_results = semantic_search(query, collection, model, n_results=20)
            result["candidates"] = search_results
            result["ranked"] = rerank(search_results, constraints, top_k=5)
            result["response_text"] = generate_meal_plan_groq(
                constraints, result["ranked"], api_key, use_rerank_prompt=True
            )

    except Exception as e:
        result["error"] = str(e)
    result["latency"] = round(time.time() - t0, 2)
    return result


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("MacroMind")
    st.caption("Constraint-aware meal planning with RAG")

    variant = st.selectbox(
        "Variant",
        options=["baseline", "rag", "rag-rerank"],
        format_func=lambda v: {"baseline": "V1 Baseline", "rag": "V2 RAG", "rag-rerank": "V3 RAG + Rerank"}[v],
    )

    api_key = st.text_input("Groq API Key", type="password", help="Free at console.groq.com")

    st.divider()

    if st.button("Rebuild Index (both sources)"):
        with st.spinner("Indexing ~10k recipes…"):
            model = load_model()
            _, col = get_or_create_collection()
            nutrition_cache = load_cache()
            recipes = list({r.id: r for r in load_recipes() + load_cleaned_recipes()}.values())
            build_chroma_index(recipes, col, model, nutrition_cache)
        st.success(f"Done. {col.count()} recipes indexed.")

# ── Main form ──────────────────────────────────────────────────────────────────

st.header("Meal Planner")

with st.form("meal_plan_form"):
    col1, col2 = st.columns(2)
    with col1:
        calories = st.number_input("Calories (kcal)", min_value=500, max_value=5000, value=2000, step=50)
        protein_g = st.number_input("Protein (g)", min_value=0, max_value=400, value=120, step=5)
    with col2:
        carbs_g = st.number_input("Carbs (g)", min_value=0, max_value=600, value=200, step=10)
        fat_g = st.number_input("Fat (g)", min_value=0, max_value=300, value=65, step=5)

    use_budget = st.checkbox("Set daily budget", value=False)
    budget_usd = st.number_input("Budget (USD/day)", min_value=1.0, max_value=50.0, value=12.0, step=0.5) if use_budget else 999.0

    dietary_tags = st.multiselect(
        "Dietary restrictions",
        options=["vegetarian", "vegan", "gluten-free", "keto"],
        default=[],
    )

    pantry_raw = st.text_area("Pantry ingredients (comma-separated)", placeholder="chicken, rice, broccoli, eggs")
    available_ingredients = [i.strip() for i in pantry_raw.split(",") if i.strip()] if pantry_raw else []

    submitted = st.form_submit_button("Generate Meal Plan", type="primary")

# ── Results ────────────────────────────────────────────────────────────────────

if submitted:
    if not api_key:
        st.error("Enter a Groq API key in the sidebar to generate a plan.")
    else:
        constraints = UserConstraints(
            calories=float(calories),
            protein_g=float(protein_g),
            carbs_g=float(carbs_g),
            fat_g=float(fat_g),
            budget_usd=float(budget_usd),
            available_ingredients=available_ingredients,
            dietary_tags=dietary_tags,
        )

        with st.spinner("Generating meal plan…"):
            output = run_pipeline(constraints, variant, api_key)

        if output["error"]:
            st.error(f"Error: {output['error']}")
        else:
            variant_label = {"baseline": "V1 Baseline", "rag": "V2 RAG", "rag-rerank": "V3 RAG + Rerank"}[variant]
            m1, m2, m3 = st.columns(3)
            m1.metric("Variant", variant_label)
            m2.metric("Latency", f"{output['latency']}s")
            m3.metric("Recipes retrieved", len(output["candidates"]) or "n/a")

            st.markdown("### Meal Plan")
            st.markdown(output["response_text"])

            if output["ranked"]:
                st.markdown("### Retrieved Recipes")
                for r in output["ranked"]:
                    meta = r.metadata
                    with st.expander(f"{r.name}  (score: {r.score:.3f})"):
                        cols = st.columns(4)
                        cols[0].metric("Calories", f"{meta.get('calories', '?')} kcal")
                        cols[1].metric("Protein", f"{meta.get('protein', '?')}g")
                        cols[2].metric("Carbs", f"{meta.get('carbs', '?')}g")
                        cols[3].metric("Cost", f"${meta.get('cost_usd', '?')}")
                        if meta.get("tags"):
                            st.caption(f"Tags: {meta['tags'].replace('|', ' · ')}")
