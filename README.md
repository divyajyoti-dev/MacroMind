# MacroMind

MacroMind is a constraint-aware meal planning system that generates one-day meal plans from a user's macro targets, budget, and dietary restrictions. Unlike static meal planners, it retrieves semantically relevant recipes from a 10,000-recipe corpus and uses a constraint-aware reranker to select the best five matches before passing them to an LLM — so the generated plan is grounded in real recipes that actually fit the user's numbers.

---

## Architecture

Four-stage pipeline: **User Constraints → Query Construction → Semantic Retrieval → Constraint-Aware Reranking → LLM Generation**

The system runs in three variants with increasing sophistication. See [architecture.html](architecture.html) for an interactive diagram.

| Variant | Retrieval | Reranking | Purpose |
|---|---|---|---|
| V1 Baseline | None | None | Parametric LLM knowledge only |
| V2 RAG | Semantic top-5 | None | Grounding benefit, no constraint filtering |
| V3 RAG + Rerank | Semantic top-20 → top-5 | Constraint-aware MAPE + budget + waste | Full constraint-optimised pipeline |

**Reranker formula (V3):**
```
score = 1.0 − (0.50 × macro_MAPE + 0.30 × budget_overshoot + 0.20 × waste_fraction)
```
Recipes that violate a dietary tag receive a hard −999 penalty and never surface in the top-5.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your GROQ_API_KEY (free at console.groq.com)
python ingest.py --reset --source both
streamlit run app.py
```

For development, index only the 65 curated recipes (fast):
```bash
python ingest.py --source sample
```

---

## Corpus

9,999 recipes sampled from the [RecipeNLG dataset](https://recipenlg.cs.put.poznan.pl/) (2.2M total; Bień et al., 2020), plus 65 hand-curated recipes with verified macro data. Nutrition data from [USDA FoodData Central](https://fdc.nal.usda.gov/). Gram weights default to 100g per ingredient for RecipeNLG records (no weights in source).

---

## Evaluation

A three-variant ablation across five user profiles (Body Recomp, Muscle Bulk, Vegetarian Weight Loss, Budget Meal Prep, Keto/Low-Carb) shows that V3 outperforms V1 on Precision@5 for all five profiles — including perfect scores for the Vegetarian and Keto profiles where the dietary tag penalty eliminates non-compliant candidates entirely.

See [`notebooks/divya_evaluation.ipynb`](notebooks/divya_evaluation.ipynb) for the full results. The notebook runs offline with no LLM API key required.

---

## Project Structure

```
src/
  config.py            — centralised paths and constants
  data_pipeline.py     — USDA lookup, scoring, local cache
  recipe_processor.py  — Recipe dataclass, loaders, embedding prep
  retriever.py         — UserConstraints, ChromaDB setup, semantic search
  reranker.py          — RankedResult, score_recipe, rerank()
  llm_generator.py     — Groq (primary) and Gemini (fallback) generation
  evaluator.py         — macro_deviation, precision_at_k, evaluate_variant
prompts/
  meal_planner.py      — versioned prompt templates (v1)
notebooks/
  divya_evaluation.ipynb — three-variant ablation evaluation
ingest.py              — CLI for embedding + indexing
app.py                 — Streamlit UI
architecture.html      — interactive system diagram
```

---

## LLM Providers

- **Primary:** Groq `llama-3.3-70b-versatile` — free tier, no billing required
- **Fallback:** Gemini 2.0 Flash — requires `GOOGLE_API_KEY`

Both are supported; switch by calling the `generate_*` or `generate_*_groq` functions in `src/llm_generator.py`.
