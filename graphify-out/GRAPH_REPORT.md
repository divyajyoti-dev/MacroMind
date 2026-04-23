# MacroMind — Knowledge Graph Report (Full Codebase)

Generated from: `.` (MacroMind_master branch, post-Issue #20)

---

## Corpus

| Type | Count |
|---|---|
| Code | 13 files (src/, prompts/, tests/, ingest.py, app.py) |
| Documents | 4 files (CLAUDE.md, PROGRESS.md, requirements.txt, GRAPH_REPORT.md) |
| Papers | 1 file (Gen AI Project Proposal) |
| Images | 2 files (evaluation_results.png, weight_sensitivity.png) |
| **Total** | **20 files · ~8,381 words** |

---

## Graph Stats

| Metric | Value |
|---|---|
| Nodes | 173 |
| Edges | 254 (166 AST structural + 88 semantic) |
| Hyperedges | 19 |
| Communities | 21 |

---

## Communities

| # | Theme | Nodes | Cohesion |
|---|---|---|---|
| 0 | Vector Store & Evaluation (ChromaDB, SentenceTransformer, offline notebook) | 25 | — |
| 1 | Scoring & Evaluation Metrics (reranker weights, ROUGE-L, DataFrame export) | 23 | — |
| 2 | USDA Data Pipeline (batch_fetch, _estimate_price, get_nutrition, score_usda_match) | 18 | — |
| 3 | Ingestion CLI (ingest.py, _merge_dedup, 100g default) | 14 | — |
| 4 | LLM Providers & Agents (Groq, Gemini, sub-agents) | 14 | — |
| 5 | Issue Backlog — Data Phase (Issues #6–#8, Phase 2) | 14 | — |
| 6 | Config & Pricing (PRICE_PER_100G, USDA_NUTRIENT_IDS, datatype weights) | 14 | — |
| 7 | Streamlit App (app.py, run_pipeline, load_model, load_collection) | 14 | — |
| 8 | Generation Functions (Groq + Gemini baselines, model name constants) | 11 | — |
| 9 | Prompt Formatting (_format_constraints, generate_* functions) | 8 | — |
| 10 | Issue Backlog — Foundation Phase (Issues #2, #4, #5, #10) | 8 | — |
| 11–20 | Singletons (dataclass nodes, __init__.py, graph report) | 1 each | — |

---

## God Nodes (highest degree)

| Node | Degree | Role |
|---|---|---|
| MacroMind System (Proposal) | 14 | Central concept hub — all pipeline stages reference it |
| run_pipeline() | 13 | Streamlit orchestrator — dispatches all three variants |
| main() (ingest.py) | 11 | CLI entry point — wires corpus loading + embedding + upsert |
| data_pipeline.py | 10 | Data layer hub — USDA lookup, cache, scoring |
| MacroMind System (CLAUDE.md) | 10 | Project config hub — rules, agents, corpus refs |
| score_usda_match() | 8 | Scoring function — used by _fetch_one, connects data type weights |
| Retrieval Stage | 8 | RAG core — SentenceTransformer + ChromaDB + query builder |
| _fetch_one() | 7 | Internal orchestrator — calls search, score, get_nutrition |

---

## Hyperedges (cross-cutting groups)

1. **End-to-End RAG Pipeline** — build_query_text → semantic_search → rerank → generate_meal_plan_groq
2. **Composite Reranker** — macro_deviation_score, budget_overshoot_score, ingredient_waste_fraction, dietary_tag_penalty → score_recipe
3. **Offline Evaluation Suite** — macro_deviation, precision_at_k, waste_fraction, evaluate_variant (no API key)
4. **Ablation Variants** — V1 Baseline, V2 RAG, V3 RAG+Rerank
5. **LLM Provider Pair** — Groq (primary), Gemini (fallback)
6. **USDA Scoring Pipeline** — search_ingredient, score_usda_match, get_nutrition, _fetch_one
7. **Embedding Index** — SentenceTransformer, ChromaDB, recipe_to_document, build_chroma_index
8. **Test Coverage** — test_data_pipeline.py covers score_usda_match with 5 fixture pairs
9. **Four-Stage Proposal Pipeline** — Query Construction → Retrieval → Reranking → LLM Generation
10. **Data Sources** — RecipeNLG (9,999), curated set (65), USDA FoodData Central

---

## Surprising Cross-Codebase Connections

- **`_format_constraints()` in llm_generator.py ↔ `build_query_text()` in retriever.py**: Both convert `UserConstraints` into a text representation, but for entirely different purposes — one for the LLM prompt, one for the embedding query. Neither calls the other. This is a semantic near-duplicate that could be unified.
- **`macro_deviation()` in evaluator.py ↔ `macro_deviation_score()` in reranker.py**: Both compute per-macro deviation from the same `UserConstraints` targets. The evaluator returns a percentage; the reranker returns a normalized [0,1] score. They share the same conceptual intent but diverged in implementation.
- **`weight_sensitivity.png` ↔ `w_macro` in `UserConstraints`**: The image is a direct empirical measurement of the hyperparameter defined in `src/retriever.py`. The graph connects a runtime constant to its empirical consequence across two separate files.
- **`100g gram default` (CLAUDE.md project rule) ↔ `load_cleaned_recipes()` (src/recipe_processor.py)**: The project rule enforcing 100g defaults is structurally connected to the code that implements it — a compliance relationship across documentation and implementation.

---

## Suggested Questions

1. What connects `run_pipeline()` in `app.py` to `rerank()` in `src/reranker.py` — trace the V3 call chain?
2. Where does `UserConstraints` flow from the Streamlit form to the LLM prompt template?
3. Why do `_format_constraints()` and `build_query_text()` solve similar problems differently?
4. How does the USDA scoring pipeline defend against Branded food noise?
5. Which evaluation metrics are shared between the reranker and the offline notebook?

---

*Outputs: `graph.html` (interactive), `graph.json` (GraphRAG-ready), `GRAPH_REPORT.md` (this file)*
*Nodes: 173 · Edges: 254 · Communities: 21*
