# MacroMind — Knowledge Graph Report

Generated from: `.` (MacroMind_master branch)

---

## Corpus

| Type | Count |
|---|---|
| Code | 2 files |
| Documents | 2 files (CLAUDE.md, PROGRESS.md) |
| Papers | 1 file (Gen AI Project Proposal) |
| **Total** | **5 files · ~412 words** |

---

## Graph Stats

| Metric | Value |
|---|---|
| Nodes | 64 |
| Edges | 85 |
| Hyperedges | 8 |
| Communities | 10 |

---

## Communities

| # | Theme | Nodes | Cohesion |
|---|---|---|---|
| 0 | LLM Providers & Agents (Groq, Gemini, sub-agents) | 12 | 0.20 |
| 1 | Data Layer Issues (USDA, recipe processor) | 11 | 0.24 |
| 2 | Project Skeleton (ChromaDB, ingest, issues #2–#10) | 10 | 0.22 |
| 3 | Evaluation & Reranking (Divya's notebook, MAPE, offline constraint) | 8 | 0.36 |
| 4 | RAG Pipeline (query construction, retrieval, all-MiniLM-L6-v2, V2/V3) | 8 | 0.29 |
| 5 | System Vision (V1 baseline, Streamlit UI, memory layer) | 7 | 0.29 |
| 6 | Corpus Rationale (RecipeNLG 2.2M, 100g default, bottleneck rationale) | 5 | 0.40 |
| 7–9 | Singletons (prompts init, src init, issue block #11–21) | 3 | 1.00 |

---

## God Nodes (highest degree)

| Node | Degree | Role |
|---|---|---|
| MacroMind System (Proposal) | 14 | Central hub — all pipeline stages connect through it |
| MacroMind System (CLAUDE.md) | 10 | Project config hub — agents, rules, corpus refs |
| Retrieval Stage | 8 | RAG core — connects embedding, ChromaDB, query builder |
| RecipeNLG Corpus | 6 | Primary data source — referenced by data layer, ingest |
| Reranking Stage | 6 | Downstream of retrieval; connects to MAPE, budget scorer |
| USDA FoodData Central | 5 | Nutrition source — NER, scoring, cache |
| LLM Generation Stage | 5 | Final pipeline step — connects to all variants |

---

## Hyperedges (cross-cutting groups)

1. **Four-Stage RAG Pipeline** — Query Construction → Retrieval → Reranking → LLM Generation
2. **Ablation Variants** — V1 Baseline, V2 RAG, V3 RAG+Reranking
3. **Evaluation Metrics Suite** — Macro MAPE, Precision@5, Budget Adherence
4. **LLM Providers** — Groq (primary), Gemini (fallback)
5. **Data Sources** — RecipeNLG, USDA FoodData Central
6. **Sub-Agent System** — security-reviewer, verifier, progress-tracker
7. **Development Phases** — Foundation, Data, Vector Store
8. **Team Notebooks** — Haoming, Iyu, Noah, Divya

---

## Surprising Cross-Document Connections

- **Offline constraint (CLAUDE.md) ↔ Divya's evaluation notebook**: The project rule banning API keys in notebooks is directly encoded as a design rationale node, connecting the developer tooling layer to the evaluation methodology.
- **RecipeNLG corpus rationale (Proposal) ↔ PROGRESS.md issues**: The "corpus size as bottleneck" rationale in the proposal aligns with why Issue #7 (scored USDA matching) exists — both acknowledge that data quality gates retrieval diversity.
- **100g gram default (CLAUDE.md) ↔ NER field (Proposal)**: The gram-weight default is a downstream consequence of RecipeNLG's NER format, connecting a project rule to the data layer's schema constraint.

---

## Suggested Questions

1. How does the reranking formula connect macro MAPE to the budget overshoot scorer?
2. Which components share direct dependencies with the retrieval stage?
3. What connects the corpus bottleneck rationale to the Phase 2 data issues?
4. How do the four team notebooks partition the evaluation work?
5. Where does the 100g gram default appear across both the data pipeline and the embedding logic?

---

*Outputs: `graph.html` (interactive), `graph.json` (GraphRAG-ready), `GRAPH_REPORT.md` (this file)*
