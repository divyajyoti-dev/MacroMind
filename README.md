# MacroMind 🥗

**Macro-aware meal planning powered by RAG + constraint-aware reranking**

MacroMind is a multimodal Retrieval-Augmented Generation (RAG) system that turns your macro targets, budget, and fridge inventory into a grounded daily meal plan. It retrieves recipes from a vector database, re-ranks them against your constraints, and uses an LLM to generate an explainable plan.

Built for INFO 290 GenAI — UC Berkeley, Spring 2026.

---

## Team

| Name | Email | Notebook |
|------|-------|----------|
| Jyoti Divya | divya_jyoti@berkeley.edu | `notebooks/divya_evaluation.ipynb` |
| Pan Haoming | haoming.p@berkeley.edu | `notebooks/haoming_data_pipeline.ipynb` |
| Lin Iyu | iylin@berkeley.edu | `notebooks/iyu_recipe_processing.ipynb` |
| Baier Noah | jnoah_baier@berkeley.edu | `notebooks/noah_rag_pipeline.ipynb` |

---

## System Architecture

```
User Input
(macros, budget, ──► Query Builder ──► ChromaDB
 ingredients,         (text form)      (semantic search,
 dietary tags)                          all-MiniLM-L6-v2)
                                             │
                                    top-20 candidates
                                             ▼
                                       Reranker
                                 (macro deviation MAPE +
                                  budget penalty +
                                  ingredient waste score)
                                             │
                                      top-5 recipes
                                             ▼
                                     Gemini (LLM)
                                 (meal plan + explanation)
                                             │
                                    Meal Plan Output
```

**Three system variants are compared:**

| Variant | Retrieval | Reranking | Purpose |
|---------|-----------|-----------|---------|
| Baseline LLM | None | None | LLM parametric knowledge only |
| RAG | Semantic top-5 | None | Grounding benefit |
| RAG + Reranking | Semantic top-20 → top-5 | Constraint-aware | Full pipeline |

---

## Data Sources

| Source | Scale | Used for |
|--------|-------|----------|
| USDA FoodData Central API | 30 ingredients cached | Macro data per ingredient |
| Sample recipe dataset | 65 recipes | RAG retrieval corpus |
| BLS CPI estimates | 30 ingredients | Price-per-100g table |

---

## Project Structure

```
MacroMind/
├── src/
│   ├── config.py            # Central config: paths, API keys, constants
│   ├── data_pipeline.py     # USDA API integration + caching
│   ├── recipe_processor.py  # Recipe loading, macro computation, indexing
│   ├── retriever.py         # ChromaDB semantic search wrapper
│   ├── reranker.py          # Constraint-aware scoring + reranking
│   ├── llm_generator.py     # Gemini meal plan generation
│   └── evaluator.py         # Evaluation metrics (macro deviation, P@k, ROUGE)
│
├── notebooks/
│   ├── haoming_data_pipeline.ipynb    # USDA API, nutrition schema, price table
│   ├── iyu_recipe_processing.ipynb    # Embedding, ChromaDB indexing, search validation
│   ├── noah_rag_pipeline.ipynb        # System architecture + 3-variant demo
│   └── divya_evaluation.ipynb         # 5 test cases, metrics, weight sensitivity
│
├── data/
│   ├── recipes/sample_recipes.json    # 65 hand-crafted recipes
│   └── usda_cache/nutrition_cache.json # Pre-fetched USDA macros (30 ingredients)
│
├── requirements.txt
├── setup.py
└── .env.example
```

---

## Quickstart

**1. Clone and set up environment**
```bash
git clone https://github.com/divyajyoti-dev/MacroMind.git
cd MacroMind

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

**2. Add your API key**
```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key
```

```
GOOGLE_API_KEY=your_key_here
USDA_API_KEY=DEMO_KEY
```

**3. Run notebooks in order**

```bash
jupyter lab
```

| Order | Notebook | What it does |
|-------|----------|--------------|
| 1 | `iyu_recipe_processing.ipynb` | Builds the ChromaDB index (run once) |
| 2 | `haoming_data_pipeline.ipynb` | Explores USDA nutrition data |
| 3 | `noah_rag_pipeline.ipynb` | End-to-end demo of all 3 variants |
| 4 | `divya_evaluation.ipynb` | Evaluation across 5 user profiles |

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (local, no API key) |
| Vector store | ChromaDB (persistent, HNSW cosine index) |
| LLM | Google Gemini 2.0 Flash via `google-genai` SDK |
| Nutrition data | USDA FoodData Central REST API |
| Reranking | Custom MAPE-based scoring (pure Python) |
