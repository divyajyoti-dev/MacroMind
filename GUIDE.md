# MacroMind — Developer & User Guide

> This file is the living manual for MacroMind. Every time the codebase is
> meaningfully extended, a new entry is added to the **Iteration Log** at the
> bottom. Think of it as a changelog + architecture explanation in one place.

---

## Table of Contents

1. [What MacroMind Does](#1-what-macromind-does)
2. [Architecture Overview](#2-architecture-overview)
3. [File-by-File Explanation](#3-file-by-file-explanation)
4. [Data Files](#4-data-files)
5. [How the Three Variants Work](#5-how-the-three-variants-work)
6. [The Scoring Formula](#6-the-scoring-formula)
7. [Getting Started (Quick Start)](#7-getting-started-quick-start)
8. [Running the App](#8-running-the-app)
9. [Running Evaluations](#9-running-evaluations)
10. [Extending MacroMind](#10-extending-macromind)
11. [Iteration Log](#11-iteration-log)

---

## 1. What MacroMind Does

MacroMind is a **RAG-powered meal planning assistant**. You tell it:

- Your daily calorie and macro targets (protein / carbs / fat in grams)
- Optional daily food budget
- Dietary restrictions (vegetarian, vegan, gluten-free, etc.)
- What ingredients you already have at home

And it returns a **2–3 meal plan** for the day, with trade-off explanations
and a shopping list of what you still need to buy.

The interesting part is the **three pipeline variants** it supports, which
lets you A/B-test whether retrieval + reranking actually improves results over
just asking the LLM cold.

---

## 2. Architecture Overview

```
User Input (Streamlit)
        │
        ▼
  config.py  ──── selects variant ────▶ VARIANT=baseline
        │                               VARIANT=rag
        │                               VARIANT=rag-rerank
        ▼
    rag.py  (run_pipeline)
   ┌────────────────────────────────────────────────────┐
   │  baseline      │  rag            │  rag-rerank     │
   │                │                 │                 │
   │  No retrieval  │  embed query    │  embed query    │
   │                │  query ChromaDB │  query ChromaDB │
   │                │  top-20 results │  top-20 results │
   │                │                 │  scoring.py     │
   │                │                 │  rerank → top-5 │
   │  constraints   │  recipes +      │  recipes +      │
   │  → GPT-4o      │  constraints    │  scores         │
   │                │  → GPT-4o       │  → GPT-4o       │
   └────────────────────────────────────────────────────┘
        │
        ▼
   prompts.py  ──── formats the right prompt for each variant
        │
        ▼
   GPT-4o response
        │
        ▼
   eval.py ── logs metrics (latency, tokens, cost, macro accuracy)
        │
        ▼
   Streamlit renders meal plan + metrics
```

**ChromaDB** stores recipe embeddings. It is populated once by `ingest.py`.
The Streamlit sidebar has an "Ingest Data" button that calls `ingest.py`
programmatically so you never need to open a terminal for day-to-day use.

---

## 3. File-by-File Explanation

### `config.py`
Central configuration. Everything that might be tweaked as an experiment
parameter lives here:
- `OPENAI_API_KEY` — read from `.env` file or environment variable
- `OPENAI_MODEL` — default `gpt-4o`
- `EMBEDDING_MODEL` — `all-MiniLM-L6-v2` (fast, 384-dim, runs on CPU)
- `CHROMA_DB_PATH` — where ChromaDB stores its files on disk
- `TOP_K_RETRIEVAL` — how many candidates to pull from ChromaDB (default 20)
- `TOP_K_RERANK` — how many to pass to GPT-4o after reranking (default 5)
- `W1, W2, W3` — scoring weights (see §6)
- `VARIANT` — which pipeline mode to default to

### `ingest.py`
One-shot data pipeline. Run it once (or after updating the CSV).

Steps:
1. Load `data/recipes_sample.csv`
2. For each recipe, parse the `NER` column (pipe-separated ingredient names)
3. Match each NER token against `data/usda_common.json` using fuzzy substring matching
4. Sum macros across all matched ingredients (assuming ~100g per ingredient)
5. Detect dietary tags (vegetarian / vegan / gluten-free) from ingredient list
6. Build embedding text: `"{title}. Ingredients: {list}. Key items: {NER}"`
7. Batch-embed with `sentence-transformers`
8. Upsert into ChromaDB (IDs are stable: `recipe_0`, `recipe_1`, …)

**Why 100g per ingredient?** It's a known simplification. A production system
would parse quantities ("2 cups", "400g") using a library like `ingredient-parser`.
For MVP purposes this gives a reasonable ballpark for relative macro comparison.

### `rag.py`
The core pipeline. Public entry point: `run_pipeline(user_constraints, variant)`.

Returns a `PipelineResult` dataclass with:
- `response_text` — the LLM's meal plan
- `recipes_used` — list of recipe dicts that were passed to the LLM
- Token counts, latency, cost estimate

Key internals:
- `build_query_string()` — turns macro targets + pantry → keyword query
- `retrieve_candidates()` — embeds query, queries ChromaDB, optionally filters by dietary tags
- `call_llm()` — single OpenAI chat completion call

### `scoring.py`
The reranking math. Three components:

| Component | What it measures |
|-----------|-----------------|
| `macro_deviation` | How far the recipe's estimated macros are from the user's targets (normalised, scale-free) |
| `budget_overshoot` | How much the recipe exceeds the budget as a fraction of budget |
| `ingredient_overlap` | Fraction of recipe ingredients already in the user's pantry |

Combined: `score = -W1·dev - W2·overshoot + W3·overlap`

Higher score = better fit. The sign convention means:
- Deviations and overshoots are **penalised** (negative weight)
- Ingredient overlap is **rewarded** (positive weight)

### `prompts.py`
Every string that gets sent to GPT-4o lives here. This is intentional — keeping
prompts in one place makes it easy to A/B-test wording without touching pipeline
logic. See the `PROMPT CATALOG` section at the top of that file.

`format_recipe_block()` is a helper that converts a list of recipe dicts into
a numbered text block suitable for pasting into the LLM prompt.

### `eval.py`
Lightweight evaluation layer. Every pipeline run can be logged with `log_result()`,
which writes a JSON record to `eval_log.jsonl`.

Metrics captured:
- Latency (seconds)
- Token usage (prompt / completion / total)
- Estimated cost (USD)
- Macro accuracy (heuristic: parse the LLM's output text for numbers like "Protein: 148g")
- Constraint violations (heuristic: check if the LLM output mentions a restricted ingredient)

`compare_variants()` aggregates the log and returns average metrics per variant,
which `print_summary()` renders as a comparison table.

### `app.py`
Streamlit single-page app.

Layout:
- **Sidebar**: variant selector, Ingest Data button, API key input
- **Main form**: macro inputs, budget toggle, restriction multi-select, pantry textarea
- **Results**: LLM meal plan, collapsible recipe list with scores, metrics row,
  cumulative variant comparison table

---

## 4. Data Files

### `data/usda_common.json`
54 common ingredients with macros **per 100g**:
```json
{
  "chicken breast": { "calories": 165, "protein": 31.0, "carbs": 0.0, "fat": 3.6 },
  ...
}
```
Matching in `ingest.py` is case-insensitive substring: a recipe NER token like
`"chicken breast"` matches the key `"chicken breast"` directly; `"breast of chicken"`
would still match via substring.

### `data/recipes_sample.csv`
50 hand-crafted realistic recipes with columns:

| Column | Description |
|--------|-------------|
| `title` | Recipe name |
| `ingredients` | Pipe-separated ingredient strings with quantities |
| `directions` | Step-by-step cooking instructions |
| `NER` | Pipe-separated canonical ingredient names (for macro lookup + embedding) |

The `NER` column uses clean names that match the USDA JSON keys as closely as
possible — this is what drives macro estimation and dietary tag detection.

---

## 5. How the Three Variants Work

### Variant: `baseline`
- **No ChromaDB query at all.**
- Sends only the user's constraints to GPT-4o using `BASELINE_USER_PROMPT`.
- GPT-4o invents meals from its training knowledge.
- Fastest (one LLM call, no embedding).
- Least controlled — macro accuracy depends entirely on the model's calibration.
- **Use when**: you want a zero-retrieval baseline for comparison.

### Variant: `rag`
- Embeds a keyword query built from the user's constraints.
- Retrieves top-20 nearest recipes from ChromaDB via cosine similarity.
- Passes all 20 to GPT-4o using `RAG_USER_PROMPT`.
- GPT-4o picks 2–3 recipes from the pool.
- More grounded than baseline, but the context window is large (20 recipes).
- **Use when**: you trust the embedding model to surface relevant recipes and
  want the LLM to do its own selection.

### Variant: `rag-rerank`
- Same retrieval as `rag` (top-20).
- Applies `scoring.py` to rerank candidates using macro fit + budget + pantry overlap.
- Passes only top-5 to GPT-4o using `RAG_RERANK_USER_PROMPT`.
- Smaller prompt → lower cost; pre-filtering → more relevant pool for LLM.
- **Use when**: you want the best macro-aware recommendations (recommended default).

---

## 6. The Scoring Formula

```
score = -W1 × macro_deviation - W2 × budget_overshoot + W3 × ingredient_overlap
```

**`macro_deviation`** (scale-free, ≥ 0):
```
mean over {calories, protein, carbs, fat} of:
    |recipe_value - target_value| / max(target_value, 1)
```
A recipe that perfectly matches targets on all four dims scores 0.
A recipe that's 50% off on every dim scores 0.5.

**`budget_overshoot`** (≥ 0):
```
max(0, (recipe_cost - budget) / budget)
```
Recipes within budget score 0. If no budget is set, this is always 0.

**`ingredient_overlap`** ([0, 1]):
```
count of recipe NER tokens matching user pantry / total NER tokens
```
Matching is substring-based: pantry item "chicken" matches recipe token "chicken breast".

**Default weights** (in `config.py`):
- `W1 = 1.0` (macro fit is most important)
- `W2 = 0.5` (budget overshoot is penalised but less strongly)
- `W3 = 0.3` (pantry overlap is a bonus)

To run experiments with different weights, change `W1/W2/W3` in `config.py`
and re-run. The eval log will capture each run's metrics separately.

---

## 7. Getting Started (Quick Start)

### Prerequisites
- Python 3.11+
- An OpenAI API key with GPT-4o access

### Installation

```bash
# Clone and enter the repo
cd MacroMind

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
cp .env.example .env           # or create .env manually
# Then edit .env and add: OPENAI_API_KEY=sk-...
```

**.env file format:**
```
OPENAI_API_KEY=sk-your-key-here
MACROMIND_VARIANT=rag-rerank   # optional: baseline | rag | rag-rerank
```

### First run: build the recipe index

```bash
python ingest.py
```

You only need to do this once. It takes ~30 seconds on a MacBook (CPU).
Add `--reset` to wipe and rebuild: `python ingest.py --reset`

---

## 8. Running the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

**In the sidebar:**
1. Paste your OpenAI API key (or set it in `.env`)
2. Select a variant (rag-rerank is recommended)
3. Click "Ingest Data" if you haven't already

**In the main form:**
1. Set your daily calorie and macro targets
2. Optionally set a budget
3. Select dietary restrictions
4. Type what's in your pantry (comma-separated)
5. Click "Generate My Meal Plan"

---

## 9. Running Evaluations

To compare variants systematically:

```bash
# Run each variant from the command line (useful for scripting)
MACROMIND_VARIANT=baseline   python -c "
from rag import run_pipeline; from eval import log_result
constraints = {'calories':2000,'protein':150,'carbs':200,'fat':65,'budget':20,'dietary_restrictions':[],'available_ingredients':['chicken','broccoli']}
result = run_pipeline(constraints)
log_result(result, constraints)
"

# Repeat for rag and rag-rerank, then:
python eval.py --summary
```

Or just use the Streamlit app across all three variants — every submission
is automatically logged to `eval_log.jsonl`.

---

## 10. Extending MacroMind

### Add more recipes
Append rows to `data/recipes_sample.csv` following the existing format, then
re-run `python ingest.py --reset`.

### Add more USDA ingredients
Add entries to `data/usda_common.json` with the format:
```json
"ingredient name": { "calories": N, "protein": N, "carbs": N, "fat": N }
```
Values are per 100g. Re-run ingestion after changes.

### Tune the scoring weights
Edit `W1_MACRO_DEVIATION`, `W2_BUDGET_OVERSHOOT`, `W3_INGREDIENT_OVERLAP` in
`config.py`. Run the app with each weight set and compare `eval_log.jsonl`.

### Change the LLM prompts
All prompts are in `prompts.py`. Edit the relevant constant and restart the app.
No other files need to change.

### Add a new variant
1. Add a new branch in `rag.py → run_pipeline()`
2. Add a new prompt in `prompts.py`
3. Add the variant name to the radio options in `app.py → sidebar()`

### Parse actual ingredient quantities
Replace the 100g-per-ingredient assumption in `ingest.py → estimate_recipe_macros()`
with a quantity parser. The `ingredient-parser` PyPI package is a good starting point.

### Add price estimation
The `estimated_cost` field in ChromaDB metadata is currently `0.0`. Hook into the
Kroger, Instacart, or Open Food Facts API to populate real prices, then set a budget
in the UI.

---

## 11. Iteration Log

### v1.0 — 2026-03-31 — Initial MVP

**What was built:**
- Full project scaffold: `config.py`, `ingest.py`, `rag.py`, `scoring.py`,
  `prompts.py`, `eval.py`, `app.py`
- 50-recipe sample dataset (`recipes_sample.csv`) hand-crafted to cover a range
  of protein sources, dietary tags, and cuisines
- 54-ingredient USDA nutrition lookup (`usda_common.json`)
- All three pipeline variants implemented end-to-end
- Streamlit app with sidebar controls, form, and metrics display
- Evaluation logger with cumulative variant comparison table

**Key design decisions:**
- Prompts in a separate file (`prompts.py`) to decouple iteration on wording
  from iteration on pipeline logic
- Lazy singleton pattern in `rag.py` for the embedding model, ChromaDB client,
  and OpenAI client — avoids reloading 400MB model on every Streamlit rerun
- `PipelineResult` dataclass as the contract between `rag.py`, `app.py`,
  and `eval.py` — each module only needs to know about this one type
- Macro estimation assumes 100g per ingredient as a pragmatic MVP shortcut;
  documented clearly so a future contributor knows exactly where to improve it
- Ingredient matching uses bidirectional substring ("chicken" matches
  "chicken breast" AND "chicken breast" matches "chicken") for maximum recall
  given the small USDA lookup table

**Known limitations:**
- Macro estimates are rough (no quantity parsing)
- Macro accuracy metric is a heuristic (regex parse of LLM output text)
- No authentication or multi-user support
- `estimated_cost` is always 0 (price API not integrated)
- ChromaDB's metadata filtering only supports `$eq` / `$in` — the dietary
  restriction filter in `rag.py` is done in Python after retrieval, not at
  the DB level

**Next steps (v1.1 ideas):**
- Parse ingredient quantities with `ingredient-parser` library
- Add price lookup via Open Food Facts API
- Add a structured JSON output mode (`response_format={"type":"json_object"}`)
  to make macro accuracy parsing reliable
- Add unit tests for `scoring.py` and `ingest.py`
- Deploy on Streamlit Community Cloud
