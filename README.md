# MacroMind 🥗

> AI-powered meal planning using RAG + GPT-4o.
> Tell it your macros, budget, and pantry — get a personalised daily meal plan.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python ingest.py       # build the recipe index (run once)
streamlit run app.py   # launch the app
```

## Three Pipeline Variants

| Variant | How it works |
|---------|-------------|
| `baseline` | GPT-4o invents meals from your constraints (no retrieval) |
| `rag` | Retrieve top-20 recipes from ChromaDB → GPT-4o selects |
| `rag-rerank` | Retrieve top-20 → rerank by macro fit + budget + pantry overlap → GPT-4o selects from top-5 |

Switch variants in the Streamlit sidebar. All runs are logged to `eval_log.jsonl` for comparison.

## Project Structure

```
MacroMind/
├── app.py              # Streamlit frontend
├── ingest.py           # Data ingestion + embedding pipeline
├── rag.py              # Retrieval + reranking + LLM generation
├── scoring.py          # Reranking scoring function
├── prompts.py          # All LLM prompt templates
├── config.py           # Variant flags, weights, API keys
├── eval.py             # Evaluation: compare variants, log metrics
├── GUIDE.md            # Full developer + user manual
├── data/
│   ├── recipes_sample.csv   # 50 sample recipes
│   └── usda_common.json     # 54 ingredients with macros per 100g
└── requirements.txt
```

See [GUIDE.md](GUIDE.md) for full architecture explanation, scoring formula details,
and how to extend the system.
