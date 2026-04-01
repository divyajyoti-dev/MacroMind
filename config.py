"""
MacroMind Configuration
=======================
Central config for API keys, model settings, ChromaDB paths, scoring weights,
and experiment variant flags.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── Model Settings ────────────────────────────────────────────────────────────
# Groq free tier models (no billing required):
#   llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
GROQ_MODEL: str = "llama-3.3-70b-versatile"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_DB_PATH: str = "./chroma_db"
COLLECTION_NAME: str = "recipes"

# ── Data Paths ────────────────────────────────────────────────────────────────
RECIPES_CSV: str = "./data/recipes_sample.csv"
USDA_JSON: str = "./data/usda_common.json"

# ── RAG Settings ──────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = 20   # candidates pulled from ChromaDB
TOP_K_RERANK: int = 5       # top recipes passed to GPT-4o after reranking

# ── Scoring Weights ───────────────────────────────────────────────────────────
# score = -W1 * macro_deviation - W2 * budget_overshoot + W3 * ingredient_overlap
W1_MACRO_DEVIATION: float = 1.0
W2_BUDGET_OVERSHOOT: float = 0.5
W3_INGREDIENT_OVERLAP: float = 0.3

# ── Experiment Variants ───────────────────────────────────────────────────────
# "baseline"   → skip retrieval, send constraints directly to GPT-4o
# "rag"        → retrieve from ChromaDB, pass top-K to LLM (no reranking)
# "rag-rerank" → retrieve + custom scoring rerank, then pass to LLM
VARIANT: str = os.getenv("MACROMIND_VARIANT", "rag-rerank")

# ── Evaluation / Logging ──────────────────────────────────────────────────────
EVAL_LOG_PATH: str = "./eval_log.jsonl"

# ── Cost Estimation (USD per token, approximate) ──────────────────────────────
# Groq is free tier — effectively $0, but track tokens anyway
GROQ_INPUT_COST_PER_1K: float = 0.0
GROQ_OUTPUT_COST_PER_1K: float = 0.0
