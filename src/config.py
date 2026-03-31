"""
Central configuration for MacroMind.
All paths, API keys, and constants live here so notebooks import from one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API keys ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
USDA_API_KEY: str = os.getenv("USDA_API_KEY", "DEMO_KEY")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
RECIPES_PATH: Path = DATA_DIR / "recipes" / "sample_recipes.json"
USDA_CACHE_PATH: Path = DATA_DIR / "usda_cache" / "nutrition_cache.json"
CHROMA_DB_PATH: str = str(DATA_DIR / "chroma_db")

# ── Vector store ──────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = "macromind_recipes"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # 22 MB, CPU-friendly, 384-dim

# ── LLM ───────────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-2.0-flash"   # fast, works with new SDK

# ── USDA nutrient IDs ─────────────────────────────────────────────────────────
USDA_NUTRIENT_IDS: dict[str, int] = {
    "calories": 208,
    "protein":  203,
    "fat":      204,
    "carbs":    205,
}

# ── Hardcoded price table (USD per 100 g) ─────────────────────────────────────
# Sourced from BLS CPI averages (2024) and typical supermarket prices.
PRICE_PER_100G: dict[str, float] = {
    "chicken breast":   0.35,
    "ground beef":      0.45,
    "salmon":           0.90,
    "tuna":             0.30,
    "eggs":             0.20,
    "tofu":             0.25,
    "greek yogurt":     0.22,
    "cottage cheese":   0.18,
    "whole milk":       0.10,
    "cheddar cheese":   0.50,
    "brown rice":       0.08,
    "white rice":       0.07,
    "quinoa":           0.20,
    "oats":             0.09,
    "pasta":            0.10,
    "whole wheat bread":0.15,
    "black beans":      0.12,
    "chickpeas":        0.12,
    "lentils":          0.11,
    "sweet potato":     0.13,
    "broccoli":         0.15,
    "spinach":          0.18,
    "tomato":           0.14,
    "onion":            0.08,
    "garlic":           0.20,
    "avocado":          0.40,
    "banana":           0.10,
    "apple":            0.14,
    "olive oil":        0.60,
    "peanut butter":    0.25,
}
