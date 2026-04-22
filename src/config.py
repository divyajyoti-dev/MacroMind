from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
RECIPES_PATH = DATA_DIR / "recipes" / "sample_recipes.json"
CLEANED_RECIPES_PATH = DATA_DIR / "cleaned_recipes.json"
USDA_CACHE_PATH = DATA_DIR / "usda_cache" / "nutrition_cache.json"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"

CHROMA_COLLECTION_NAME = "macromind_recipes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.0-flash"

USDA_NUTRIENT_IDS = {
    "calories": 1008,
    "protein": 1003,
    "fat": 1004,
    "carbs": 1005,
}

# BLS CPI-based estimates (USD per 100g)
PRICE_PER_100G = {
    "chicken breast": 0.44,
    "chicken thigh": 0.33,
    "ground beef": 0.60,
    "beef steak": 0.90,
    "salmon": 1.10,
    "tuna": 0.55,
    "shrimp": 0.88,
    "eggs": 0.18,
    "whole milk": 0.06,
    "cheddar cheese": 0.55,
    "greek yogurt": 0.22,
    "butter": 0.72,
    "olive oil": 1.00,
    "white rice": 0.09,
    "brown rice": 0.11,
    "pasta": 0.14,
    "bread": 0.22,
    "oats": 0.10,
    "flour": 0.06,
    "sugar": 0.07,
    "brown sugar": 0.08,
    "potato": 0.11,
    "sweet potato": 0.16,
    "broccoli": 0.22,
    "spinach": 0.33,
    "tomato": 0.18,
    "onion": 0.09,
    "garlic": 0.44,
    "carrot": 0.11,
    "lentils": 0.14,
    "black beans": 0.13,
    "chickpeas": 0.15,
    "tofu": 0.28,
    "almonds": 1.10,
    "peanut butter": 0.33,
}
