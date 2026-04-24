import base64

from groq import Groq
from google import genai as google_genai
from google.genai import types as genai_types

from src.config import GROQ_MODEL, GEMINI_MODEL
from src.retriever import UserConstraints, SearchResult
from prompts.meal_planner import (
    SYSTEM_MEAL_PLANNER,
    BASELINE_USER_PROMPT,
    RAG_USER_PROMPT,
    RAG_RERANK_USER_PROMPT,
    format_recipe_block,
)


def _format_constraints(constraints: UserConstraints) -> dict:
    return {
        "calories": int(constraints.calories),
        "protein_g": constraints.protein_g,
        "carbs_g": constraints.carbs_g,
        "fat_g": constraints.fat_g,
        "budget_usd": constraints.budget_usd,
        "available_ingredients": ", ".join(constraints.available_ingredients) or "any",
        "dietary_tags": ", ".join(constraints.dietary_tags) or "none",
        "allergy_tags": ", ".join(constraints.allergy_tags) if constraints.allergy_tags else "none",
        "cultural_dietary": ", ".join(constraints.cultural_dietary) if constraints.cultural_dietary else "none",
    }


# --- Groq (primary) ---

def generate_baseline_plan_groq(
    constraints: UserConstraints,
    api_key: str,
    model_name: str = GROQ_MODEL,
) -> str:
    client = Groq(api_key=api_key)
    user_msg = BASELINE_USER_PROMPT.format(**_format_constraints(constraints))
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_MEAL_PLANNER},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
        max_tokens=1500,
    )
    return response.choices[0].message.content


def generate_meal_plan_groq(
    constraints: UserConstraints,
    ranked_recipes: list[SearchResult],
    api_key: str,
    model_name: str = GROQ_MODEL,
    use_rerank_prompt: bool = False,
) -> str:
    client = Groq(api_key=api_key)
    recipe_block = format_recipe_block(ranked_recipes[:5], include_score=use_rerank_prompt)
    prompt_template = RAG_RERANK_USER_PROMPT if use_rerank_prompt else RAG_USER_PROMPT
    user_msg = prompt_template.format(
        **_format_constraints(constraints),
        recipe_block=recipe_block,
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_MEAL_PLANNER},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
        max_tokens=1500,
    )
    return response.choices[0].message.content


# --- Gemini (fallback) ---

def generate_baseline_plan(
    constraints: UserConstraints,
    api_key: str,
    model_name: str = GEMINI_MODEL,
) -> str:
    client = google_genai.Client(api_key=api_key)
    user_msg = BASELINE_USER_PROMPT.format(**_format_constraints(constraints))
    response = client.models.generate_content(
        model=model_name,
        contents=user_msg,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_MEAL_PLANNER,
            temperature=0.4,
            max_output_tokens=1500,
        ),
    )
    return response.text


def extract_ingredients_from_image(image_bytes: bytes, api_key: str) -> str:
    """Send a fridge/pantry photo to Gemini vision and return a comma-separated ingredient list."""
    client = google_genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            "List the food ingredients visible in this image, comma-separated. "
            "Only list recognisable food items. Do not include brands, packaging, or non-food items.",
        ],
    )
    return response.text.strip()


def generate_meal_plan(
    constraints: UserConstraints,
    ranked_recipes: list[SearchResult],
    api_key: str,
    model_name: str = GEMINI_MODEL,
    use_rerank_prompt: bool = False,
) -> str:
    client = google_genai.Client(api_key=api_key)
    recipe_block = format_recipe_block(ranked_recipes[:5], include_score=use_rerank_prompt)
    prompt_template = RAG_RERANK_USER_PROMPT if use_rerank_prompt else RAG_USER_PROMPT
    user_msg = prompt_template.format(
        **_format_constraints(constraints),
        recipe_block=recipe_block,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=user_msg,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_MEAL_PLANNER,
            temperature=0.4,
            max_output_tokens=1500,
        ),
    )
    return response.text
