"""
Google Gemini integration for meal plan generation.
Uses the new google-genai SDK (v1 API).
"""
from google import genai
from google.genai import types

from src.config import GOOGLE_API_KEY, GEMINI_MODEL
from src.reranker import UserConstraints, RankedResult

SYSTEM_PROMPT = """You are MacroMind, a precision nutrition assistant.
Your job is to create personalised daily meal plans that precisely hit the user's macro targets
while respecting their budget, available ingredients, and dietary preferences.

Always:
- Select meals for breakfast, lunch, dinner, and one snack
- Calculate total daily macros and compare against targets
- Explain briefly why each meal fits the constraints
- Flag any constraint violations transparently
- Be concise and actionable
"""


def _get_client(api_key: str = GOOGLE_API_KEY) -> genai.Client:
    return genai.Client(api_key=api_key)


def build_meal_plan_prompt(
    constraints: UserConstraints,
    ranked_recipes: list[RankedResult],
) -> str:
    """Construct the prompt injecting constraints and retrieved recipes."""
    constraint_block = (
        f"Macro targets per day:\n"
        f"  Calories: {constraints.calories} kcal\n"
        f"  Protein:  {constraints.protein_g} g\n"
        f"  Carbs:    {constraints.carbs_g} g\n"
        f"  Fat:      {constraints.fat_g} g\n"
        f"Daily budget: ${constraints.budget_usd:.2f}\n"
        f"Available ingredients: {', '.join(constraints.available_ingredients) or 'any'}\n"
        f"Dietary requirements: {', '.join(constraints.dietary_tags) or 'none'}\n"
    )

    recipe_block_lines = []
    for i, r in enumerate(ranked_recipes, 1):
        m = r.metadata
        recipe_block_lines.append(
            f"{i}. {r.name}\n"
            f"   Macros: {m.get('calories', 0):.0f} kcal | "
            f"P {m.get('protein', 0):.0f}g | "
            f"C {m.get('carbs', 0):.0f}g | "
            f"F {m.get('fat', 0):.0f}g\n"
            f"   Cost: ${m.get('cost_usd', 0):.2f} | "
            f"Ingredients: {m.get('ingredients', '').replace('|', ', ')}\n"
            f"   Tags: {m.get('tags', '').replace('|', ', ')}\n"
        )

    recipe_block = "\n".join(recipe_block_lines)

    return (
        f"USER CONSTRAINTS:\n{constraint_block}\n"
        f"RETRIEVED RECIPE CANDIDATES:\n{recipe_block}\n"
        f"Please build a complete daily meal plan using the candidates above (or subsets of them).\n"
        f"Calculate total macros for the day and compare to targets."
    )


def generate_meal_plan(
    constraints: UserConstraints,
    ranked_recipes: list[RankedResult],
    model_name: str = GEMINI_MODEL,
    api_key: str = GOOGLE_API_KEY,
    max_tokens: int = 1500,
) -> str:
    """Call Gemini and return the generated meal plan as a string."""
    client = _get_client(api_key)
    prompt = build_meal_plan_prompt(constraints, ranked_recipes)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text


def generate_baseline_plan(
    constraints: UserConstraints,
    model_name: str = GEMINI_MODEL,
    api_key: str = GOOGLE_API_KEY,
    max_tokens: int = 1500,
) -> str:
    """
    Baseline variant: ask Gemini to build a meal plan from scratch
    without any retrieved recipes. Used for A/B comparison.
    """
    client = _get_client(api_key)

    prompt = (
        f"USER CONSTRAINTS:\n"
        f"Macro targets per day:\n"
        f"  Calories: {constraints.calories} kcal\n"
        f"  Protein:  {constraints.protein_g} g\n"
        f"  Carbs:    {constraints.carbs_g} g\n"
        f"  Fat:      {constraints.fat_g} g\n"
        f"Daily budget: ${constraints.budget_usd:.2f}\n"
        f"Available ingredients: {', '.join(constraints.available_ingredients) or 'any'}\n"
        f"Dietary requirements: {', '.join(constraints.dietary_tags) or 'none'}\n\n"
        f"Build a complete daily meal plan (breakfast, lunch, dinner, snack) from your own knowledge.\n"
        f"Calculate total daily macros and compare to targets."
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text
