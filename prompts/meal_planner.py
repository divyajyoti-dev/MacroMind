# v1
SYSTEM_MEAL_PLANNER = """\
You are a precision nutrition assistant. Given a user's macro targets, budget, \
and dietary restrictions, you create a one-day meal plan with breakfast, lunch, \
dinner, and one snack.

Rules:
- Always output all four meals: Breakfast, Lunch, Dinner, Snack.
- Calculate and report total daily macros (calories, protein, carbs, fat).
- Briefly explain each meal choice in one sentence.
- If a constraint cannot be met, flag it transparently rather than hiding it.
- Keep responses concise and structured.\
"""

# v1
BASELINE_USER_PROMPT = """\
Create a one-day meal plan for the following targets:

Calories: {calories} kcal
Protein: {protein_g}g
Carbs: {carbs_g}g
Fat: {fat_g}g
Budget: ${budget_usd:.2f} per day
Available ingredients: {available_ingredients}
Dietary requirements: {dietary_tags}

Do not reference any specific recipes — use your knowledge to suggest meals.\
"""

# v1
RAG_USER_PROMPT = """\
Create a one-day meal plan for the following targets:

Calories: {calories} kcal
Protein: {protein_g}g
Carbs: {carbs_g}g
Fat: {fat_g}g
Budget: ${budget_usd:.2f} per day
Available ingredients: {available_ingredients}
Dietary requirements: {dietary_tags}

Use the following retrieved recipes as your primary source. \
You may adapt them but should stay close to their ingredient lists:

{recipe_block}\
"""

# v1
RAG_RERANK_USER_PROMPT = """\
Create a one-day meal plan for the following targets:

Calories: {calories} kcal
Protein: {protein_g}g
Carbs: {carbs_g}g
Fat: {fat_g}g
Budget: ${budget_usd:.2f} per day
Available ingredients: {available_ingredients}
Dietary requirements: {dietary_tags}

The following recipes have been pre-scored for how well they fit your constraints \
(higher score = better fit). Prefer higher-scored options when building the plan:

{recipe_block}\
"""


def format_recipe_block(ranked_recipes, include_score: bool = False) -> str:
    lines = []
    for i, r in enumerate(ranked_recipes, start=1):
        meta = getattr(r, "metadata", {})
        name = getattr(r, "name", meta.get("name", "Unknown"))
        calories = meta.get("calories", "?")
        protein = meta.get("protein", "?")
        carbs = meta.get("carbs", "?")
        fat = meta.get("fat", "?")
        tags = meta.get("tags", "")
        ingredients = meta.get("ingredients", "").replace("|", ", ")

        score_part = ""
        if include_score:
            score = getattr(r, "score", None)
            if score is not None:
                score_part = f" [score: {score:.3f}]"

        lines.append(
            f"{i}. {name}{score_part}\n"
            f"   Macros: {calories} kcal | {protein}g protein | {carbs}g carbs | {fat}g fat\n"
            f"   Ingredients: {ingredients}\n"
            f"   Tags: {tags if tags else 'none'}"
        )
    return "\n\n".join(lines)
