"""
MacroMind Prompts
=================
All LLM prompt templates live here.

ITERATION LOG
─────────────
v1.0 (2026-03-31): Initial prompts for baseline, RAG, and RAG+rerank variants.

HOW TO ADD A NEW PROMPT
─────────────────────────
1. Define a new string constant (UPPER_SNAKE_CASE).
2. Use {placeholders} for dynamic values.
3. Format at call-time:  SOME_PROMPT.format(key=value)
4. Add it to this file's docstring with a brief description.

PROMPT CATALOG
──────────────
- SYSTEM_MEAL_PLANNER       Base system role for all variants.
- BASELINE_USER_PROMPT      Baseline: no recipes, just constraints → LLM invents meals.
- RAG_USER_PROMPT           RAG: retrieved recipes + constraints → LLM selects.
- RAG_RERANK_USER_PROMPT    RAG+rerank: pre-scored top-5 recipes → LLM selects & explains.
- QUERY_BUILDER_PROMPT      (optional) Asks GPT to expand user constraints into a richer
                            search query string for ChromaDB.
"""

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_MEAL_PLANNER = """You are MacroMind, an expert nutrition coach and meal planner.
Your job is to design a practical, realistic daily meal plan that meets the user's
macro targets, respects their dietary restrictions, fits their budget, and uses
ingredients they already have.

When responding, always:
1. Select 2-3 meals that together hit the daily calorie and macro targets as closely as possible.
2. Explain trade-offs clearly (e.g. "Recipe A is 20g over on carbs but uses 3 of your pantry items").
3. Provide a short shopping list of only the ingredients the user DOESN'T already have.
4. Keep the tone practical and encouraging — not preachy or overly clinical.
5. Format your response with clear sections: MEAL PLAN, MACRO SUMMARY, TRADE-OFFS, SHOPPING LIST.
"""

# ── Baseline Prompt (no retrieval) ───────────────────────────────────────────

BASELINE_USER_PROMPT = """Plan a daily meal schedule for me based on the following constraints.
Invent meals from scratch — do not use a recipe database.

USER CONSTRAINTS:
- Daily calorie target: {calories} kcal
- Protein target: {protein}g
- Carb target: {carbs}g
- Fat target: {fat}g
- Budget: {budget}
- Dietary restrictions: {dietary_restrictions}
- Available ingredients (already in my kitchen): {available_ingredients}

Create a 2–3 meal plan for the day. Each meal should have:
- Name
- Key ingredients
- Approximate macros (calories, protein, carbs, fat)
- Estimated cost

End with a combined MACRO SUMMARY and a SHOPPING LIST for ingredients I need to buy.
"""

# ── RAG Prompt (retrieved recipes, no reranking) ─────────────────────────────

RAG_USER_PROMPT = """I've retrieved the following recipes from my database that might suit the user's needs.
Choose the best 2–3 recipes to form a complete daily meal plan.

USER CONSTRAINTS:
- Daily calorie target: {calories} kcal
- Protein target: {protein}g
- Carb target: {carbs}g
- Fat target: {fat}g
- Budget: {budget}
- Dietary restrictions: {dietary_restrictions}
- Available ingredients (already in my kitchen): {available_ingredients}

RETRIEVED RECIPES (unranked):
{recipes}

Instructions:
1. Select the 2–3 best-fit recipes for a full day of eating.
2. Explain why you chose each recipe and what trade-offs exist.
3. List the combined macros for the whole day vs. the targets.
4. Produce a shopping list of ingredients NOT in the user's available list.
"""

# ── RAG + Rerank Prompt (pre-scored top-5 recipes) ───────────────────────────

RAG_RERANK_USER_PROMPT = """The following recipes have already been filtered and ranked by a scoring algorithm
based on macro fit, budget, and ingredient overlap with what the user already has.
Recipe #1 has the highest score.

USER CONSTRAINTS:
- Daily calorie target: {calories} kcal
- Protein target: {protein}g
- Carb target: {carbs}g
- Fat target: {fat}g
- Budget: {budget}
- Dietary restrictions: {dietary_restrictions}
- Available ingredients (already in my kitchen): {available_ingredients}

TOP RANKED RECIPES (best first):
{recipes}

Instructions:
1. Select the best combination of 2–3 recipes for a full daily meal plan.
   You may override the ranking if a lower-ranked recipe creates a better macro balance.
2. For each selected recipe, briefly explain the trade-offs.
3. Show the combined MACRO SUMMARY: target vs. actual for calories, protein, carbs, fat.
4. Produce a concise SHOPPING LIST of ingredients the user still needs to buy.
"""

# ── Optional: Query Expansion Prompt ─────────────────────────────────────────

QUERY_EXPANSION_PROMPT = """Convert the following user meal-planning constraints into a short, keyword-rich
search query (max 20 words) suitable for a semantic recipe search. Focus on protein sources,
cooking style, dietary tags, and key ingredients.

CONSTRAINTS:
- Calories: {calories} kcal
- Protein: {protein}g | Carbs: {carbs}g | Fat: {fat}g
- Dietary restrictions: {dietary_restrictions}
- Available ingredients: {available_ingredients}

Return ONLY the query string, no explanation.
"""

# ── Formatting helper ─────────────────────────────────────────────────────────

def format_recipe_block(recipes: list[dict], include_score: bool = False) -> str:
    """
    Converts a list of recipe dicts into a numbered text block for LLM prompts.

    Args:
        recipes: List of dicts with keys title, ingredients, estimated_macros,
                 and optionally score, estimated_cost.
        include_score: If True, include the rerank score in the block.

    Returns:
        Formatted multi-line string.
    """
    lines = []
    for i, r in enumerate(recipes, 1):
        macros = r.get("estimated_macros", {})
        line = (
            f"Recipe #{i}: {r.get('title', 'Unknown')}\n"
            f"  Ingredients: {r.get('ingredients', '')}\n"
            f"  Est. Macros — Cal: {macros.get('calories', '?')} kcal | "
            f"Protein: {macros.get('protein', '?')}g | "
            f"Carbs: {macros.get('carbs', '?')}g | "
            f"Fat: {macros.get('fat', '?')}g\n"
        )
        if include_score:
            line += f"  Rerank Score: {r.get('score', 0):.3f}\n"
        if r.get("estimated_cost"):
            line += f"  Est. Cost: ${r.get('estimated_cost', '?'):.2f}\n"
        lines.append(line)
    return "\n".join(lines)
