# MacroMind — Development Rules

## Workflow
- **Plan before coding.** Propose the approach, wait for approval, then implement.
- **All LLM prompts live in `prompts/`** as named Python constants. No inline strings in `src/`.
- **Do not commit or push without explicit instruction.**
- **Run `/graphify` after any architecture change** (new modules, new pipeline stages, new data sources).

## LLM Providers
- Primary: Groq (`llama-3.3-70b-versatile`) — free tier, no billing required
- Fallback: Gemini 2.0 Flash — requires `GOOGLE_API_KEY`
- Both must remain supported.

## Evaluation Notebook
- `notebooks/divya_evaluation.ipynb` must run fully offline — no LLM API key required.
- Retrieval and reranking metrics (macro MAPE, Precision@5, waste fraction) are local operations.

## Corpus
- Source: RecipeNLG (2.2M total). Indexed sample: ~9,999 recipes.
- Curated set: 65 recipes in `data/recipes/sample_recipes.json` (ground truth / unit tests).
- Gram weights default to 100g per ingredient for RecipeNLG records (no weights in source).

## Branches
- Working branch: `MacroMind_master`
- Base: `main` — never commit here.

## Token Efficiency
- Use `/caveman` when sessions run long — cuts token usage ~75% while keeping full accuracy.
- Available skills: `/simplify`, `/security-review`, `/graphify`
- Sub-agents (`security-reviewer`, `verifier`, `progress-tracker`) use Haiku — cheap and fast.

## Agents
- `/agents security-reviewer` — reviews a file for security issues (run after writing new src/ files)
- `/agents verifier` — checks acceptance criteria for the current issue
- `progress-tracker` — runs automatically at session end, updates PROGRESS.md
