import { useState, useEffect } from "react";

const VARIANTS = {
  V1: {
    id: "V1",
    label: "Baseline LLM",
    sublabel: "No retrieval",
    color: "#6b7280",
    activeColor: "#9ca3af",
    accent: "border-gray-500",
    bg: "bg-gray-800",
    activeLayers: ["input", "llm", "output"],
  },
  V2: {
    id: "V2",
    label: "Standard RAG",
    sublabel: "Retrieve → Generate",
    color: "#3b82f6",
    activeColor: "#60a5fa",
    accent: "border-blue-500",
    bg: "bg-blue-900/20",
    activeLayers: ["input", "data", "retrieval", "llm", "output"],
  },
  V3: {
    id: "V3",
    label: "RAG + Reranking",
    sublabel: "Retrieve → Score → Generate",
    color: "#8b5cf6",
    activeColor: "#a78bfa",
    accent: "border-purple-500",
    bg: "bg-purple-900/20",
    activeLayers: ["input", "data", "retrieval", "reranker", "llm", "output"],
  },
  V4: {
    id: "V4",
    label: "RAG + Vision",
    sublabel: "Proposed future variant",
    color: "#22c55e",
    activeColor: "#4ade80",
    accent: "border-green-500",
    bg: "bg-green-900/20",
    activeLayers: ["input", "vision", "data", "retrieval", "reranker", "llm", "output"],
    isProposed: true,
  },
};

const TOOLTIPS = {
  input: "User provides: calorie & macro targets (protein/carbs/fat), optional budget, dietary restrictions (vegetarian, vegan, gluten-free), and available pantry ingredients via text input.",
  vision: "⚠️ Proposed (not yet implemented): fridge photo → ingredient list extraction via vision model. Would feed into constraint parser as available_ingredients.",
  query_builder: "src/retriever.py → build_query_text(). Converts structured constraints into a natural-language query like '2000 calorie meal high protein 150g with chicken broccoli'. Better for semantic search than raw JSON.",
  data_usda: "src/data_pipeline.py → USDA FoodData Central API. Fetches per-100g macros (calories, protein, fat, carbs) for each ingredient. Results cached in data/usda_cache/nutrition_cache.json to avoid repeat calls.",
  data_recipes: "data/recipes/sample_recipes.json — 65 hand-crafted recipes. Original proposal mentioned RecipeNLG (2.2M recipes); implementation uses a curated 65-recipe corpus. Each recipe: {id, name, ingredients[{name, grams}], tags, instructions, prep_time, cost}.",
  data_price: "src/config.py → PRICE_PER_100G table. Hardcoded BLS CPI-based prices for ~30 ingredients (USD per 100g). Original proposal mentioned Kroger Product API — not implemented; this static table is used instead.",
  embed: "src/recipe_processor.py → recipe_to_document() + SentenceTransformer('all-MiniLM-L6-v2'). Converts each recipe to prose ('Grilled Chicken Rice Bowl. Ingredients: chicken breast, brown rice... Tags: high-protein...') then encodes to 384-dim vector.",
  chroma: "ChromaDB (local persistent store). Collection: 'macromind_recipes'. Distance metric: cosine. Metadata stored per recipe: name, macros, cost, tags, ingredients (pipe-delimited), prep_time. Original proposal mentioned FAISS; ChromaDB was used.",
  retrieval: "src/retriever.py → semantic_search(). Encodes user query → 384-dim vector, queries ChromaDB for top-20 nearest recipes by cosine distance. Returns SearchResult[20] sorted ascending (lower distance = more similar).",
  reranker: "src/reranker.py → rerank(). Multi-objective scoring: score = 1.0 − (0.5 × MAPE_macros + 0.3 × budget_overshoot + 0.2 × waste_fraction). Hard penalty: −999 for dietary tag violations. Uses difflib fuzzy matching for ingredient overlap. Returns top-5 RankedResult.",
  llm: "src/llm_generator.py → generate_meal_plan_groq() / generate_baseline_plan_groq(). Default: Groq API (llama-3.3-70b-versatile, free tier). Gemini (gemini-2.0-flash) available as alternative. Temperature: 0.4, max_tokens: 1500. Original proposal used GPT-4o — switched to Groq for free access.",
  output: "Streamlit UI (app.py) renders: LLM-generated markdown meal plan (breakfast/lunch/dinner/snack), top-5 ranked recipe cards with macro breakdown, pipeline latency, and number of candidates retrieved.",
  eval: "src/evaluator.py. Metrics: macro MAPE per dimension, precision@k vs. ground-truth, total daily cost, within_budget flag, avg ingredient waste fraction, ROUGE-L score of generated plan vs. reference. Used in notebooks/divya_evaluation.ipynb.",
  variants: "Three variants implemented: V1 Baseline (LLM only, no retrieval), V2 RAG (retrieve top-5 by cosine similarity), V3 RAG+Rerank (retrieve top-20, constraint-aware rerank to top-5). V4 Vision is proposed but not yet implemented.",
};

function Tooltip({ text, children }) {
  const [show, setShow] = useState(false);
  return (
    <div className="relative" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      {show && (
        <div className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 bg-gray-900 border border-gray-600 rounded-lg p-3 text-xs text-gray-300 shadow-2xl pointer-events-none">
          <div className="leading-relaxed">{text}</div>
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-600" />
        </div>
      )}
    </div>
  );
}

function Arrow({ active, color = "#3b82f6", vertical = true, pulse = false }) {
  const baseColor = active ? color : "#374151";
  return (
    <div className={`flex ${vertical ? "flex-col" : "flex-row"} items-center justify-center`} style={{ height: vertical ? 32 : "auto", width: vertical ? "auto" : 32 }}>
      {vertical ? (
        <svg width="24" height="32" viewBox="0 0 24 32">
          <defs>
            <marker id={`arrowhead-${color.replace("#", "")}`} markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
              <path d="M0,0 L0,8 L8,4 z" fill={baseColor} />
            </marker>
          </defs>
          <line x1="12" y1="2" x2="12" y2="26"
            stroke={baseColor}
            strokeWidth="2"
            strokeDasharray={pulse && active ? "4 3" : "none"}
            markerEnd={`url(#arrowhead-${color.replace("#", "")})`}
            style={pulse && active ? { animation: "dash 1.2s linear infinite" } : {}}
          />
        </svg>
      ) : (
        <svg width="32" height="24" viewBox="0 0 32 24">
          <defs>
            <marker id={`arrowh-${color.replace("#", "")}`} markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
              <path d="M0,0 L0,8 L8,4 z" fill={baseColor} />
            </marker>
          </defs>
          <line x1="2" y1="12" x2="26" y2="12"
            stroke={baseColor}
            strokeWidth="2"
            strokeDasharray={pulse && active ? "4 3" : "none"}
            markerEnd={`url(#arrowh-${color.replace("#", "")})`}
            style={pulse && active ? { animation: "dash 1.2s linear infinite" } : {}}
          />
        </svg>
      )}
    </div>
  );
}

function Box({ id, icon, title, subtitle, badge, active, color = "#3b82f6", dashed = false, proposed = false, tooltipText, small = false }) {
  const isActive = active;
  return (
    <Tooltip text={tooltipText || title}>
      <div
        className={`
          relative rounded-xl border transition-all duration-500 cursor-default select-none
          ${small ? "px-3 py-2" : "px-4 py-3"}
          ${dashed ? "border-dashed" : "border-solid"}
          ${isActive
            ? `border-opacity-80 shadow-lg`
            : "border-gray-700 opacity-30 grayscale"
          }
        `}
        style={{
          borderColor: isActive ? color : undefined,
          backgroundColor: isActive ? `${color}14` : "#111827",
          boxShadow: isActive ? `0 0 20px ${color}22` : "none",
        }}
      >
        {proposed && (
          <span className="absolute -top-2 -right-2 text-xs bg-gray-700 text-gray-300 px-1.5 py-0.5 rounded-full border border-gray-600 font-mono">
            proposed
          </span>
        )}
        <div className="flex items-center gap-2">
          <span className={small ? "text-base" : "text-lg"}>{icon}</span>
          <div>
            <div className={`font-semibold ${small ? "text-xs" : "text-sm"} text-gray-100`}>{title}</div>
            {subtitle && <div className={`${small ? "text-[10px]" : "text-xs"} text-gray-400 mt-0.5`}>{subtitle}</div>}
          </div>
          {badge && (
            <span className="ml-auto text-[10px] font-mono px-1.5 py-0.5 rounded-md"
              style={{ backgroundColor: `${color}33`, color: color }}>
              {badge}
            </span>
          )}
        </div>
      </div>
    </Tooltip>
  );
}

export default function MacroMindArchitecture() {
  const [selected, setSelected] = useState("V3");
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setTick(n => n + 1), 50);
    return () => clearInterval(t);
  }, []);

  const v = VARIANTS[selected];
  const active = (layer) => v.activeLayers.includes(layer);
  const arrowColor = v.color;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 md:p-8 font-sans">
      <style>{`
        @keyframes dash { to { stroke-dashoffset: -14; } }
        @keyframes pulse-glow { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }
      `}</style>

      {/* Header */}
      <div className="mb-8 text-center">
        <div className="flex items-center justify-center gap-3 mb-2">
          <span className="text-3xl">🥗</span>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-green-400 bg-clip-text text-transparent">
            MacroMind
          </h1>
        </div>
        <p className="text-gray-400 text-sm">Multimodal RAG-Based Meal Planning System · INFO 290 GenAI @ UC Berkeley</p>
      </div>

      {/* Variant Selector */}
      <Tooltip text={TOOLTIPS.variants}>
        <div className="flex flex-wrap justify-center gap-2 mb-10">
          {Object.values(VARIANTS).map((vv) => (
            <button
              key={vv.id}
              onClick={() => setSelected(vv.id)}
              className={`
                relative px-4 py-2.5 rounded-xl border text-sm font-medium transition-all duration-300
                ${selected === vv.id
                  ? "border-opacity-100 text-white shadow-lg scale-105"
                  : "border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-500"
                }
              `}
              style={{
                borderColor: selected === vv.id ? vv.color : undefined,
                backgroundColor: selected === vv.id ? `${vv.color}22` : "#1f2937",
                boxShadow: selected === vv.id ? `0 0 16px ${vv.color}44` : "none",
              }}
            >
              <div className="font-bold text-xs mb-0.5" style={{ color: selected === vv.id ? vv.activeColor : undefined }}>
                {vv.id} {vv.isProposed && <span className="text-[10px] opacity-60">(proposed)</span>}
              </div>
              <div>{vv.label}</div>
              <div className="text-[10px] text-gray-400 mt-0.5">{vv.sublabel}</div>
            </button>
          ))}
        </div>
      </Tooltip>

      <div className="max-w-5xl mx-auto">

        {/* ── Layer 1: User Input ── */}
        <div className="flex justify-center mb-1">
          <div className="w-full max-w-2xl">
            <div className="text-[10px] uppercase tracking-widest text-gray-500 text-center mb-2 font-mono">Layer 1 — User Input</div>
            <Tooltip text={TOOLTIPS.input}>
              <div
                className={`rounded-xl border px-6 py-4 transition-all duration-500 ${active("input") ? "border-blue-500/60 shadow-lg" : "border-gray-700 opacity-30"}`}
                style={{ backgroundColor: active("input") ? "#3b82f614" : "#111827", boxShadow: active("input") ? "0 0 24px #3b82f622" : "none" }}
              >
                <div className="flex flex-wrap justify-center gap-4">
                  {[
                    { icon: "🔥", label: "Calories", val: "kcal target" },
                    { icon: "💪", label: "Macros", val: "P / C / F (g)" },
                    { icon: "💰", label: "Budget", val: "optional $/day" },
                    { icon: "🥦", label: "Dietary Tags", val: "veg, GF, vegan..." },
                    { icon: "🧺", label: "Pantry", val: "available ingredients" },
                  ].map(({ icon, label, val }) => (
                    <div key={label} className="flex flex-col items-center gap-1">
                      <span className="text-xl">{icon}</span>
                      <span className="text-xs font-semibold text-gray-200">{label}</span>
                      <span className="text-[10px] text-gray-400">{val}</span>
                    </div>
                  ))}
                </div>
              </div>
            </Tooltip>
          </div>
        </div>

        <div className="flex justify-center">
          <Arrow active={active("input")} color={arrowColor} pulse />
        </div>

        {/* ── Layer 2: Input Processing + Vision ── */}
        <div className="text-[10px] uppercase tracking-widest text-gray-500 text-center mb-2 font-mono">Layer 2 — Input Processing</div>
        <div className="flex justify-center gap-4 mb-1 flex-wrap">
          <div className="flex-1 min-w-48 max-w-xs">
            <Box id="query_builder" icon="🔧" title="Query Builder" subtitle="build_query_text() → NL query string" badge="src/retriever.py" active={active("input")} color={arrowColor} tooltipText={TOOLTIPS.query_builder} />
          </div>
          <div className="flex-1 min-w-48 max-w-xs">
            <Box id="vision" icon="📷" title="Vision Module" subtitle="fridge photo → ingredient list" badge="V4 only" active={active("vision")} color="#22c55e" dashed proposed tooltipText={TOOLTIPS.vision} />
          </div>
        </div>

        <div className="flex justify-center">
          <Arrow active={active("input")} color={arrowColor} pulse />
        </div>

        {/* ── Layer 3: Data Layer ── */}
        <div className="text-[10px] uppercase tracking-widest text-gray-500 text-center mb-2 font-mono">Layer 3 — Data Layer</div>
        <div className="flex justify-center gap-3 mb-1 flex-wrap">
          <div className="flex-1 min-w-40 max-w-xs">
            <Box id="data_usda" icon="🔬" title="USDA FoodData Central" subtitle="Macros per 100g · cached JSON" badge="API" active={active("data")} color={arrowColor} tooltipText={TOOLTIPS.data_usda} />
          </div>
          <div className="flex-1 min-w-40 max-w-xs">
            <Box id="data_recipes" icon="📋" title="Recipe Corpus" subtitle="65 recipes · was 2.2M proposed" badge="JSON" active={active("data")} color={arrowColor} tooltipText={TOOLTIPS.data_recipes} />
          </div>
          <div className="flex-1 min-w-40 max-w-xs">
            <Box id="data_price" icon="🏷️" title="Price Table" subtitle="Hardcoded BLS CPI · was Kroger API" badge="config.py" active={active("data")} color={arrowColor} tooltipText={TOOLTIPS.data_price} />
          </div>
        </div>

        <div className="flex justify-center">
          <Arrow active={active("data")} color={arrowColor} pulse />
        </div>

        {/* ── Embedding + Vector Store ── */}
        <div className="text-[10px] uppercase tracking-widest text-gray-500 text-center mb-2 font-mono">Embedding & Vector Store</div>
        <div className="flex justify-center gap-4 mb-1 flex-wrap">
          <div className="flex-1 min-w-48 max-w-xs">
            <Box id="embed" icon="🧬" title="Sentence Transformer" subtitle="all-MiniLM-L6-v2 · 384-dim · local CPU" badge="22 MB" active={active("data")} color={arrowColor} tooltipText={TOOLTIPS.embed} />
          </div>
          <div className="flex-1 min-w-48 max-w-xs">
            <Box id="chroma" icon="🗄️" title="ChromaDB" subtitle="Cosine similarity · was FAISS proposed" badge="local" active={active("data")} color={arrowColor} tooltipText={TOOLTIPS.chroma} />
          </div>
        </div>

        <div className="flex justify-center">
          <Arrow active={active("data") || active("retrieval")} color={arrowColor} pulse />
        </div>

        {/* ── Layer 4: Retrieval ── */}
        <div className="text-[10px] uppercase tracking-widest text-gray-500 text-center mb-2 font-mono">Layer 4 — Retrieval</div>
        <div className="flex justify-center mb-1">
          <div className="w-full max-w-md">
            <Box id="retrieval" icon="🔍" title="Semantic Search" subtitle="Cosine similarity → top-20 candidates" badge="src/retriever.py" active={active("retrieval")} color={arrowColor} tooltipText={TOOLTIPS.retrieval} />
          </div>
        </div>

        <div className="flex justify-center">
          <Arrow active={active("retrieval") || active("reranker")} color={arrowColor} pulse />
        </div>

        {/* ── Layer 5: Variants ── */}
        <div className="text-[10px] uppercase tracking-widest text-gray-500 text-center mb-2 font-mono">Layer 5 — Experiment Variants</div>
        <div className="flex flex-wrap justify-center gap-3 mb-1">
          {Object.values(VARIANTS).map((vv) => {
            const isSelected = selected === vv.id;
            return (
              <button
                key={vv.id}
                onClick={() => setSelected(vv.id)}
                className={`flex-1 min-w-36 max-w-48 rounded-xl border px-3 py-3 text-left transition-all duration-300 ${isSelected ? "scale-105" : "opacity-40 hover:opacity-70"} ${vv.isProposed ? "border-dashed" : "border-solid"}`}
                style={{
                  borderColor: vv.color,
                  backgroundColor: `${vv.color}18`,
                  boxShadow: isSelected ? `0 0 18px ${vv.color}44` : "none",
                }}
              >
                <div className="font-bold text-sm mb-1" style={{ color: vv.activeColor }}>{vv.id}</div>
                <div className="text-xs text-gray-200 font-medium">{vv.label}</div>
                <div className="text-[10px] text-gray-400 mt-1">{vv.sublabel}</div>
                {vv.isProposed && <div className="text-[10px] text-green-400 mt-1 font-mono">⚠ not yet implemented</div>}
              </button>
            );
          })}
        </div>

        <div className="flex justify-center">
          <Arrow active={active("reranker")} color={arrowColor} pulse />
        </div>

        {/* ── Layer 6: Reranker ── */}
        <div className="text-[10px] uppercase tracking-widest text-gray-500 text-center mb-2 font-mono">Layer 6 — Macro-Aware Reranker (V3 & V4)</div>
        <div className="flex justify-center mb-1">
          <div className="w-full max-w-2xl">
            <Tooltip text={TOOLTIPS.reranker}>
              <div
                className={`rounded-xl border px-5 py-4 transition-all duration-500 ${active("reranker") ? "border-purple-500/70 shadow-lg" : "border-gray-700 opacity-30"}`}
                style={{ backgroundColor: active("reranker") ? "#8b5cf614" : "#111827", boxShadow: active("reranker") ? "0 0 22px #8b5cf622" : "none" }}
              >
                <div className="text-xs font-semibold text-gray-200 mb-3">
                  score = 1.0 − (0.5 × MAPE<sub>macros</sub> + 0.3 × budget_overshoot + 0.2 × waste_fraction)
                </div>
                <div className="flex flex-wrap gap-3">
                  {[
                    { icon: "📊", label: "Macro MAPE", sub: "MAPE across cal/P/C/F · w=0.50", color: "#8b5cf6" },
                    { icon: "💵", label: "Budget Overshoot", sub: "fraction over per-meal limit · w=0.30", color: "#f59e0b" },
                    { icon: "♻️", label: "Ingredient Waste", sub: "missing from pantry · w=0.20", color: "#22c55e" },
                    { icon: "🚫", label: "Dietary Violation", sub: "hard −999 penalty", color: "#ef4444" },
                  ].map(({ icon, label, sub, color }) => (
                    <div key={label} className="flex items-start gap-2 flex-1 min-w-36">
                      <span className="text-base mt-0.5">{icon}</span>
                      <div>
                        <div className="text-xs font-medium" style={{ color }}>{label}</div>
                        <div className="text-[10px] text-gray-400">{sub}</div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-2 text-[10px] text-gray-500 font-mono">src/reranker.py · difflib fuzzy ingredient matching · SearchResult[20] → RankedResult[5]</div>
              </div>
            </Tooltip>
          </div>
        </div>

        <div className="flex justify-center">
          <Arrow active={active("llm")} color={arrowColor} pulse />
        </div>

        {/* ── Layer 7: LLM Generation ── */}
        <div className="text-[10px] uppercase tracking-widest text-gray-500 text-center mb-2 font-mono">Layer 7 — LLM Generation</div>
        <div className="flex justify-center gap-4 mb-1 flex-wrap">
          <div className="flex-1 min-w-48 max-w-sm">
            <Tooltip text={TOOLTIPS.llm}>
              <div
                className={`rounded-xl border px-4 py-3 transition-all duration-500 ${active("llm") ? "border-blue-400/70 shadow-lg" : "border-gray-700 opacity-30"}`}
                style={{ backgroundColor: active("llm") ? "#3b82f614" : "#111827", boxShadow: active("llm") ? "0 0 22px #3b82f622" : "none" }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xl">⚡</span>
                  <div>
                    <div className="text-sm font-semibold text-gray-100">Groq · Llama 3.3 70B</div>
                    <div className="text-[10px] text-blue-400 font-mono">default · free tier · no billing</div>
                  </div>
                </div>
                <div className="text-[10px] text-gray-400">temp=0.4 · max_tokens=1500 · generate_meal_plan_groq()</div>
                <div className="text-[10px] text-yellow-500/70 mt-1 italic">⚠ Original proposal: GPT-4o — switched to Groq for free access</div>
              </div>
            </Tooltip>
          </div>
          <div className="flex-1 min-w-48 max-w-sm">
            <Tooltip text={TOOLTIPS.llm}>
              <div
                className={`rounded-xl border border-dashed px-4 py-3 transition-all duration-500 ${active("llm") ? "border-gray-500/60" : "border-gray-700 opacity-20"}`}
                style={{ backgroundColor: active("llm") ? "#ffffff08" : "#111827" }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xl">🤖</span>
                  <div>
                    <div className="text-sm font-semibold text-gray-300">Gemini 2.0 Flash</div>
                    <div className="text-[10px] text-gray-500 font-mono">alternative · requires API key</div>
                  </div>
                </div>
                <div className="text-[10px] text-gray-500">generate_meal_plan() · generate_baseline_plan()</div>
              </div>
            </Tooltip>
          </div>
        </div>

        <div className="flex justify-center">
          <Arrow active={active("output")} color={arrowColor} pulse />
        </div>

        {/* ── Layer 8: Output ── */}
        <div className="text-[10px] uppercase tracking-widest text-gray-500 text-center mb-2 font-mono">Layer 8 — Output</div>
        <div className="flex justify-center mb-1">
          <div className="w-full max-w-2xl">
            <Tooltip text={TOOLTIPS.output}>
              <div
                className={`rounded-xl border px-5 py-4 transition-all duration-500 ${active("output") ? "border-blue-500/60 shadow-lg" : "border-gray-700 opacity-30"}`}
                style={{ backgroundColor: active("output") ? "#3b82f614" : "#111827", boxShadow: active("output") ? "0 0 22px #3b82f622" : "none" }}
              >
                <div className="flex flex-wrap gap-4 justify-center">
                  {[
                    { icon: "🍽️", label: "Meal Plan", sub: "breakfast · lunch · dinner · snack" },
                    { icon: "📈", label: "Macro Summary", sub: "target vs. actual" },
                    { icon: "🛒", label: "Shopping List", sub: "missing ingredients" },
                    { icon: "⚖️", label: "Trade-off Notes", sub: "why each recipe was chosen" },
                  ].map(({ icon, label, sub }) => (
                    <div key={label} className="flex flex-col items-center text-center min-w-24">
                      <span className="text-2xl mb-1">{icon}</span>
                      <div className="text-xs font-semibold text-gray-200">{label}</div>
                      <div className="text-[10px] text-gray-400">{sub}</div>
                    </div>
                  ))}
                </div>
              </div>
            </Tooltip>
          </div>
        </div>

        {/* ── Evaluation Footer ── */}
        <div className="mt-6">
          <div className="text-[10px] uppercase tracking-widest text-gray-500 text-center mb-2 font-mono">Evaluation Layer — src/evaluator.py · notebooks/divya_evaluation.ipynb</div>
          <Tooltip text={TOOLTIPS.eval}>
            <div className="rounded-xl border border-gray-700/60 px-5 py-3 bg-gray-900/50">
              <div className="flex flex-wrap gap-2 justify-center">
                {[
                  { icon: "📊", label: "Macro MAPE", sub: "per-dimension %" },
                  { icon: "🎯", label: "Precision@K", sub: "vs. ground truth" },
                  { icon: "💰", label: "Budget Adherence", sub: "within_budget flag" },
                  { icon: "♻️", label: "Waste Fraction", sub: "avg missing ingredients" },
                  { icon: "📝", label: "ROUGE-L", sub: "plan vs. reference" },
                  { icon: "⚡", label: "Latency", sub: "end-to-end seconds" },
                ].map(({ icon, label, sub }) => (
                  <div key={label} className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-gray-800/80 border border-gray-700/50">
                    <span className="text-sm">{icon}</span>
                    <div>
                      <div className="text-[11px] font-medium text-gray-300">{label}</div>
                      <div className="text-[9px] text-gray-500">{sub}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Tooltip>
        </div>

        {/* ── Legend ── */}
        <div className="mt-6 flex flex-wrap justify-center gap-4 text-[11px] text-gray-400">
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-blue-500/60 border border-blue-500" /> Core pipeline</div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-purple-500/60 border border-purple-500" /> Reranker (V3+)</div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-green-500/60 border border-green-500 border-dashed" /> Vision module (proposed)</div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-gray-700/60 border border-gray-600 border-dashed" /> Inactive / dimmed</div>
          <div className="flex items-center gap-1.5"><span className="text-yellow-400">⚠</span> Differs from original proposal</div>
          <div className="flex items-center gap-1.5"><span>💬</span> Hover any box for details</div>
        </div>
      </div>
    </div>
  );
}
