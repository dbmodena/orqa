## SingleTablePandasCodeGeneration
Generate **exactly ONE** Python Pandas query that implements the single business question given to you below — not a new question you invent, and not a set of three. This call is one of several independent generation calls in the same run, each one bound to its own query plan; you are only ever answering the one plan attached to THIS call.

### Task Types & Plan-Driven Generation
Below this prompt, you will see:
- **Inputs** — the detected languages and the table schema
- **Column Statistics** — per-table cardinality, nullness, and value distributions to inform your code choices
- **Available Skill** (if present) — when this plan's `task_types` includes an ML/predictive task (e.g. `classification`), the matching skill (e.g. TabPFN) is injected with its documented usage pattern
- **Structured Plan** — the ordered decomposition steps for THIS query, including its `question` (the business question you must answer — do not replace it with a different one) and `task_types` that specify the analytical nature of the operation (e.g. `aggregation`, `classification`, `timeseries`, `window`, etc.)

### Question rule — use the plan's question, don't invent one
- The `question` field of the Structured Plan below is the exact business question this query must answer. Copy its intent faithfully into your output's `question`; you may only lightly polish phrasing (grammar, fluency), never its topic, scope, or the metrics/filters it implies.
- The question reads as if asked by an average, non-expert user who does not know table or column names, has a general understanding of the business domain, and would recognize it phrased naturally and conversationally — the plan's question is already written this way, so preserve that framing.
- The plan's question deliberately carries the distinctive subject vocabulary (topic, entity type, agency, place, period) that a downstream keyword extractor uses to retrieve the right table from a reverse index. Any polish you apply must keep those identifying terms intact — never paraphrase them away into generic wording.
✗ Never mention table names, column names, DataFrame names, or any Pandas/Python operations in the question.

### Skill Usage & Plain Pandas Fallback (NON-NEGOTIABLE when a skill is present)
- **If this plan's `task_types` includes an ML/predictive op (e.g. `classification`, `regression`, `timeseries`, `causal`) AND an "### Available Skill" section appears below**, you MUST follow that skill's documented pattern exactly — do not substitute a plain-Pandas approximation instead of the skill just because it would be simpler.
- **Only when no skill is available, or this plan's `task_types` contains no matching ML op**, generate plain Pandas code that accomplishes the analytical goal directly (no external APIs or ML frameworks).
- Always produce executable Python code that runs on the provided DataFrame.
- A skill's example code may use illustrative/placeholder column names (e.g. for
  `pd.get_dummies(df, columns=[...])`). NEVER copy those literal names into your code —
  substitute the REAL column names from the **Table**/**Column Statistics** sections. A column
  that doesn't exist in the table fails at execution with a `KeyError`.

### Query rules
- The DataFrame is **pre-loaded** with its designated alias — start operations directly on it.
- Never use `pd.DataFrame()`, `pd.read_csv()`, or reassign the DataFrame variable.
- Prefer method chaining. Correct Python/Pandas only.
- Follow the Structured Plan's steps, in order — they are the validated decomposition of this exact question.

### Difficulty
Assign whichever of `easy` / `medium` / `hard` genuinely reflects the complexity of THIS plan's steps — don't force it up or down to hit a particular tier:

| Level | Typically reflects |
|---|---|
| Easy | Single DataFrame filtering, column selection, simple `sort_values`/`head`, ≤1 aggregation |
| Medium | `groupby` with multiple aggregations, compound boolean filters, or reshaping (`pivot`/`melt`) |
| Hard | Multi-step pipelines with groupby, window-style ops (`rolling`/`expanding`/`rank`), complex `apply`/custom functions, hierarchical indexes, advanced reshaping combined, or an ML/skill-driven operation (classification, prediction, etc.) |

A plan whose `task_types` includes an ML/predictive op is typically `hard`, since it involves feature engineering or a skill call (e.g. TabPFN) when one is available.

### Output (conform to the Pydantic schema — `queries` must contain exactly ONE item)
- `difficulty`: easy / medium / hard
- `question`: the plan's business question (faithfully preserved, see above)
- `query`: Python/Pandas code
- `motivation`: 2–3 sentences in business language explaining (1) the analytical value, (2) which specific columns are used and why they matter.

Note: `topic`, `story`, `translated_question`, `detected_language`,
`question_keywords`, `translated_question_keywords`, and `tables` are NOT
part of this output — the query plan already decided them (table usage,
justification, and columns were already fixed and judged during planning)
and they are attached to your query automatically. Do not generate them.

Notes:
- Return a `queries` list containing exactly one query object — not three.

### Inputs
- **Detected languages** - The following datasets can have the following langauges `{languages}`
- **Table** — schema, columns, types, metadata: `{table}`
