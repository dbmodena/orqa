## PandasCodeGeneration
Generate **exactly ONE** Python Pandas query that implements the single business question given to you below — not a new question you invent, and not a set of three. This call is one of several independent generation calls in the same run, each one bound to its own query plan; you are only ever answering the one plan attached to THIS call.

### Task Types & Plan-Driven Generation
Below this prompt, you will see:
- **Inputs** — detected languages, DataFrame aliases, table schemas, and the verified table relationships
- **Column Statistics** — per-table cardinality, nullness, and value distributions to inform your code choices
- **Available Skill** (if present) — when this plan's `task_types` includes an ML/predictive task (e.g. `classification`), the matching skill (e.g. TabPFN) is injected with its documented usage pattern
- **Structured Plan** — the ordered decomposition steps for THIS query, including its `question` (the business question you must answer — do not replace it with a different one) and `task_types` that specify the analytical nature of the operation (e.g. `aggregation`, `classification`, `timeseries`, `join`, etc.)

### Skill Usage & Plain Pandas Fallback (NON-NEGOTIABLE when a skill is present)
- **If this plan's `task_types` includes an ML/predictive op (e.g. `classification`, `regression`, `timeseries`, `causal`) AND an "### Available Skill" section appears below**, you MUST follow that skill's documented pattern exactly — do not substitute a plain-Pandas approximation instead of the skill just because it would be simpler.
- **Only when no skill is available, or this plan's `task_types` contains no matching ML op**, generate plain Pandas code that accomplishes the analytical goal directly (no external APIs or ML frameworks).
- Always produce executable Python code that runs on the provided DataFrames.
- A skill's example code may use illustrative/placeholder column names (e.g. for
  `pd.get_dummies(df, columns=[...])`). NEVER copy those literal names into your code —
  substitute the REAL column names from the **Tables**/**Column Statistics** sections. A column
  that doesn't exist in the table fails at execution with a `KeyError`.

### Rename-first rule (NON-NEGOTIABLE)
When the **Structured Plan** (below) lists a Step 0 with `.rename()` calls, you MUST emit them at the top of your query — before any merge, filter, aggregation, or skill-based operation. Use the renamed column names everywhere in the rest of the code. Never reference the original column names after renaming.

### Question rule — use the plan's question, don't invent one
- The `question` field of the Structured Plan below is the exact business question this query must answer. Copy its intent faithfully into your output's `question`; you may only lightly polish phrasing (grammar, fluency), never its topic, scope, or the metrics/filters it implies.
- The question reads as if asked by an average, non-expert user who does not know table or column names, has a general understanding of the business domain, and would recognize it phrased naturally and conversationally — the plan's question is already written this way, so preserve that framing.
- Use table keywords from the table analysis when they help the question stay non-technical and recognizable, without adding technical detail not present in the plan.
- The plan's question deliberately carries the distinctive subject vocabulary (topic, entity type, agency, place, period) that a downstream keyword extractor uses to retrieve the right table from a reverse index. Any polish you apply must keep those identifying terms intact — never paraphrase them away into generic wording.
✗ Never mention table names, column names, DataFrame names, or any Pandas/Python operations in the question.

### Query rules
- All DataFrames are **pre-loaded** with their designated aliases — start operations directly on them.
- For every relationship you use from **Verified table relationships**, its join type (INNER, LEFT, etc.), key columns, and case-normalisation requirement are NON-NEGOTIABLE — apply them exactly as listed.
- The relationships are building blocks, not a prescribed chain: compose them in whatever shape the Structured Plan dictates — a sequential chain of merges, independent merges whose results are then compared/combined, or a mix. Never merge DataFrames that have no listed relationship.
- When a join is marked `Case-insensitive keys: YES`, you MUST apply `.str.lower()` on both sides before joining, using the pattern shown. Omitting this is a hard error.
- Never use `pd.DataFrame()`, `pd.read_csv()`, or reassign the original DataFrame variables (renaming into a new variable is fine and required when Step 0 is present).
- All DataFrames used must be genuinely necessary (see Table Usage below).
- Only use explicitly defined match relationships — no inferred merges.
- Prefer method chaining. Correct Python/Pandas only.
- Follow the Structured Plan's steps, in order — they are the validated decomposition of this exact question.

### Table usage
A DataFrame is justified only if its columns appear in output, filters, or aggregations — not merely in a merge key. Using a DataFrame solely to restrict rows via a merge is **not** justified unless the question explicitly requires cross-table validation (e.g. "find records appearing in both sources"). Touch only the minimal columns actually needed — this was already decided (and judged) during planning, so don't reach for columns beyond what the plan's steps call for.

### Difficulty
Assign whichever of `easy` / `medium` / `hard` genuinely reflects the complexity of THIS plan's steps — don't force it up or down to hit a particular tier:

| Level | Typically reflects |
|---|---|
| Easy | Single DataFrame filtering, column selection, simple `sort_values`/`head`, ≤1 aggregation |
| Medium | `merge`/`join` of multiple DataFrames, `groupby` with multiple aggregations, compound boolean filters, or reshaping (`pivot`/`melt`) |
| Hard | Multi-step pipelines with several merges + groupby, window-style ops (`rolling`/`expanding`/`rank`), complex `apply`/custom functions, hierarchical indexes, advanced reshaping combined, or an ML/skill-driven operation (classification, prediction, etc.) |

A plan whose `task_types` includes an ML/predictive op is typically `hard`, since it involves feature engineering or a skill call (e.g. TabPFN) when one is available.

### Output (conform to the Pydantic schema — `queries` must contain exactly ONE item)
- `difficulty`: easy / medium / hard
- `question`: the plan's business question (faithfully preserved, see above)
- `query`: Python/Pandas code
- `motivation`: 2–3 sentences in business language explaining (1) the analytical value, (2) what specific columns each DataFrame uniquely contributes, (3) why this merge/join strategy is correct.

Note: `topic`, `story`, `translated_question`, `detected_language`,
`question_keywords`, `translated_question_keywords`, and `tables` are NOT
part of this output — the query plan already decided them (table usage,
justification, and columns were already fixed and judged during planning)
and they are attached to your query automatically. Do not generate them.

Notes:
- Return a `queries` list containing exactly one query object — not three.

### Inputs
- **Detected languages** - The following datasets can have the following langauges `{languages}`
- **Aliases** — use only these DataFrame variable names in your code: `{aliases}`
- **Tables** — schema, columns, types, metadata: `{table}`
- **Verified table relationships** — the ONLY ways DataFrames may be combined; use each relationship's keys and settings exactly as listed, composed in the shape the Structured Plan dictates:

```
{matches}
```
