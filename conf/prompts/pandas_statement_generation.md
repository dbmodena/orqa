## PandasCodeGeneration
Generate **exactly ONE** Python Pandas query that implements the single business question given to you below — not a new question you invent, and not a set of three. This call is one of several independent generation calls in the same run, each one bound to its own query plan; you are only ever answering the one plan attached to THIS call.

### Plan-Driven Generation
Below this prompt, you will see:
- **Inputs** — detected languages, DataFrame aliases, table schemas, and the verified table relationships
- **Column Statistics** — per-table cardinality, nullness, and value distributions to inform your code choices
- **Structured Plan** — the ordered decomposition steps for THIS query, including its `question` (the business question you must answer — do not replace it with a different one)

- Generate plain Pandas code that accomplishes the analytical goal directly (no external APIs or ML frameworks).
- Always produce executable Python code that runs on the provided DataFrames.

### Rename-first rule (NON-NEGOTIABLE)
When the **Structured Plan** (below) lists a Step 0 with `.rename()` calls, you MUST emit them at the top of your query — before any merge, filter, or aggregation. Use the renamed column names everywhere in the rest of the code. Never reference the original column names after renaming.

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
- Column access (mandatory): always use bracket/getitem indexing — `df["column name"]` — never dot/attribute access (`df.column_name`) and never a bare column name inside `.query()`/`.eval()`. Some columns contain spaces, accented/non-ASCII characters, punctuation, or are purely numeric text (e.g. `"2023"`), all of which break attribute access or eval-string parsing.

### Data quality — tables are RAW
DataFrames are loaded exactly as stored: no bad-token→NaN conversion, no numeric coercion, no null-row dropping has been applied upstream. If the Structured Plan includes a `clean` step, implement its `params.actions` literally using the EXACT literal tokens shown in Column Statistics' `bad_token_counts` (never invented ones): `.replace([...], pd.NA)` to blank sentinel tokens, `pd.to_numeric(col, errors="coerce")` — never a raw `.astype(float)`, which raises on any stray token — to cast, `.dropna(subset=[...])` to drop rows, `.fillna(...)` to impute. If a column you use shows a non-trivial `bad_token_counts` or a `numeric_parseable_ratio` near 1.0 but no `clean` step covers it, still handle it defensively (`pd.to_numeric(..., errors="coerce")`) rather than assuming the data is already clean. If an action carries `"treat_as_missing"`, `.replace([...], pd.NA)` those exact values first — they're planner-discovered sentinels not already in `bad_token_counts`.

If the Structured Plan includes a `derive` step whose `params.actions` use `"technique": "flag"` or `"bucket"` (preserving an outlier/censoring pattern as its own feature instead of discarding it): implement `flag` as a boolean column via `.str.contains(...)`/`.isin([...])`/a regex matching the pattern named in the action's `rule`; implement `bucket` as a categorical column via `np.select([...], [...], default=...)` or `pd.cut(...)` for numeric ranges, mapping the ranges/patterns named in `rule` to their labels — using the exact values/patterns from `numeric_outliers`/`minority_value_groups`, never invented ones.

### Correlate / limit / rank steps
If the Structured Plan includes a `correlate` step: compute `df[columns].corr(method=params.get("method", "pearson"))` for the whole-table case, or `df.groupby(params["group_by"])[columns].corr(method=...)` when `params.group_by` is set; extract the pairwise coefficient (for exactly 2 columns, index into the resulting matrix, e.g. `.iloc[0, 1]`, rather than returning the whole symmetric matrix), and name the result `params.output_column` when that key is present.

If the Structured Plan includes a `limit` step: `params.how` (default `"head"`) selects the idiom — `"head"` -> `.head(params["n"])` (a row cap in whatever order the plan already produced, typically right after a `sort` step); `"largest"`/`"smallest"` -> `.nlargest(params["n"], params["by"])` / `.nsmallest(params["n"], params["by"])` for a self-contained top/bottom-N with no separate `sort` step.

If the Structured Plan includes a `rank` step: `.rank(method=params.get("method", "average"), ascending=params.get("ascending", True))` on the column(s) in `params["by"]`, assigned to the new column named `params["output_column"]`; when `params.group_by` is set, compute it per group with `.groupby(params["group_by"])[params["by"]].rank(...)` instead of over the whole DataFrame.

### Table usage
A DataFrame is justified only if its columns appear in output, filters, or aggregations — not merely in a merge key. Using a DataFrame solely to restrict rows via a merge is **not** justified unless the question explicitly requires cross-table validation (e.g. "find records appearing in both sources"). Touch only the minimal columns actually needed — this was already decided (and judged) during planning, so don't reach for columns beyond what the plan's steps call for.

### Output (conform to the Pydantic schema — `queries` must contain exactly ONE item)
- `question`: the plan's business question (faithfully preserved, see above)
- `query`: Python/Pandas code
- `motivation`: 2–3 sentences in business language explaining (1) the analytical value, (2) what specific columns each DataFrame uniquely contributes, (3) why this merge/join strategy is correct.

Note: `difficulty`, `topic`, `story`, `translated_question`, `detected_language`,
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
