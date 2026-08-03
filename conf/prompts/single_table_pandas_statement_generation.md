## SingleTablePandasCodeGeneration
Generate **exactly ONE** Python Pandas query that implements the single business question given to you below — not a new question you invent, and not a set of three. This call is one of several independent generation calls in the same run, each one bound to its own query plan; you are only ever answering the one plan attached to THIS call.

### Plan-Driven Generation
Below this prompt, you will see:
- **Inputs** — the detected languages and the table schema
- **Column Statistics** — per-table cardinality, nullness, and value distributions to inform your code choices
- **Structured Plan** — the ordered decomposition steps for THIS query, including its `question` (the business question you must answer — do not replace it with a different one)

### Question rule — use the plan's question, don't invent one
- The `question` field of the Structured Plan below is the exact business question this query must answer. Copy its intent faithfully into your output's `question`; you may only lightly polish phrasing (grammar, fluency), never its topic, scope, or the metrics/filters it implies.
- The question reads as if asked by an average, non-expert user who does not know table or column names, has a general understanding of the business domain, and would recognize it phrased naturally and conversationally — the plan's question is already written this way, so preserve that framing.
- The plan's question deliberately carries the distinctive subject vocabulary (topic, entity type, agency, place, period) that a downstream keyword extractor uses to retrieve the right table from a reverse index. Any polish you apply must keep those identifying terms intact — never paraphrase them away into generic wording.
✗ Never mention table names, column names, DataFrame names, or any Pandas/Python operations in the question.

- Generate plain Pandas code that accomplishes the analytical goal directly (no external APIs or ML frameworks).
- Always produce executable Python code that runs on the provided DataFrame.

### Query rules
- The DataFrame is **pre-loaded** with its designated alias — start operations directly on it.
- Never use `pd.DataFrame()`, `pd.read_csv()`, or reassign the DataFrame variable.
- Prefer method chaining. Correct Python/Pandas only.
- Follow the Structured Plan's steps, in order — they are the validated decomposition of this exact question.

### Data quality — the table is RAW
The DataFrame is loaded exactly as stored: no bad-token→NaN conversion, no numeric coercion, no null-row dropping has been applied upstream. If the Structured Plan includes a `clean` step, implement its `params.actions` literally using the EXACT literal tokens shown in Column Statistics' `bad_token_counts` (never invented ones): `.replace([...], pd.NA)` to blank sentinel tokens, `pd.to_numeric(col, errors="coerce")` — never a raw `.astype(float)`, which raises on any stray token — to cast, `.dropna(subset=[...])` to drop rows, `.fillna(...)` to impute. If a column you use shows a non-trivial `bad_token_counts` or a `numeric_parseable_ratio` near 1.0 but no `clean` step covers it, still handle it defensively (`pd.to_numeric(..., errors="coerce")`) rather than assuming the data is already clean. If an action carries `"treat_as_missing"`, `.replace([...], pd.NA)` those exact values first — they're planner-discovered sentinels not already in `bad_token_counts`.

If the Structured Plan includes a `derive` step whose `params.actions` use `"technique": "flag"` or `"bucket"` (preserving an outlier/censoring pattern as its own feature instead of discarding it): implement `flag` as a boolean column via `.str.contains(...)`/`.isin([...])`/a regex matching the pattern named in the action's `rule`; implement `bucket` as a categorical column via `np.select([...], [...], default=...)` or `pd.cut(...)` for numeric ranges, mapping the ranges/patterns named in `rule` to their labels — using the exact values/patterns from `numeric_outliers`/`minority_value_groups`, never invented ones.

### Correlate / limit / rank steps
If the Structured Plan includes a `correlate` step: compute `df[columns].corr(method=params.get("method", "pearson"))` for the whole-table case, or `df.groupby(params["group_by"])[columns].corr(method=...)` when `params.group_by` is set; extract the pairwise coefficient (for exactly 2 columns, index into the resulting matrix, e.g. `.iloc[0, 1]`, rather than returning the whole symmetric matrix), and name the result `params.output_column` when that key is present.

If the Structured Plan includes a `limit` step: `params.how` (default `"head"`) selects the idiom — `"head"` -> `.head(params["n"])` (a row cap in whatever order the plan already produced, typically right after a `sort` step); `"largest"`/`"smallest"` -> `.nlargest(params["n"], params["by"])` / `.nsmallest(params["n"], params["by"])` for a self-contained top/bottom-N with no separate `sort` step.

If the Structured Plan includes a `rank` step: `.rank(method=params.get("method", "average"), ascending=params.get("ascending", True))` on the column(s) in `params["by"]`, assigned to the new column named `params["output_column"]`; when `params.group_by` is set, compute it per group with `.groupby(params["group_by"])[params["by"]].rank(...)` instead of over the whole DataFrame.

### Output (conform to the Pydantic schema — `queries` must contain exactly ONE item)
- `question`: the plan's business question (faithfully preserved, see above)
- `query`: Python/Pandas code
- `motivation`: 2–3 sentences in business language explaining (1) the analytical value, (2) which specific columns are used and why they matter.

Note: `difficulty`, `topic`, `story`, `translated_question`, `detected_language`,
`question_keywords`, `translated_question_keywords`, and `tables` are NOT
part of this output — the query plan already decided them (table usage,
justification, and columns were already fixed and judged during planning)
and they are attached to your query automatically. Do not generate them.

Notes:
- Return a `queries` list containing exactly one query object — not three.

### Inputs
- **Detected languages** - The following datasets can have the following langauges `{languages}`
- **Table** — schema, columns, types, metadata: `{table}`
