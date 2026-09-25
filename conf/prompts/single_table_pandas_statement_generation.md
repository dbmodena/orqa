## SingleTablePandasCodeGeneration
Generate **exactly ONE** Python Pandas query implementing the single business question below — not a new question you invent. Return a `queries` list containing exactly one query object. This call is one of several independent generation calls, each bound to its own plan; you answer only the plan attached to THIS call.

Below this prompt you will see the **Inputs** (detected languages, table schema), the **Column Statistics**, and the **Structured Plan** — the ordered decomposition steps for THIS query, including the `question` you must answer.

### Question rule — use the plan's question, don't invent one
- The plan's `question` is the exact business question this query must answer. Copy its intent faithfully; you may only lightly polish phrasing (grammar, fluency), never its topic, scope, or the metrics/filters it implies.
- It is already written as an average, non-expert user would ask it — conversational, no knowledge of table or column names. Preserve that framing.
- It deliberately carries the distinctive subject vocabulary (topic, entity type, agency, place, period) a downstream keyword extractor uses to retrieve the right table from a reverse index. Any polish must keep those identifying terms intact — never paraphrase them into generic wording.
- ✗ Never mention table names, column names, DataFrame names, or any Pandas/Python operation in the question.

### Query rules
- The DataFrame is **pre-loaded** under its alias — start operations directly on it. Never use `pd.DataFrame()`, `pd.read_csv()`, or reassign the alias variable.
- Follow the Structured Plan's steps, in order — they are the validated decomposition of this exact question. Generate plain Pandas only (no external APIs or ML frameworks), always executable against the provided DataFrame.
- Prefer method chaining. Correct Python/Pandas only.
- **Column access (mandatory):** always use bracket indexing — `df["column name"]` — never dot access (`df.column_name`) and never a bare column name inside `.query()`/`.eval()`. Columns may contain spaces, accented/non-ASCII characters, punctuation, or be purely numeric text (`"2023"`), all of which break attribute access and eval-string parsing.

### Data quality — the table is RAW
The DataFrame is loaded exactly as stored: no bad-token→NaN conversion, no numeric coercion, no null-row dropping.

If the plan has a `clean` step, implement its `params.actions` literally, using the EXACT literal tokens from Column Statistics' `bad_token_counts` (never invented ones):
- `.replace([...], pd.NA)` to blank sentinel tokens
- `pd.to_numeric(col, errors="coerce")` to cast — never a raw `.astype(float)`, which raises on any stray token
- `.dropna(subset=[...])` to drop rows; `.fillna(...)` to impute
- if an action carries `"treat_as_missing"`, `.replace([...], pd.NA)` those exact values FIRST — planner-discovered sentinels not yet in `bad_token_counts`

If a column you use shows a non-trivial `bad_token_counts` or a `numeric_parseable_ratio` near 1.0 but no `clean` step covers it, still handle it defensively (`pd.to_numeric(..., errors="coerce")`) rather than assuming clean data.

If the plan has a `derive` step whose `params.actions` use `"technique": "flag"` or `"bucket"` (preserving an outlier/censoring pattern as a feature instead of discarding it): implement `flag` as a boolean column via `.str.contains(...)`/`.isin([...])`/a regex matching the action's `rule`; implement `bucket` as a categorical column via `np.select([...], [...], default=...)` or `pd.cut(...)` for numeric ranges, mapping the ranges/patterns in `rule` to their labels — using the exact values from `numeric_outliers`/`minority_value_groups`, never invented ones.

### Correlate / limit / rank steps
- `correlate`: `df[columns].corr(method=params.get("method", "pearson"))`, or `df.groupby(params["group_by"])[columns].corr(method=...)` when `params.group_by` is set. For exactly 2 columns, extract the pairwise coefficient (e.g. `.iloc[0, 1]`) rather than returning the whole symmetric matrix; name it `params.output_column` when present.
- `limit`: `params.how` (default `"head"`) selects the idiom — `"head"` → `.head(params["n"])` in whatever order the plan already produced (typically right after a `sort`); `"largest"`/`"smallest"` → `.nlargest(params["n"], params["by"])` / `.nsmallest(...)` for a self-contained top/bottom-N with no separate `sort`.
- `rank`: `.rank(method=params.get("method", "average"), ascending=params.get("ascending", True))` on `params["by"]`, assigned to `params["output_column"]`; with `params.group_by` set, compute per group via `.groupby(params["group_by"])[params["by"]].rank(...)`.

### Output (conform to the Pydantic schema — `queries` holds exactly ONE item)
- `question`: the plan's business question, faithfully preserved (see above)
- `query`: Python/Pandas code
- `motivation`: 2–3 sentences in business language covering (1) the analytical value, (2) which specific columns are used and why they matter.

Do NOT generate `difficulty`, `topic`, `story`, `translated_question`, `detected_language`, `question_keywords`, `translated_question_keywords`, or `tables` — the plan already decided them and they are attached to your query automatically.

### Inputs
- **Detected languages** — these datasets may use: `{languages}`
- **Table** — schema, columns, types, metadata: `{table}`
