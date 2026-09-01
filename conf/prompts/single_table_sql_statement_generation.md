## SingleTableSQLGeneration
Generate **exactly ONE** DuckDB SQL query implementing the single business question below — not a new question you invent. Return a `queries` list containing exactly one query object. This call is one of several independent generation calls, each bound to its own plan; you answer only the plan attached to THIS call.

Below this prompt you will see the **Inputs** (detected languages, alias, table schema), the **Column Statistics**, and the **Structured Plan** — the ordered decomposition steps for THIS query, including the `question` you must answer.

### Question rule — use the plan's question, don't invent one
- The plan's `question` is the exact business question this query must answer. Copy its intent faithfully; you may only lightly polish phrasing (grammar, fluency), never its topic, scope, or the metrics/filters it implies.
- It is already written as an average, non-expert user would ask it — conversational, no knowledge of table or column names. Preserve that framing.
- It deliberately carries the distinctive subject vocabulary (topic, entity type, agency, place, period) a downstream keyword extractor uses to retrieve the right table from a reverse index. Any polish must keep those identifying terms intact — never paraphrase them into generic wording.
- ✗ Never mention table names, column names, or any SQL operation in the question.

### Query rules
- Follow the Structured Plan's steps, in order — they are the validated decomposition of this exact question.
- DuckDB/ANSI SQL syntax only, over the single table under its given alias.
- **Column-name quoting (mandatory):** ALWAYS wrap every column reference in double quotes — `"column name"` — even for names that look like ordinary identifiers. Columns may contain spaces, accented/non-ASCII characters, punctuation, or be purely numeric text (`"2023"`). Left unquoted, a purely-numeric name is silently parsed as a numeric LITERAL instead of a column reference — a wrong result with no error — and the others break the query outright. If a name itself contains a `"`, double it: `col "a" b` → `"col ""a"" b"`.

### Data quality — the table is RAW
The table is loaded exactly as stored: no bad-token→NULL conversion, no numeric coercion, no null-row dropping.

If the plan has a `clean` step, implement its `params.actions` literally, using the EXACT literal tokens from Column Statistics' `bad_token_counts` (never invented ones):
- `CASE WHEN col IN ('n/a', 'not available') THEN NULL ELSE col END` to blank a sentinel
- `TRY_CAST(col AS DOUBLE)` to cast — never a raw `CAST`, which errors on any stray token
- `WHERE col IS NOT NULL` to drop rows; `COALESCE(col, <value>)` to impute
- if an action carries `"treat_as_missing"`, fold those exact values into the same `CASE`/`IN (...)` blanking first — planner-discovered sentinels not yet in `bad_token_counts`

If a column your query touches shows a non-trivial `bad_token_counts` or a `numeric_parseable_ratio` near 1.0 but no `clean` step covers it, still use `TRY_CAST` defensively rather than letting a stray token crash the query.

If the plan has a `derive` step whose `params.actions` use `"technique": "flag"` or `"bucket"` (preserving an outlier/censoring pattern as a feature instead of discarding it): implement `flag` as `CASE WHEN col ~ '<pattern>' OR col IN (...) THEN TRUE ELSE FALSE END AS output_column`, and `bucket` as `CASE WHEN ... THEN 'label' ... ELSE ... END AS output_column`, mapping the ranges/patterns in the action's `rule` to their labels — using the exact values from `numeric_outliers`/`minority_value_groups`, never invented ones.

### Correlate / limit / rank steps
- `correlate`: DuckDB's `corr(col_a, col_b)` aggregate — Pearson only, so `params.method` is always `"pearson"` for a SQL plan (never generate Spearman/Kendall). Add `GROUP BY <params.group_by>` when set, for one coefficient per group; alias `AS <params.output_column>` when present.
- `limit`: `params.how` (default `"head"`) selects the idiom — `"head"` → a bare `LIMIT params.n` on top of the existing row order (typically right after the `ORDER BY` implementing a `sort` step); `"largest"`/`"smallest"` → `ORDER BY <params.by> DESC`/`ASC LIMIT params.n` for a self-contained top/bottom-N with no separate `sort`. (The validator overrides any trailing `LIMIT` with its own `LIMIT 100` during test execution only — still emit the plan's real `LIMIT params.n`.)
- `rank`: the window function matching `params.method` — `"min"` → `RANK()`, `"dense"` → `DENSE_RANK()`, `"first"` → `ROW_NUMBER()` (SQL plans are restricted to these three; never `"average"`/`"max"` tie-handling) — `OVER ([PARTITION BY <params.group_by>] ORDER BY <params.by> [DESC])`, aliased `AS <params.output_column>`. Omit `PARTITION BY` when `params.group_by` is absent; add `DESC` when `params.ascending` is `false`.

### Output (conform to the Pydantic schema — `queries` holds exactly ONE item)
- `question`: the plan's business question, faithfully preserved (see above)
- `query`: DuckDB SQL
- `motivation`: 2–3 sentences in business language covering (1) the analytical value, (2) which specific columns are used and why they matter.

Do NOT generate `difficulty`, `topic`, `story`, `translated_question`, `detected_language`, `question_keywords`, `translated_question_keywords`, or `tables` — the plan already decided them and they are attached to your query automatically.

### Inputs
- **Detected languages** — these datasets may use: `{languages}`
- **Alias** — use only this in queries: `{alias}`
- **Table** — schema, columns, types, metadata: `{table}`
