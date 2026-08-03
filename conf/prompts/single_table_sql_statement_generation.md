## SingleTableSQLGeneration
Generate **exactly ONE** DuckDB SQL query that implements the single business question given to you below — not a new question you invent, and not a set of three. This call is one of several independent generation calls in the same run, each one bound to its own query plan; you are only ever answering the one plan attached to THIS call.

Below this prompt you will see the **Inputs** (detected languages, alias, table schema), the **Column Statistics** for this table, and the **Structured Plan** — the ordered decomposition steps for THIS query, including its `question` (the business question you must answer — do not replace it with a different one).

### Question rule — use the plan's question, don't invent one
- The `question` field of the Structured Plan below is the exact business question this query must answer. Copy its intent faithfully into your output's `question`; you may only lightly polish phrasing (grammar, fluency), never its topic, scope, or the metrics/filters it implies.
- The question reads as if asked by an average, non-expert user who does not know table or column names, has a general understanding of the business domain, and would recognize it phrased naturally and conversationally — the plan’s question is already written this way, so preserve that framing.
- The plan's question deliberately carries the distinctive subject vocabulary (topic, entity type, agency, place, period) that a downstream keyword extractor uses to retrieve the right table from a reverse index. Any polish you apply must keep those identifying terms intact — never paraphrase them away into generic wording.
✗ Never mention table names, column names, or any query/SQL operations in the question.

### Query rules
- Only reference the single provided alias — never reference other tables.
- DuckDB/ANSI SQL syntax only. Correct and executable code only.
- Follow the Structured Plan's steps, in order — they are the validated decomposition of this exact question.

### Data quality — the table is RAW
The table is loaded exactly as stored: no bad-token→NULL conversion, no numeric coercion, no null-row dropping has been applied upstream. If the Structured Plan includes a `clean` step, implement its `params.actions` literally in SQL using the EXACT literal tokens shown in Column Statistics' `bad_token_counts` (never invented ones): `CASE WHEN col IN ('n/a', 'not available') THEN NULL ELSE col END` to blank a sentinel, `TRY_CAST(col AS DOUBLE)` — never a raw `CAST`, which errors on any stray token — to cast, `WHERE col IS NOT NULL` to drop rows, `COALESCE(col, <value>)` to impute a constant. If a column your query touches shows a non-trivial `bad_token_counts` or a `numeric_parseable_ratio` near 1.0 but no `clean` step covers it, still handle it defensively (`TRY_CAST` instead of `CAST`) rather than letting a stray token crash the query. If an action carries `"treat_as_missing"`, fold those exact values into the same `CASE`/`IN (...)` blanking before whatever the action does next — they're planner-discovered sentinels not already in `bad_token_counts`.

If the Structured Plan includes a `derive` step whose `params.actions` use `"technique": "flag"` or `"bucket"` (preserving an outlier/censoring pattern as its own feature instead of discarding it): implement `flag` as `CASE WHEN col ~ '<pattern>' OR col IN (...) THEN TRUE ELSE FALSE END AS output_column`, and `bucket` as a `CASE WHEN ... THEN 'label' WHEN ... THEN 'label' ELSE ... END AS output_column` mapping the ranges/patterns named in the action's `rule` to their labels — using the exact values/patterns from `numeric_outliers`/`minority_value_groups`, never invented ones.

### Correlate / limit / rank steps
If the Structured Plan includes a `correlate` step: use DuckDB's `corr(col_a, col_b)` aggregate — Pearson only, so `params.method` will always be `"pearson"` for a SQL plan (never generate Spearman/Kendall SQL). Add `GROUP BY <params.group_by columns>` when `params.group_by` is set, for one coefficient per group instead of one over the whole table; alias the result `AS <params.output_column>` when that key is present.

If the Structured Plan includes a `limit` step: `params.how` (default `"head"`) selects the idiom — `"head"` -> a bare `LIMIT params.n` applied on top of the query's existing row order (typically right after an `ORDER BY` implementing the plan's `sort` step); `"largest"`/`"smallest"` -> `ORDER BY <params.by columns> DESC`/`ASC LIMIT params.n` directly, for a self-contained top/bottom-N with no separate `sort` step. Note: the validator's test-execution path strips and overrides any trailing `LIMIT` with its own `LIMIT 100` safety cap during validation only — this does not change what you should generate; still emit the plan's real `LIMIT params.n`.

If the Structured Plan includes a `rank` step: use the window function matching `params.method` — `"min"` -> `RANK()`, `"dense"` -> `DENSE_RANK()`, `"first"` -> `ROW_NUMBER()` (SQL plans are restricted to these three; never generate `"average"`/`"max"` tie-handling in SQL) — `OVER ([PARTITION BY <params.group_by columns>] ORDER BY <params.by columns> [DESC])`, aliased `AS <params.output_column>`. Omit `PARTITION BY` when `params.group_by` is absent; add `DESC` to `ORDER BY` when `params.ascending` is `false`.

### Output (conform to the Pydantic schema — `queries` must contain exactly ONE item)
- `question`: the plan's business question (faithfully preserved, see above)
- `query`: DuckDB SQL
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
- **Alias** — use only this in queries: `{alias}`
- **Table** — schema, columns, types, metadata: `{table}`
