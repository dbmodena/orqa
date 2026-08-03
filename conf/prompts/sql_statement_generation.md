## SQLGeneration
Generate **exactly ONE** DuckDB SQL query that implements the single business question given to you below — not a new question you invent, and not a set of three. This call is one of several independent generation calls in the same run, each one bound to its own query plan; you are only ever answering the one plan attached to THIS call.

Below this prompt you will see the **Inputs** (detected languages, aliases, table schemas, verified table relationships), the **Column Statistics** for these tables, and the **Structured Plan** — the ordered decomposition steps for THIS query, including its `question` (the business question you must answer — do not replace it with a different one).

### Question rule — use the plan's question, don't invent one
- The `question` field of the Structured Plan below is the exact business question this query must answer. Copy its intent faithfully into your output's `question`; you may only lightly polish phrasing (grammar, fluency), never its topic, scope, or the metrics/filters it implies.
- The question reads as if asked by an average, non-expert user who does not know table or column names, has a general understanding of the business domain, and would recognize it phrased naturally and conversationally — the plan's question is already written this way, so preserve that framing.
- Use table keywords from the table analysis when they help the question stay non-technical and recognizable, without adding technical detail not present in the plan.
- The plan's question deliberately carries the distinctive subject vocabulary (topic, entity type, agency, place, period) that a downstream keyword extractor uses to retrieve the right table from a reverse index. Any polish you apply must keep those identifying terms intact — never paraphrase them away into generic wording.
✗ Never mention table names, column names, DataFrame names, or any query/SQL operations in the question.

### Query rules
- Operation type is NON-NEGOTIABLE — implement the plan's steps exactly, and combine tables only through the verified relationships listed below (same keys, same operation type). The relationships are building blocks, not a prescribed chain: compose them in whatever shape the plan dictates — a chain of joins, or independent joins whose results are compared/combined.
- All tables used must be genuinely necessary (see Table Usage below).
- Only use explicitly defined match relationships — no inferred joins.
- DuckDB/ANSI SQL syntax only.
- When joining on string keys, always apply `LOWER()` on both sides: `ON LOWER(t1.key) = LOWER(t2.key)`.
- Follow the Structured Plan's steps, in order — they are the validated decomposition of this exact question.
- Column-name quoting (mandatory): ALWAYS wrap every column reference in double quotes — `"column name"` — even for names that look like ordinary identifiers. Some columns in these tables contain spaces, accented/non-ASCII characters, punctuation, or are purely numeric text (e.g. `"2023"`); left unquoted, a purely-numeric name is silently parsed as a numeric LITERAL instead of a column reference (wrong result, no error) and the others break the query outright. If a column name itself contains a `"`, double it: `col "a" b` -> `"col ""a"" b"`.

### Data quality — tables are RAW
Tables are loaded exactly as stored: no bad-token→NULL conversion, no numeric coercion, no null-row dropping has been applied upstream. If the Structured Plan includes a `clean` step, implement its `params.actions` literally in SQL using the EXACT literal tokens shown in Column Statistics' `bad_token_counts` (never invented ones): `CASE WHEN col IN ('n/a', 'not available') THEN NULL ELSE col END` to blank a sentinel, `TRY_CAST(col AS DOUBLE)` — never a raw `CAST`, which errors on any stray token — to cast, `WHERE col IS NOT NULL` to drop rows, `COALESCE(col, <value>)` to impute a constant. If a column your query touches shows a non-trivial `bad_token_counts` or a `numeric_parseable_ratio` near 1.0 but no `clean` step covers it, still handle it defensively (`TRY_CAST` instead of `CAST`) rather than letting a stray token crash the query. If an action carries `"treat_as_missing"`, fold those exact values into the same `CASE`/`IN (...)` blanking before whatever the action does next — they're planner-discovered sentinels not already in `bad_token_counts`.

If the Structured Plan includes a `derive` step whose `params.actions` use `"technique": "flag"` or `"bucket"` (preserving an outlier/censoring pattern as its own feature instead of discarding it): implement `flag` as `CASE WHEN col ~ '<pattern>' OR col IN (...) THEN TRUE ELSE FALSE END AS output_column`, and `bucket` as a `CASE WHEN ... THEN 'label' WHEN ... THEN 'label' ELSE ... END AS output_column` mapping the ranges/patterns named in the action's `rule` to their labels — using the exact values/patterns from `numeric_outliers`/`minority_value_groups`, never invented ones.

### Correlate / limit / rank steps
If the Structured Plan includes a `correlate` step: use DuckDB's `corr(col_a, col_b)` aggregate — Pearson only, so `params.method` will always be `"pearson"` for a SQL plan (never generate Spearman/Kendall SQL). Add `GROUP BY <params.group_by columns>` when `params.group_by` is set, for one coefficient per group instead of one over the whole table; alias the result `AS <params.output_column>` when that key is present.

If the Structured Plan includes a `limit` step: `params.how` (default `"head"`) selects the idiom — `"head"` -> a bare `LIMIT params.n` applied on top of the query's existing row order (typically right after an `ORDER BY` implementing the plan's `sort` step); `"largest"`/`"smallest"` -> `ORDER BY <params.by columns> DESC`/`ASC LIMIT params.n` directly, for a self-contained top/bottom-N with no separate `sort` step. Note: the validator's test-execution path strips and overrides any trailing `LIMIT` with its own `LIMIT 100` safety cap during validation only — this does not change what you should generate; still emit the plan's real `LIMIT params.n`.

If the Structured Plan includes a `rank` step: use the window function matching `params.method` — `"min"` -> `RANK()`, `"dense"` -> `DENSE_RANK()`, `"first"` -> `ROW_NUMBER()` (SQL plans are restricted to these three; never generate `"average"`/`"max"` tie-handling in SQL) — `OVER ([PARTITION BY <params.group_by columns>] ORDER BY <params.by columns> [DESC])`, aliased `AS <params.output_column>`. Omit `PARTITION BY` when `params.group_by` is absent; add `DESC` to `ORDER BY` when `params.ascending` is `false`.

### Table usage
A table is justified only if its columns appear in SELECT, WHERE, GROUP BY, or aggregations — not merely in a JOIN key. Using a table solely to restrict rows via a join is **not** justified unless the question explicitly requires cross-table validation (e.g. "find records appearing in both sources"). Touch only the minimal columns actually needed — this was already decided (and judged) during planning, so don't reach for columns beyond what the plan's steps call for.

### Output (conform to the Pydantic schema — `queries` must contain exactly ONE item)
- `question`: the plan's business question (faithfully preserved, see above)
- `query`: DuckDB SQL
- `motivation`: 2–3 sentences in business language explaining (1) the analytical value, (2) what specific columns each table uniquely contributes, (3) why this join/union strategy is correct.

Note: `difficulty`, `topic`, `story`, `translated_question`, `detected_language`,
`question_keywords`, `translated_question_keywords`, and `tables` are NOT
part of this output — the query plan already decided them (table usage,
justification, and columns were already fixed and judged during planning)
and they are attached to your query automatically. Do not generate them.

Notes:
- Return a `queries` list containing exactly one query object — not three.

### Inputs
- **Detected languages** - The following datasets can have the following langauges `{languages}`
- **Aliases** — use only these in queries: `{aliases}`
- **Tables** — schema, columns, types, metadata: `{table}`
- **Verified table relationships** — the ONLY ways tables may be combined; use each relationship's keys exactly as listed, composed in the shape the Structured Plan dictates: `{matches}`
