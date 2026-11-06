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
The table is loaded exactly as stored: no missing-token conversion, no number or date parsing, no null-row dropping. Read the Column Statistics below with this legend:
`Table_i: <rows> rows; <table fact>; ...`, then one line per column: `- name (type): fact; fact; ...`.
`v ×n` = value v in n rows; a bare count (`12 null`, `3 untrimmed`) counts rows; `blank` = empty or whitespace-only text, which is not NULL.
"..." = a text value exactly as stored (JSON-escaped): in SQL write it in single quotes, doubling any '. `"abc"…` = a longer stored value starting with abc (match with LIKE 'abc%' / .str.startswith). Unquoted values are numbers or booleans; dates appear as YYYY-MM-DD after the word `parsed`, and dates are UTC: `parsed a..b` is UTC, and the recipes give UTC; convert to a local zone only when the plan names it. `offsets +00:00 on every value` / `on n of m values` = how many of the ISO values carry a UTC offset.
`<k> values:` lists EVERY value; `1 value: v ×n` = v is the only stored value, the other rows are null or blank. `most common` lists only frequent ones; `+t more ×c each` = t more values with c rows each; `<k> distinct, each ×c` = every value in exactly c rows.
Shown numbers are stored values (compare non-whole floats with a tolerance, never ==), except: `quartiles` are rounded summaries; `a..b` (min..max), `far high` values and `all others <= x` are rounded outward past long decimals.
`(missing token)` / `missing tokens` = values matching the configured missing-value spellings; usually missing, but you decide (a "-" can be a real category).
`number text` casts to a number but may still be an identifier (ZIP, id, house number): cast only for arithmetic or numeric comparison. `date as number yyyymmdd` = whole numbers that are calendar dates (20200416): parse them, never add or average them.
`identifier:` = structural evidence that the column names records: never sum, average or correlate it, count its distinct values, join only on the same id space; a measure constant per category can read as one — judge by its name and values. `one row per x` in the header = each row is one x.
`filled down:` = a spreadsheet fill-down: a value holds for the blank rows below it until the next value — forward-fill it before grouping, filtering or counting per value; `(k once x is filled down)` on repeated rows = the repeats left after that fill.
Backticked text is a pattern or a literal fragment (an affix such as `%`), never a whole stored value: in `patterns` and `not ...` groups only `#` (a run of digits) is a placeholder and letters are literal; in `code shape` and `date-like text` shapes `9` = one digit and `a` = a run of letters.
An absent fact means nothing to report.

If the plan has a `clean` step, implement its `params.actions` literally, with the EXACT values Column Statistics quotes — never invented ones, never a backticked shape:
- missing tokens and an action's `"treat_as_missing"` text values: `CASE WHEN lower(trim("col")) IN ('n/a', '-') THEN NULL ELSE "col" END` (the shown values, lowercased and trimmed); a numeric sentinel in a numeric column: `CASE WHEN "col" IN (-1, 999) THEN NULL ELSE "col" END`
- `cast` to a number — always `TRY_CAST`, never a raw `CAST`, which errors on any stray value. `number text: plain`: `TRY_CAST(trim("col") AS DOUBLE)`. Any other number-text format (the action's `format`): strip the shown prefix/suffix/unit, remove the shown thousands separator (`thousands space`: `regexp_replace(x, '[\s\pZ]', '', 'g')`), replace the shown decimal mark with '.', then `TRY_CAST(... AS DOUBLE)` kept only when `isfinite(...)` — e.g. `decimal comma; thousands dot`: `TRY_CAST(replace(replace(trim("col"), '.', ''), ',', '.') AS DOUBLE)`. Ambiguous or mixed: use the format the action states.
- `cast` to a date — the result is a UTC TIMESTAMP (a value with no offset is taken as UTC). `date text ISO` with no `offsets` fact: `TRY_CAST(trim("col") AS TIMESTAMP)` (`AS DATE` when there is no time part); `offsets ... on every value`: `timezone('UTC', TRY_CAST(trim("col") AS TIMESTAMPTZ))`; `offsets ... on n of m values`: `CASE WHEN regexp_matches(trim("col"), '\d\d:\d\d(:\d\d(\.\d+)?)?\s*(Z|[+-]\d\d(:?\d\d)?)$') THEN timezone('UTC', TRY_CAST(trim("col") AS TIMESTAMPTZ)) ELSE TRY_CAST(trim("col") AS TIMESTAMP) END`. Any other shown format: `try_strptime(trim("col"), '<format>')`, e.g. `'%m/%d/%Y %I:%M:%S %p'` (a fact showing `(DuckDB: <format>)`: use that spelling); order unknown: use the single format the action states. A fact with `also <format>` holds a second format: `try_strptime(trim("col"), ['<2-digit-year format>', '<other>'])`, the `%y` format FIRST (`%Y` would read `4/5/14` as year 14). Never compare raw date text with a parsed date; never `CAST(... AS TIMESTAMPTZ) AS TIMESTAMP` (it follows the session zone) and never `SET TimeZone`.
- `date as number yyyymmdd`: `try_strptime(CAST(TRY_CAST("col" AS BIGINT) AS VARCHAR), '%Y%m%d')` (the BIGINT cast drops the `.0` a float column prints).
- `cast` to a time (`time text`): `TRY_CAST(trim("col") AS TIME)` — never `AS DATE` or `AS TIMESTAMP`, which turn every value NULL.
- `impute` with `"strategy": "ffill"` (a `filled down` column): the table keeps its file order, so number the rows and carry the last value forward: `WITH o AS (SELECT *, row_number() OVER () AS rn FROM "Table_0") SELECT last_value(NULLIF(trim("col"), '') IGNORE NULLS) OVER (ORDER BY rn ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS "col", ... FROM o` (`NULLIF`: the void cells are blank text, not NULL).
- UTC timestamps converted to local time (when the plan says so; it names the zone): `timezone('<zone>', timezone('UTC', <parsed timestamp>))` is the local TIMESTAMP — derive day, month, year or hour from it.
- `WHERE "col" IS NOT NULL` to drop rows; `COALESCE("col", <value>)` to impute

Recipes for the other facts:
- `digit code`: compare as text (`"col" = '00123'`), never cast.
- `untrimmed` / `case/space variants`: compare and group on `lower(trim(regexp_replace("col", '[\s\pZ]+', ' ', 'g')))`.
- `blank`: missing like NULL, but `IS NOT NULL` keeps it: use `NULLIF(trim("col"), '')`.
- `<k> infinite`: those rows make SUM/AVG infinite: keep `isfinite("col")` rows before aggregating.
- excluding a `total row`: `"<label col>" IS DISTINCT FROM '<label>'`, or with blank labels `"<measure>" IS DISTINCT FROM <value>` (number text: the stored text, quoted) — never by row number, and not `<>`, which also drops the rows where that column is NULL.
- a text column used as a number with no `clean` step casting it: still `TRY_CAST` it rather than letting a stray value crash the query.

If the plan has a `derive` step whose `params.actions` use `"technique": "flag"` or `"bucket"` (preserving a censored, capped or extreme value as a feature instead of discarding it): implement `flag` as `CASE WHEN col ~ '<pattern>' OR col IN (...) THEN TRUE ELSE FALSE END AS output_column`, and `bucket` as `CASE WHEN ... THEN 'label' ... ELSE ... END AS output_column`, mapping the ranges/patterns in the action's `rule` to their labels — using the exact values and bounds Column Statistics shows (a `far high` bound, a `repeated min`/`max` value, a `not numbers` value), never invented ones.

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
