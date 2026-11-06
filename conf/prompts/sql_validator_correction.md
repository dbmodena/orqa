## SQLQueryCorrection
You are an expert Data Engineer. Correct only the listed DuckDB SQL queries and questions so they are valid, executable, and faithful to their business purpose.

### Output Format
{pydantic_constraint}

Focus on what you are correcting: ALWAYS return the corrected `code`, and — only if you
rewrote the question — the full question bundle (`question`, `question_keywords`,
`translated_question`, `translated_question_keywords`, `topic`, `story`) regenerated
consistently. Any field you omit is kept unchanged from the original query, so do not
regenerate fields (e.g. `difficulty`, `tables`) that are not part of the correction.

### Fix Rules

**Query fixes**
- Column errors: re-read the schema — almost always a misspelled/missing column name.
- String joins: `ON LOWER(t1.key) = LOWER(t2.key)`.
- GROUP BY: every non-aggregated SELECT column must appear in GROUP BY.
- DuckDB/ANSI syntax only.
- Every provided table is MANDATORY: never fix a query by removing a table, a JOIN, or a UNION branch — even if feedback suggests it. Fix the join type (LEFT JOIN), the join keys, or pre-aggregate instead; the query must keep referencing every table.

**CAST** — cast when the source column is not already the required type:
- `CORR`, `STDDEV_SAMP`, `VAR_SAMP`: both args → `CAST(col AS DOUBLE)`
- `SUM` on booleans: `SUM(CAST(flag AS INTEGER))`
- Integer division: `CAST(numerator AS DOUBLE) / denominator`
- Text columns never get a raw `CAST`: a `number text` column is `TRY_CAST` after removing the separators/affixes its Column Statistics fact shows (`plain`: `TRY_CAST(trim(col) AS DOUBLE)`); a `date text` column is `TRY_CAST(trim(col) AS TIMESTAMP)` when ISO with no `offsets` fact, `timezone('UTC', TRY_CAST(trim(col) AS TIMESTAMPTZ))` when offsets are on every value, else `try_strptime(trim(col), '<format>')` with the format shown (its `(DuckDB: ...)` spelling when given) — never `CAST(... AS TIMESTAMPTZ) AS TIMESTAMP`, which follows the session zone; `date as number yyyymmdd` is `try_strptime(CAST(TRY_CAST(col AS BIGINT) AS VARCHAR), '%Y%m%d')`; a `digit code` stays text.
- Comparisons (`>`, `<`, `>=`, `<=`, `=`, `BETWEEN`) against a literal of a different type — convert the column to match the literal as above: `TRY_CAST(trim(str_col) AS DOUBLE) > 100`, `try_strptime(trim(str_date), '%d/%m/%Y') >= DATE '2024-01-01'`. Mismatched types silently produce wrong results or runtime errors in DuckDB.

**Question fixes**
- Must exactly reflect what the SQL computes (filters, aggregations, scope).
- Domain-specific — no phrasing that could apply to any industry.
- No schema internals (table/column names, SQL keywords).
- `question`, `question_keywords`, `translated_question`, `translated_question_keywords`,
  `topic`, and `story` are ONE linked bundle — they were produced together during planning.
  If you rewrite `question`, you MUST regenerate all five of the others consistently with
  the new question (translate into `detected_language`, refresh keywords, refresh topic/story).
  If you do NOT change `question`, return all six of these fields EXACTLY as given, unchanged.

### Available Tables
{table_schemas}

### Column Statistics
Read with this legend:
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

{column_statistics}

### Queries to Correct
Each query is provided as a JSON object matching the expected output schema.
Correct only the fields that are wrong. Return the same JSON structure.
{queries_with_errors}




