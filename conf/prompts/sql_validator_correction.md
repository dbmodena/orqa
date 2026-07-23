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
- Date arithmetic on strings: `CAST(col AS DATE)`
- Comparisons (`>`, `<`, `>=`, `<=`, `=`, `BETWEEN`) against a literal of a different type — cast the column to match the literal: `CAST(str_col AS DOUBLE) > 100`, `CAST(str_date AS DATE) >= '2024-01-01'`. Mismatched types silently produce wrong results or runtime errors in DuckDB.

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

### Queries to Correct
Each query is provided as a JSON object matching the expected output schema.
Correct only the fields that are wrong. Return the same JSON structure.
{queries_with_errors}




