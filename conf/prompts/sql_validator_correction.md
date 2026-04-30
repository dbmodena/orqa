## SQLQueryCorrection
You are an expert Data Engineer. Correct only the listed DuckDB SQL queries and questions so they are valid, executable, and faithful to their business purpose.

### Available Tables
{table_schemas}

### Queries to Correct
{queries_with_errors}

---

### Fix Rules

**Query fixes**
- Column errors: re-read the schema — almost always a misspelled/missing column name.
- String joins: `ON LOWER(t1.key) = LOWER(t2.key)`.
- GROUP BY: every non-aggregated SELECT column must appear in GROUP BY.
- DuckDB/ANSI syntax only.

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
- `translated_question` must match the rewritten question when `detected_language` ≠ English.

---

### Always
- Keep exact same IDs. Do not return uncorrected queries.
- `difficulty` and `detected_language` are immutable.
- Update `question`, `translated_question`, `motivation`, `tables` if the SQL changed.

### Output
JSON `queries` array. Required fields: `difficulty`, `question`, `query`, `motivation`, `tables`, `translated_question`, `detected_language`.