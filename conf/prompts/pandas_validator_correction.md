## PandasQueryCorrection
You are an expert Data Engineer. Correct only the listed Pandas queries and questions so they are executable, correct, and faithful to their business purpose.

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
- String-key merges: `.str.lower()` on both sides before merging.
- Never reference a column before a `.rename()` that creates it.
- Never use `pd.DataFrame()`, `pd.read_csv()`, or reassign DataFrame alias variables.
- Prefer method chaining.
- Every provided table is MANDATORY: never fix a query by removing a table, a `.merge()`, or a `pd.concat` branch — even if feedback suggests it. Fix the join type (`how='left'`), the join keys, or pre-aggregate instead; the query must keep referencing every table.

**String filters** — apply `.str.lower()` on both sides of string equality checks
(`df[df['col'].str.lower() == value.lower()]`); mismatched case silently drops all rows.
If the query is logically correct but likely yields no rows, relax or inspect filter
conditions — date ranges may be too narrow, numeric thresholds too strict, or string
filters may fail silently due to case mismatch.

**Chained merges (3+ DataFrames)** — suffix collisions accumulate across joins:
1. Declare `suffixes=` explicitly on every `.merge()`.
2. `.rename()` or `.drop()` any ambiguous column immediately after each merge, before the next one.
3. Finish with a column selection to keep only what's needed.

```python
result = (
    df_a
    .merge(df_b, on='key', how='left', suffixes=('', '_b'))
    .rename(columns={{'col_b': 'col_from_b'}})      # resolve before next merge
    .merge(df_c, on='key', how='left', suffixes=('', '_c'))
    .rename(columns={{'col_c': 'col_from_c'}})
    [['key', 'col_a', 'col_from_b', 'col_from_c']]
)
```

**Question fixes**
- Must exactly reflect what the code computes (filters, aggregations, scope).
- Domain-specific — no phrasing that could apply to any industry.
- No schema internals (DataFrame/column names, Pandas method names).
- `question`, `question_keywords`, `translated_question`, `translated_question_keywords`,
  `topic`, and `story` are ONE linked bundle — they were produced together during planning.
  If you rewrite `question`, you MUST regenerate all five of the others consistently with
  the new question (translate into `detected_language`, refresh keywords, refresh topic/story).
  If you do NOT change `question`, return all six of these fields EXACTLY as given, unchanged.

### Available DataFrames
{table_schemas}

### Queries to Correct
Each query is provided as a JSON object matching the expected output schema.
Correct only the fields that are wrong. Return the same JSON structure.
{queries_with_errors}