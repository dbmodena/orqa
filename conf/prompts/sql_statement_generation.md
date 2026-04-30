## SQLGeneration
Generate 3 DuckDB SQL queries (easy → medium → hard) with business-focused natural language questions designed to benchmark a text-to-query engine. Questions must be written as if by a non-technical user with no knowledge of the schema, table names, or column names.

### Inputs
- **Detected languages** - The following datasets can have the following langauges `{languages}`
- **Aliases** — use only these in queries: `{aliases}`
- **Tables** — schema, columns, types, metadata: `{table}`
- **Mandatory operations** — every query must combine tables using: `{matches}`

### Question rules
Each question must read as if asked by a domain expert who understands the business but has no knowledge of the underlying data, schema, tables, or column names.

✓ **Anchor to a topic** — every question must be driven by a single, coherent business concern (e.g. promotion equity, customer churn, inventory turnover). Listing unrelated metrics side by side is not a topic.
✓ **Anchor to a domain** — the setting must be unambiguous from the question alone (e.g. corporate HR, e-commerce, hospital staffing). A reader must know *where* the data comes from without any schema knowledge.
✓ **Reflect the query faithfully** — every metric, filter, and grouping in the query must correspond to something explicitly asked in the question, and vice versa.
✓ **Be outcome-focused** — frame questions around a decision or insight a business user would act on, not around enumerating what data is available.
✗ Never mention table names, column names, DataFrame names, or any query/SQL operations.
✗ Never produce a question that could apply unchanged to a different industry or dataset.

### Query rules
- Operation type is NON-NEGOTIABLE — match the specified type exactly.
- All tables must be used and genuinely necessary (see Table Usage below).
- Only use explicitly defined match relationships — no inferred joins.
- DuckDB/ANSI SQL syntax only.
- When joining on string keys, always apply `LOWER()` on both sides: `ON LOWER(t1.key) = LOWER(t2.key)`.

### Table usage
A table is justified only if its columns appear in SELECT, WHERE, GROUP BY, or aggregations — not merely in a JOIN key. Using a table solely to restrict rows via a join is **not** justified unless the question explicitly requires cross-table validation (e.g. "find records appearing in both sources"). Each table entry must list only the minimal columns actually used.

### Difficulty levels
| Level | Definition |
|---|---|
| Easy | Single-table SELECT, basic WHERE, optional ORDER BY/LIMIT, ≤1 aggregate, no subqueries |
| Medium | Multi-table JOIN, GROUP BY + HAVING, multiple aggregates, nested filters, or one non-correlated subquery / UNION / INTERSECT / EXCEPT |
| Hard | Correlated or multi-level subqueries, window functions, complex set ops, CASE, CTEs (incl. recursive), or combinations of several advanced features |

### Per-query output (conform to Pydantic schema)
- `difficulty`: easy / medium / hard
- `question`: natural language question
- `query`: DuckDB SQL
- `motivation`: 2–3 sentences in business language explaining (1) the analytical value, (2) what specific columns each table uniquely contributes, (3) why this join/union strategy is correct. Must be distinct across the three queries.
- `tables`: list of `alias, columns_used[]` couples  — minimal subset only
- `translated_question`: translated question into the detected target language.
- `detected_language`: detected language from the dataset.