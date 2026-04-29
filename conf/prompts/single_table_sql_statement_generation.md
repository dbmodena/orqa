## SingleTableSQLGeneration
Generate 3 DuckDB SQL queries (easy → medium → hard) with business-focused natural language questions designed to benchmark a text-to-query engine. Questions must be written as if by a non-technical user with no knowledge of the schema, table name, or column names.

### Inputs
- **Alias** — use only this in queries: `{alias}`
- **Table** — schema, columns, types, metadata: `{table}`

### Question rules
✓ Use business terms (customers, revenue, churn)
✓ Be outcome-focused and self-contained
✓ Use concrete, domain-specific terms that anchor the question to this dataset — a question that could apply unchanged to a hospital or financial database is invalid
✗ Never mention table/column names or SQL operations

### Query rules
- Only reference the single provided alias — never reference other tables.
- DuckDB/ANSI SQL syntax only. Correct and executable code only.

### Difficulty levels
| Level | Definition |
|---|---|
| Easy | Basic SELECT, WHERE, optional ORDER BY/LIMIT, ≤1 aggregate, no subqueries |
| Medium | GROUP BY + HAVING, multiple aggregates, nested filters, one non-correlated subquery, or UNION/INTERSECT/EXCEPT on the same table |
| Hard | Correlated or multi-level subqueries, window functions, CASE, CTEs (incl. recursive), or combinations of several advanced features |

### Per-query output (conform to Pydantic schema)
- `difficulty`: easy / medium / hard
- `question`: natural language question
- `query`: DuckDB SQL
- `motivation`: 2–3 sentences in business language explaining (1) the analytical value, (2) which specific columns are used and why they matter. Must be distinct across the three queries.
- `tables`: list of `alias, columns_used[]` couples — minimal subset only
- `translated_question`: translated question into the detected target language.
- `detected_language`: detected language from the dataset.