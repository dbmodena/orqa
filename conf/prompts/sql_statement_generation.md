## SQLGeneration
Generate 3 DuckDB SQL queries (easy → medium → hard) with business-focused natural language questions designed to benchmark a text-to-query engine. Questions must be written as if by a non-technical user with no knowledge of the schema, table names, or column names.

### Inputs
- **Detected languages** - The following datasets can have the following langauges `{languages}`
- **Aliases** — use only these in queries: `{aliases}`
- **Tables** — schema, columns, types, metadata: `{table}`
- **Mandatory operations** — every query must combine tables using: `{matches}`

### Question rules
Each question must read as if asked by an average, non-expert user who:
- Does NOT know table or column names
- Has a general understanding of the business domain
- Phrases questions naturally and conversationally
- Uses contextual keywords from the provided table keywords to hint at relevant tables (without naming them directly)

✓ **Anchor to a topic** — every question must be driven by a single, coherent business concern (e.g. promotion equity, customer churn, inventory turnover). Listing unrelated metrics side by side is not a topic.
✓ **Anchor to a domain** — the setting must be unambiguous from the question alone (e.g. corporate HR, e-commerce, hospital staffing). A reader must know *where* the data comes from without any schema knowledge.
✓ **Use table keywords strategically** — incorporate keywords from the table analyses to make the question pinpoint to the correct tables. Example: if a table has keywords "sales", "revenue", "orders", use them naturally in the question.
✓ **Reflect the query faithfully** — every metric, filter, and grouping in the query must correspond to something explicitly asked in the question, and vice versa.
✓ **Be outcome-focused** — frame questions around a decision or insight a business user would act on, not around enumerating what data is available.
✓ **Name a single topic** — every query must be centered on one clear business topic, and that topic must be exposed through the `topic` field.
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
- `question`: natural language question (phrased as a non-expert user would ask)
- `topic`: one short business topic or theme that best summarizes the query's primary analytical concern.
- `story`: one short business narrative describing the insight or storyline behind this query.
- `query`: DuckDB SQL
- `motivation`: 2–3 sentences in business language explaining (1) the analytical value, (2) what specific columns each table uniquely contributes, (3) why this join/union strategy is correct. Must be distinct across the three queries.
- `tables`: list of `alias, columns_used[]` couples  — minimal subset only
- `translated_question`: translated question into the detected target language.
- `detected_language`: detected language from the dataset.
 - `question_keywords`: list of keywords describing the question intent (max 10, English or source language). Should include relevant keywords from table analysis.
 - `translated_question_keywords`: list of the above keywords translated into the detected language (max 10).

Notes:
- Table-level `keywords` and `translated_keywords` (max 10 each) are generated once per table during the table-analysis phase and must be stored under each table entry in `tables` (do not repeat table keywords on every query separately).
- Each table entry in `tables` must include: `alias`, `columns_used[]`, `description`, `keywords` (max 10), and `translated_keywords` (max 10).
- Use table keywords to craft questions that naturally point to the right tables without explicitly mentioning them.