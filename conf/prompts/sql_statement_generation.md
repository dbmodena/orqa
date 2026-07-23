## SQLGeneration
Generate **exactly ONE** DuckDB SQL query that implements the single business question given to you below — not a new question you invent, and not a set of three. This call is one of several independent generation calls in the same run, each one bound to its own query plan; you are only ever answering the one plan attached to THIS call.

Below this prompt you will see the **Inputs** (detected languages, aliases, table schemas, verified table relationships), the **Column Statistics** for these tables, and the **Structured Plan** — the ordered decomposition steps for THIS query, including its `question` (the business question you must answer — do not replace it with a different one) and `task_types`.

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

### Table usage
A table is justified only if its columns appear in SELECT, WHERE, GROUP BY, or aggregations — not merely in a JOIN key. Using a table solely to restrict rows via a join is **not** justified unless the question explicitly requires cross-table validation (e.g. "find records appearing in both sources"). Touch only the minimal columns actually needed — this was already decided (and judged) during planning, so don't reach for columns beyond what the plan's steps call for.

### Difficulty
Assign whichever of `easy` / `medium` / `hard` genuinely reflects the complexity of THIS plan's steps — don't force it up or down to hit a particular tier:

| Level | Typically reflects |
|---|---|
| Easy | Single-table SELECT, basic WHERE, optional ORDER BY/LIMIT, ≤1 aggregate, no subqueries |
| Medium | Multi-table JOIN, GROUP BY + HAVING, multiple aggregates, nested filters, or one non-correlated subquery / UNION / INTERSECT / EXCEPT |
| Hard | Correlated or multi-level subqueries, window functions, complex set ops, CASE, CTEs (incl. recursive), or combinations of several advanced features |

### Output (conform to the Pydantic schema — `queries` must contain exactly ONE item)
- `difficulty`: easy / medium / hard
- `question`: the plan's business question (faithfully preserved, see above)
- `query`: DuckDB SQL
- `motivation`: 2–3 sentences in business language explaining (1) the analytical value, (2) what specific columns each table uniquely contributes, (3) why this join/union strategy is correct.

Note: `topic`, `story`, `translated_question`, `detected_language`,
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
