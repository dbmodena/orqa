## SingleTableSQLGeneration
Generate **exactly ONE** DuckDB SQL query that implements the single business question given to you below — not a new question you invent, and not a set of three. This call is one of several independent generation calls in the same run, each one bound to its own query plan; you are only ever answering the one plan attached to THIS call.

Below this prompt you will see the **Inputs** (detected languages, alias, table schema), the **Column Statistics** for this table, and the **Structured Plan** — the ordered decomposition steps for THIS query, including its `question` (the business question you must answer — do not replace it with a different one).

### Question rule — use the plan's question, don't invent one
- The `question` field of the Structured Plan below is the exact business question this query must answer. Copy its intent faithfully into your output's `question`; you may only lightly polish phrasing (grammar, fluency), never its topic, scope, or the metrics/filters it implies.
- The question reads as if asked by an average, non-expert user who does not know table or column names, has a general understanding of the business domain, and would recognize it phrased naturally and conversationally — the plan’s question is already written this way, so preserve that framing.
- The plan's question deliberately carries the distinctive subject vocabulary (topic, entity type, agency, place, period) that a downstream keyword extractor uses to retrieve the right table from a reverse index. Any polish you apply must keep those identifying terms intact — never paraphrase them away into generic wording.
✗ Never mention table names, column names, or any query/SQL operations in the question.

### Query rules
- Only reference the single provided alias — never reference other tables.
- DuckDB/ANSI SQL syntax only. Correct and executable code only.
- Follow the Structured Plan's steps, in order — they are the validated decomposition of this exact question.

### Difficulty
Assign whichever of `easy` / `medium` / `hard` genuinely reflects the complexity of THIS plan's steps — don't force it up or down to hit a particular tier:

| Level | Typically reflects |
|---|---|
| Easy | Basic SELECT, WHERE, optional ORDER BY/LIMIT, ≤1 aggregate, no subqueries |
| Medium | GROUP BY + HAVING, multiple aggregates, nested filters, one non-correlated subquery, or UNION/INTERSECT/EXCEPT on the same table |
| Hard | Correlated or multi-level subqueries, window functions, CASE, CTEs (incl. recursive), or combinations of several advanced features |

### Output (conform to the Pydantic schema — `queries` must contain exactly ONE item)
- `difficulty`: easy / medium / hard
- `question`: the plan's business question (faithfully preserved, see above)
- `query`: DuckDB SQL
- `motivation`: 2–3 sentences in business language explaining (1) the analytical value, (2) which specific columns are used and why they matter.

Note: `topic`, `story`, `translated_question`, `detected_language`,
`question_keywords`, `translated_question_keywords`, and `tables` are NOT
part of this output — the query plan already decided them (table usage,
justification, and columns were already fixed and judged during planning)
and they are attached to your query automatically. Do not generate them.

Notes:
- Return a `queries` list containing exactly one query object — not three.

### Inputs
- **Detected languages** - The following datasets can have the following langauges `{languages}`
- **Alias** — use only this in queries: `{alias}`
- **Table** — schema, columns, types, metadata: `{table}`
