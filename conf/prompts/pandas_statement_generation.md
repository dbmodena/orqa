## PandasCodeGeneration
Generate 3 Python Pandas queries (easy → medium → hard) with business-focused natural language questions designed to benchmark a text-to-query engine. Questions must be written as if by a non-technical user with no knowledge of the schema, DataFrame names, or column names.

### Inputs
- **Detected languages** - The following datasets can have the following langauges `{languages}`
- **Aliases** — use only these DataFrame variable names in your code: `{aliases}`
- **Tables** — schema, columns, types, metadata: `{table}`
- **Mandatory join operations** — every query MUST combine DataFrames by following the steps below **exactly**:

```
{matches}
```

### Rename-first rule (NON-NEGOTIABLE)
When Step 0 lists `.rename()` calls, you MUST emit them at the top of your query — before any merge, filter, or aggregation. Use the renamed column names everywhere in the rest of the code. Never reference the original column names after renaming.

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
✗ Never mention table names, column names, DataFrame names, or any Pandas/Python operations.
✗ Never produce a question that could apply unchanged to a different industry or dataset.

### Query rules
- All DataFrames are **pre-loaded** with their designated aliases — start operations directly on them.
- The join order, join type (INNER, LEFT, etc.), key columns, and case-normalisation requirement shown in **Mandatory join operations** are NON-NEGOTIABLE.
- Chained merges must follow the listed step order — do not reorder or collapse steps.
- When a join is marked `Case-insensitive keys: YES`, you MUST apply `.str.lower()` on both sides before joining, using the pattern shown. Omitting this is a hard error.
- Never use `pd.DataFrame()`, `pd.read_csv()`, or reassign the original DataFrame variables (renaming into a new variable is fine and required when Step 0 is present).
- All DataFrames must be used and genuinely necessary (see Table Usage below).
- Only use explicitly defined match relationships — no inferred merges.
- Prefer method chaining. Correct Python/Pandas only.

### Table usage
A DataFrame is justified only if its columns appear in output, filters, or aggregations — not merely in a merge key. Using a DataFrame solely to restrict rows via a merge is **not** justified unless the question explicitly requires cross-table validation (e.g. "find records appearing in both sources"). Each table entry must list only the minimal columns actually used.

### Difficulty levels
| Level | Definition |
|---|---|
| Easy | Single DataFrame filtering, column selection, simple `sort_values`/`head`, ≤1 aggregation |
| Medium | `merge`/`join` of multiple DataFrames, `groupby` with multiple aggregations, compound boolean filters, or reshaping (`pivot`/`melt`) |
| Hard | Multi-step pipelines with several merges + groupby, window-style ops (`rolling`/`expanding`/`rank`), complex `apply`/custom functions, hierarchical indexes, or advanced reshaping combined |

### Per-query output (conform to Pydantic schema)
- `difficulty`: easy / medium / hard
- `question`: natural language question (phrased as a non-expert user would ask)
- `topic`: one short business topic or theme that best summarizes the query's primary analytical concern.
- `story`: one short business narrative describing the insight or storyline behind this query.
- `query`: Python/Pandas code
- `motivation`: 2–3 sentences in business language explaining (1) the analytical value, (2) what specific columns each DataFrame uniquely contributes, (3) why this merge/join strategy is correct. Must be distinct across the three queries.
- `tables`: list of `alias, columns_used[]` couples — minimal subset only
- `translated_question`: translated question into the detected target language.
- `detected_language`: detected language from the dataset.
 - `question_keywords`: list of keywords describing the question intent (max 10, English or source language). Should include relevant keywords from table analysis.
 - `translated_question_keywords`: list of the above keywords translated into the detected language (max 10).

Notes:
- Table-level `keywords` and `translated_keywords` (max 10 each) are generated once per table during the table-analysis phase and must be stored under each table entry in `tables` (do not repeat table keywords on every query separately).
- Each table entry in `tables` must include: `alias`, `columns_used[]`, `description`, `keywords` (max 10), and `translated_keywords` (max 10).
- Use table keywords to craft questions that naturally point to the right tables without explicitly mentioning them.