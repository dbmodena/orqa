## SingleTablePandasCodeGeneration
Generate 3 Python Pandas queries (easy → medium → hard) with business-focused natural language questions designed to benchmark a text-to-query engine. Questions must be written as if by a non-technical user with no knowledge of the schema, DataFrame name, or column names.

### Inputs
- **Detected languages** - The following datasets can have the following langauges `{languages}`
- **Alias** — use only this in queries: `{alias}`
- **Table** — schema, columns, types, metadata: `{table}`

### Question rules
Each question must read as if asked by a domain expert who understands the business but has no knowledge of the underlying data, schema, DataFrame name, or column names.

✓ **Anchor to a topic** — every question must be driven by a single, coherent business concern (e.g. promotion equity, customer churn, inventory turnover). Listing unrelated metrics side by side is not a topic.
✓ **Anchor to a domain** — the setting must be unambiguous from the question alone (e.g. corporate HR, e-commerce, hospital staffing). A reader must know *where* the data comes from without any schema knowledge.
✓ **Reflect the query faithfully** — every metric, filter, and grouping in the query must correspond to something explicitly asked in the question, and vice versa.
✓ **Be outcome-focused** — frame questions around a decision or insight a business user would act on, not around enumerating what data is available.
✗ Never mention table names, column names, DataFrame names, or any Pandas/Python operations.
✗ Never produce a question that could apply unchanged to a different industry or dataset.

### Query rules
- The DataFrame is **pre-loaded** with its designated alias — start operations directly on it.
- Never use `pd.DataFrame()`, `pd.read_csv()`, or reassign the DataFrame variable.
- Prefer method chaining. Correct Python/Pandas only.

### Difficulty levels
| Level | Definition |
|---|---|
| Easy | Single DataFrame filtering, column selection, simple `sort_values`/`head`, ≤1 aggregation |
| Medium | `groupby` with multiple aggregations, compound boolean filters, or reshaping (`pivot`/`melt`) |
| Hard | Multi-step pipelines with groupby, window-style ops (`rolling`/`expanding`/`rank`), complex `apply`/custom functions, hierarchical indexes, or advanced reshaping combined |

### Per-query output (conform to Pydantic schema)
- `difficulty`: easy / medium / hard
- `question`: natural language question
- `query`: Python/Pandas code
- `motivation`: 2–3 sentences in business language explaining (1) the analytical value, (2) which specific columns are used and why they matter. Must be distinct across the three queries.
- `tables`: list of `alias, columns_used[]` couples  — minimal subset only
- `translated_question`: translated question into the detected target language.
- `detected_language`: detected language from the dataset.