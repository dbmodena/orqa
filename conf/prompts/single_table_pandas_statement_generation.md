## SingleTablePandasCodeGeneration
Generate 3 Python Pandas queries (easy → medium → hard) with business-focused natural language questions designed to benchmark a text-to-query engine. Questions must be written as if by a non-technical user with no knowledge of the schema, DataFrame name, or column names.

### Inputs
- **Alias** — use only this in queries: `{alias}`
- **Table** — schema, columns, types, metadata: `{table}`

### Question rules
✓ Use business terms (customers, revenue, churn)
✓ Be outcome-focused and self-contained
✓ Use concrete, domain-specific terms that anchor the question to this dataset — a question that could apply unchanged to a hospital or financial database is invalid
✗ Never mention DataFrame/column names or Pandas/Python operations

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
