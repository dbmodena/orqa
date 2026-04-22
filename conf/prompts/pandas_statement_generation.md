## PandasCodeGeneration
Generate 3 Python Pandas queries (easy → medium → hard) with business-focused natural language questions designed to benchmark a text-to-query engine. Questions must be written as if by a non-technical user with no knowledge of the schema, DataFrame names, or column names.

### Inputs
- **Aliases** — use only these in queries: `{aliases}`
- **Tables** — schema, columns, types, metadata: `{table}`
- **Mandatory operations** — every query must combine DataFrames using: `{matches}`

### Question rules
✓ Use business terms (customers, revenue, churn)
✓ Be outcome-focused and self-contained
✓ Use concrete, domain-specific terms that anchor the question to this dataset — a question that could apply unchanged to a hospital or financial database is invalid
✗ Never mention DataFrame/column names or Pandas/Python operations

### Query rules
- All DataFrames are **pre-loaded** with their designated aliases — start operations directly on them.
- Operation type is NON-NEGOTIABLE — match the specified type exactly.
- All DataFrames must be used and genuinely necessary (see Table Usage below).
- Only use explicitly defined match relationships — no inferred merges.
- Never use `pd.DataFrame()`, `pd.read_csv()`, or reassign DataFrame variables.
- Prefer method chaining. Correct Python/Pandas only.
- When merging on string keys, apply `.str.lower()` on both sides before joining:
  ```python
  df1.assign(key=df1["k"].str.lower()).merge(
      df2.assign(key=df2["k"].str.lower()), on="key", ...
  )
  ```

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
- `question`: natural language question
- `query`: Python/Pandas code
- `motivation`: 2–3 sentences in business language explaining (1) the analytical value, (2) what specific columns each DataFrame uniquely contributes, (3) why this merge/join strategy is correct. Must be distinct across the three queries.
- `tables`: list of `alias, columns_used[]` couples  — minimal subset only
