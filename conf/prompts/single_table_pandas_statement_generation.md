## SingleTablePandasCodeGeneration
Generate python Pandas queries with business-focused natural language questions over a single table.
### Lookup alias
Use only the following alias in the queries:
{alias}

### Data Context
You will be provided with:
- Alias of the dataset: Alias of the dataset name in order to create the queries.
- Dataset name: Name of the dataset
- Schema: Table structure, column names, data types
- Metadata: The table might have additional metadata that can assist in the question generation

### Critical Rules
The DataFrame is PRE-LOADED and AVAILABLE:
- The DataFrame exists with its designated alias
- Start operations directly on the existing DataFrame
- NEVER create DataFrames with `pd.DataFrame()`, `pd.read_csv()`, or similar
- NEVER reassign the DataFrame variable (e.g., `Table_1 = ...`)
- ONLY reference the exact DataFrame alias provided
- Create correct Python and Pandas code only
- Prefer method chaining for clarity and efficiency

### Natural Language Questions
✓ Business terms (customers, revenue)
✓ Outcome-focused (insights, trends)
✗ No table/column names
✗ No Pandas or Python operations mentioned

### Generate
3 queries of incremental difficulty (easy, medium, hard) where:
- Each operates on the single provided table only
- Each has a semantically aligned NL question
- Each has a difficulty value that can be "easy", "medium" or "hard"
- Each has a **motivation**: a concise justification (2–3 sentences) explaining *why* this question is analytically valuable and what business insight or decision it supports
- Each table entry includes only the minimal column subset actually used to answer the question
- All conform to Pydantic schema

### Motivation Guidelines
The motivation must:
- Be written in business language, not technical terms
- Explain the analytical purpose (e.g., identifying churn risk, optimizing revenue, benchmarking performance)
- Justify what **specific columns** are used and why they matter for answering the question
- Be distinct across the three queries — avoid repeating the same business rationale

### Pandas query difficulty levels:
- Easy: single DataFrame filtering, column selection, simple `sort_values`/`head`, ≤1 aggregation (`sum/mean/count`).
- Medium: `groupby` with multiple aggregations, compound boolean filters, reshaping (`pivot`/`melt`), or one intermediate step.
- Hard: multi-step pipelines with groupby, window-style ops (`rolling`/`expanding`/`rank`), complex `apply`/custom functions, hierarchical indexes, or advanced reshaping combined together.

### Table involved in the queries
Realize the queries and their natural language counterparts using the following table.
{table}
