## PandasCodeGeneration
Generate python Pandas queries with business-focused natural language questions.
### Data Context
You will be provided with:
- Alias of the dataset: Alias of the dataset name in order to create the queries.
- Dataset name: Name of the dataset
- Schema: Table structures, column names, data types, and relationships
- Metadata: Each table might have additional metadata that can assist in the question generation
- Match Definitions: Explicit DataFrame relationships specifying mandatory operations with merge keys, join types, and correlation logic

### Critical Rules
All DataFrames are PRE-LOADED and AVAILABLE:
- DataFrames exist with their designated aliases
- Start operations directly on existing DataFrames
- NEVER create DataFrames with `pd.DataFrame()`, `pd.read_csv()`, or similar
- NEVER reassign DataFrame variables (e.g., `Table_1 = ...`)
- ONLY reference the exact DataFrame aliases provided
- Operation type is NON-NEGOTIABLE - must match the specified type exactly
- All Dataframes must be used
- Only use explicitly defined matches - no inferred relationships
- Create correct Python and Pandas code only
- Prefer method chaining for clarity and efficiency

### Merge/Join Rules
When merging or joining DataFrames, ALWAYS apply `.str.lower()` to string-type merge keys on BOTH sides before joining:
```python
df1.merge(df2, left_on="key_col_1".str.lower(), right_on="key_col_2".str.lower(), ...)
# or, if mutating is needed before chaining:
df1.assign(key=df1["key_col"].str.lower()).merge(
    df2.assign(key=df2["key_col"].str.lower()),
    on="key", ...
)
```
This prevents silent mismatches caused by inconsistent casing in string keys.

### Natural Language Questions
✓ Business terms (customers, revenue)  
✓ Outcome-focused (insights, trends)  
✗ No table/column names  
✗ No Pandas or Python operations mentioned

### Generate
3 queries of incremental difficulty (easy, medium and hard) where:
- Each implements the mandatory operation type
- Each uses all provided tables
- Each follows match definitions exactly
- Each has semantically aligned NL question
- Each has a difficulty value that can be "easy", "medium" or "hard"
- Each has a **motivation**: a concise justification (2–3 sentences) explaining *why* this question is analytically valuable, *why each table is necessary*, and *why the specified join/union strategy is the correct way to combine them
- Each table entry includes only the minimal column subset actually used to answer the question
- All conform to Pydantic schema

### Motivation Guidelines
The motivation must:
- Be written in business language, not technical terms
- Explain the analytical purpose (e.g., identifying churn risk, optimizing revenue, benchmarking performance)
- Justify why each dataset is required and what unique information it contributes to answering the question
- Justify why the tables are combined in the specified way (join vs union, which keys, what the combination unlocks)
- Be distinct across the three queries — avoid repeating the same business rationale

### Pandas query difficulty levels:
- Easy: single DataFrame filtering, column selection, simple `sort_values`/`head`, ≤1 aggregation (`sum/mean/count`).  
- Medium: `merge`/`join` of multiple DataFrames, `groupby` with multiple aggregations, compound boolean filters, reshaping (`pivot`/`melt`) or one intermediate step.  
- Hard: multi-step pipelines with several merges + groupby, window-style ops (`rolling`/`expanding`/`rank`), complex `apply`/custom functions, hierarchical indexes, or advanced reshaping combined together.

### Tables involved in the queries
Realize the queries and their natural language counterparts using the following tables. 
{table}  
### Mandatory operations
Every query MUST combine the tables with the following operations:
{matches}
### Lookup aliases
Make sure to use only the aliases in the queries in the following:
{aliases}