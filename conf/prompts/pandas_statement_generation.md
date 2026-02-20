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

### Natural Language Questions
✓ Business terms (customers, revenue)  
✓ Outcome-focused (insights, trends)  
✗ No table/column names  
✗ No Pandas or Python operations mentioned

### Generate
3 queries of incremental difficulty (simple, hard and challenging) where:
- Each implements the mandatory operation type
- Each uses all provided tables
- Each follows match definitions exactly
- Each has semantically aligned NL question
- Each has a difficulty value that can be "simple", "hard" or challenging
- All conform to Pydantic schema

### Pandas query difficulty levels:
- Simple: single DataFrame filtering, column selection, simple `sort_values`/`head`, ≤1 aggregation (`sum/mean/count`).  
- Hard: `merge`/`join` of multiple DataFrames, `groupby` with multiple aggregations, compound boolean filters, reshaping (`pivot`/`melt`) or one intermediate step.  
- Challenging: multi-step pipelines with several merges + groupby, window-style ops (`rolling`/`expanding`/`rank`), complex `apply`/custom functions, hierarchical indexes, or advanced reshaping combined together.

### Tables involved in the queries
Realize the queries and their natural language counterparts using the following tables. 
{table}  
### Mandatory operations
Every query MUST combine the tables with the following operations:
{matches}
### Lookup aliases
Make sure to use only the aliases in the queries in the following:
{aliases}