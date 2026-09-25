## Analyze
Analyze this Dataset and identify columns for UNION and JOIN operations.
### Table details
{table}

### Task Instructions

1. UNION TASKS (union_tasks):
   - Identify groups of columns that could be combined with similar columns from other datasets
   - Look for columns with similar names, types, or semantic meaning
   - Each entry: {{"columns": ["col1", "col2"]}}

2. JOIN TASKS (join_tasks):
   - Identify columns that could serve as join keys (unique identifiers or foreign keys)
   - Each entry: {{"columns": ["id_col1", "id_col2"]}}

3. JOIN-CORRELATION TASKS (join_correlation_tasks):
   - Identify pairs where one column is a join key and another correlates with it
   - Each entry: {{"join_column": "id_col", "correlation_column": "metric_col"}}
   - Examples: user_id with total_purchases, product_id with price
   - The correlation column MUST be numeric (e.g., amount, price, quantity, score, balance)

CRITICAL RULES:
1. Your response must be a SINGLE JSON OBJECT (not an array)
2. The object must have exactly these 3 fields: "union_tasks", "join_tasks", "join_correlation_tasks"
3. Each field is an ARRAY of objects
4. DO NOT include a "task" field in any object
5. DO NOT wrap your response in an array
6. Column names must exactly match those in the CSV


## Table
#### {filename} Information:
- Rows: {numrows}
- Columns: {numcolumns}


##### Dataset Metadata:
{metadata}

##### Column Details: 
{coldetails}

##### Sample Data (first 5 rows):
{sample}


## TableMatch
#### {filename} Information:
- Rows: {numrows}
- Columns: {numcolumns}


##### Dataset Metadata:
{metadata}

##### Column Details: 
{coldetails}

##### Sample Data (first 5 rows):
{sample}


---

## Match

### Task: {task}

You are given a set of database tables.

Your goal is to analyze how well each table matches with every other table, and assign a match score from 1 to 10 for every unordered pair of tables.

### Input

{table}


### Instructions

1. Identify all distinct tables present in the input.
2. Consider every possible unordered pair of tables.

    For N tables, you must produce exactly N·(N−1)/2 scores.
    Do not omit any pair.
    Do not include self-pairs (e.g., table A with itself).
3. For each table pair, assign a match score from 1 to 10, where:

    1 = no meaningful relationship
    10 = very strong relationship (e.g., clear foreign-key / primary-key join, strong semantic overlap)
4. Base your score on signals such as:

    Shared or joinable identifiers
    Semantic similarity of column names
    Overlap in data meaning or granularity
    Potential for meaningful joins or correlations
5. Treat table pairs as unordered:

    `(A, B)` is the same as `(B, A)`
    Provide only one score per pair
6. Ensure completeness and consistency:

    Every table must appear in exactly `N−1` pairs
    No duplicate or missing pairs are allowed






## Pydantic
 
### Output format instructions
Return the final answer as a JSON object that contains ONLY the data fields defined in the schema below. Do not include schema metadata such as "title", "description", "$defs", "type", "properties", "required", or any other schema definition keywords.

Schema: {format}

Your response must:
- Include all required fields with correct names and data types
- Satisfy all validation constraints (value ranges, uniqueness, completeness)
- Contain ONLY the actual data values, not the schema structure itself
- Be valid JSON without any explanatory text before or after
- Include only the data fields (not schema metadata like "properties", "type", "$defs", etc.)











## SQLGeneration
Generate DuckDB SQL queries with business-focused natural language questions.

### Data Context
You will be provided with:
- Alias of the dataset: Alias of the dataset name in order to create the queries.
- Dataset name: Name of the dataset
- Schema: Table structures, column names, data types, and relationships
- Metadata: Each table might have additional metadata that can assist in the question generation
- Match Definitions: Explicit DataFrame relationships specifying mandatory operations with merge keys, join types, and correlation logic

### Critical Rules
- Operation type is NON-NEGOTIABLE - must match the specified type exactly
- All tables must be used
- Only use explicitly defined matches - no inferred relationships
- DuckDB syntax only - ANSI SQL compatible

### Natural Language Questions
✓ Business terms (customers, revenue)  
✓ Outcome-focused (insights, trends)  
✗ No table/column names  
✗ No SQL operations mentioned

### Generate
3 queries of incremental difficulty (simple, hard and challenging) where:
- Each implements the mandatory operation type
- Each uses all provided tables
- Each follows match definitions exactly
- Each has semantically aligned NL question
- Each has a difficulty value that can be "simple", "hard" or challenging
- All conform to Pydantic schema

### SQL difficulty levels
- Simple: single-table `SELECT`, basic `WHERE`, optional `ORDER BY`/`LIMIT`, ≤1 aggregate, no subqueries.  
- Hard: multi-table `JOIN`, `GROUP BY`+`HAVING`, multiple aggregates, nested filters, one non-correlated subquery or `UNION`/`INTERSECT`/`EXCEPT`.  
- Challenging: correlated or multi-level subqueries, window functions, complex set ops, `CASE`, CTEs (incl. recursive), or combinations of several advanced features.

### Tables involved in the queries
Realize the queries and their natural language counterparts using the following tables. 
{table}  
### Mandatory operations
Every query MUST combine the tables with the following operations:
{matches}
### Lookup aliases
Make sure to use only the aliases in the queries in the following:
{aliases}


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




