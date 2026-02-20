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