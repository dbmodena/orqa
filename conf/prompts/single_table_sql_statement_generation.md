## SingleTableSQLGeneration
Generate DuckDB SQL queries with business-focused natural language questions over a single table.
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
- Only use the single provided table alias in queries
- DuckDB syntax only - ANSI SQL compatible
- NEVER reference tables or aliases not explicitly provided
- Create correct and executable DuckDB SQL code only

### Natural Language Questions
✓ Business terms (customers, revenue)
✓ Outcome-focused (insights, trends)
✗ No table/column names
✗ No SQL operations mentioned

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

### SQL difficulty levels
- Easy: single-table `SELECT`, basic `WHERE`, optional `ORDER BY`/`LIMIT`, ≤1 aggregate, no subqueries.
- Medium: `GROUP BY`+`HAVING`, multiple aggregates, nested filters, one non-correlated subquery, or `UNION`/`INTERSECT`/`EXCEPT` on the same table.
- Hard: correlated or multi-level subqueries, window functions, complex set ops, `CASE`, CTEs (incl. recursive), or combinations of several advanced features.

### Table involved in the queries
Realize the queries and their natural language counterparts using the following table.
{table}
