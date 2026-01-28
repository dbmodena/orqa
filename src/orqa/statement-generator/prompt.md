## Analyze
Analyze this Dataset and identify columns for UNION and JOIN operations.
### Table details
{table}

### Task Instructions

1. **UNION TASKS** (union_tasks):
   - Identify groups of columns that could be combined with similar columns from other datasets
   - Look for columns with similar names, types, or semantic meaning
   - Each entry: {{"columns": ["col1", "col2"]}}

2. **JOIN TASKS** (join_tasks):
   - Identify columns that could serve as join keys (unique identifiers or foreign keys)
   - Each entry: {{"columns": ["id_col1", "id_col2"]}}

3. **JOIN-CORRELATION TASKS** (join_correlation_tasks):
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

Your goal is to **analyze how well each table matches with every other table**, and assign a **match score from 1 to 10** for **every unordered pair of tables**.

### Input

{table}


### Instructions

1. Identify **all distinct tables** present in the input.
2. Consider **every possible unordered pair of tables**.

   * For **N tables**, you must produce **exactly N·(N−1)/2 scores**.
   * Do **not** omit any pair.
   * Do **not** include self-pairs (e.g., table A with itself).
3. For each table pair, assign a **match score from 1 to 10**, where:

   * **1** = no meaningful relationship
   * **10** = very strong relationship (e.g., clear foreign-key / primary-key join, strong semantic overlap)
4. Base your score on signals such as:

   * Shared or joinable identifiers (e.g., `customer_id`, `order_id`)
   * Semantic similarity of column names
   * Overlap in data meaning or granularity
   * Potential for meaningful joins or correlations
5. Treat table pairs as **unordered**:

   * `(A, B)` is the same as `(B, A)`
   * Provide **only one score per pair**
6. Ensure **completeness and consistency**:

   * Every table must appear in exactly `N−1` pairs
   * No duplicate or missing pairs are allowed






## Pydantic
 
### Output format instructions
Return the final answer **strictly in JSON** and **exactly matching** the Pydantic following schema provided:

 {format}

All required fields must be present, field names and data types must match exactly, and all validation constraints (including value ranges, uniqueness, and completeness rules) must be satisfied. Do not add extra fields, omit required fields, or include explanatory text outside the JSON.










## SQLGeneration

You are an expert data analyst and SQL engineer.

Your task is to generate SQL queries **and** their corresponding natural language questions
based on the provided database schema and **explicit table & column match definitions**.

---

### Table Details
{table}

### Matching Tables to query on
{matches}

> The matches define **the mandatory operation type** and **how tables relate to each other**, including:
> - **The required operation type (MULTI-JOIN, UNION, or JOIN-CORRELATION)** - this is mandatory
> - Join keys and join direction
> - Allowed join paths and correlations
> - Required unions or relationship mappings  
>
> These matches are **authoritative** and must be followed exactly.

---

### IMPORTANT RULES FOR NATURAL LANGUAGE QUESTIONS

Each SQL query MUST have **exactly one** associated natural language question.

The question MUST:
- Describe the business, analytical, or real-world intent
- Be understandable by a non-technical stakeholder
- Avoid mentioning tables, columns, joins, filters, or SQL concepts
- Avoid describing how the query is implemented

The question MUST NOT:
- Refer to tables or columns by name
- Mention SQL operations (JOIN, GROUP BY, UNION, WHERE, etc.)
- Describe schema mechanics or query steps

---

### REQUIRED USE OF TABLE & COLUMN MATCHES (CRITICAL)

**MANDATORY OPERATION ENFORCEMENT:**
- The operation type specified in `{matches}` is **REQUIRED** and **NON-NEGOTIABLE**
- If the match specifies MULTI-JOIN → all queries MUST use multi-table joins as defined
- If the match specifies UNION → all queries MUST use UNION operations as defined
- If the match specifies JOIN-CORRELATION → all queries MUST use correlation joins as defined
- **Queries that do not implement the specified operation type are INVALID**

**Match Compliance Rules:**
- **All joins, unions, and relationships MUST use the provided matches exactly**
- Do **not** infer or invent relationships outside of `{matches}`
- If a match specifies:
  - a join → it must be used with the exact columns specified
  - a union → it must be respected with the exact tables specified
  - a correlation → it must be followed with the exact relationship defined
- Queries that do not rely on the listed matches are **INVALID**

The matches are the **only allowed source of truth** for:
1. **WHAT operation type to use** (MULTI-JOIN, UNION, or JOIN-CORRELATION)
2. **HOW tables connect** (join keys, directions, and conditions)

---

### DuckDB SQL Syntax Requirements

All SQL queries MUST be compatible with DuckDB.

- Use DuckDB-supported ANSI SQL
- CTEs and window functions are allowed if supported by DuckDB
- Use DuckDB-compatible date/time expressions
- Avoid vendor-specific or procedural SQL

---

### Query Generation Instructions

1. Generate **5 SQL queries**, each with a difficulty score from **1 to 5**:
   - **1** = simple filter using the mandatory operation
   - **3** = the mandatory operation with aggregation and multiple conditions
   - **5** = advanced logic using the mandatory operation with CTEs, window functions, or correlated subqueries

2. Each query MUST:
   - **Implement the mandatory operation type specified in `{matches}`**
   - Use **only** the provided tables
   - Use **only** the relationships defined in `{matches}`
   - Use **all** the provided tables
   - Follow DuckDB SQL syntax
   - Correctly answer its associated natural language question

3. The natural language question and SQL query must be:
   - Semantically aligned
   - Not a trivial restatement of each other

4. Output MUST strictly conform to the provided **Pydantic schema**
   - No extra text
   - No explanations
   - No markdown outside the schema

---

### Goal

Produce realistic, business-relevant analytical questions and their **correct, match-compliant**
DuckDB SQL implementations that use the **mandatory operation type** specified in the matches.
