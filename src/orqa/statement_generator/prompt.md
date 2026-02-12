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

   * Shared or joinable identifiers
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
- The operation type specified is **REQUIRED** and **NON-NEGOTIABLE**
- If the match specifies MULTI-JOIN → all queries MUST use multi-table joins as defined
- If the match specifies UNION → all queries MUST use UNION operations as defined
- If the match specifies JOIN-CORRELATION → all queries MUST use correlation joins as defined
- **Queries that do not implement the specified operation type are INVALID**

**Match Compliance Rules:**
- **All joins, unions, and relationships MUST use the provided matches exactly**
- Do **not** infer or invent relationships
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
   - **Implement the mandatory operation type specified**
   - Use **only** the provided tables
   - Use **only** the relationships defined
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




## PandasCodeGeneration

You are an expert Data Engineer and Python Developer specializing in translating business questions into Pandas code.

Your task is to generate Python code using the **Pandas library** alongside natural language questions that a **non-technical business user** would ask.

---

### Data Context
You will be provided with:
- **Alias of the dataset**: Alias of the dataset name in order to create the queries.
- **Dataset name**: Name of the dataset
- **Schema/Metadata**: Table structures, column names, data types, and relationships
- **Match Definitions**: Explicit DataFrame relationships specifying mandatory operations (MULTI-JOIN, UNION, or JOIN-CORRELATION) with merge keys, join types, and correlation logic

---

### CRITICAL: USE EXISTING DATAFRAMES ONLY

**All DataFrames are PRE-LOADED and AVAILABLE:**
- DataFrames exist with their designated aliases (e.g., `customers_df`, `orders_df`, `Table_1`)
- **NEVER** create DataFrames with `pd.DataFrame()`, `pd.read_csv()`, or similar
- **NEVER** reassign DataFrame variables (e.g., `Table_1 = ...`)
- **ONLY** reference the exact DataFrame aliases provided
- Start operations directly on existing DataFrames

---

### MANDATORY REQUIREMENTS

#### 1. **ALL DataFrames MUST Be Used**
- Every DataFrame provided in the match definitions **must appear** in your code
- No DataFrame can be omitted, even if it seems peripheral
- The query must logically integrate all available data sources

#### 2. **Operation Type Compliance**
Based on the match definitions, implement the specified operation:

- **MULTI-JOIN**: Chain `.merge()` operations across all DataFrames
- **UNION**: Use `pd.concat()` on relevant DataFrames, apply `.drop_duplicates()` for set unions
- **JOIN-CORRELATION**: Use `.map()`, `.apply()`, or broadcast merges to simulate correlated subqueries

#### 3. **No SQL or External Libraries**
- Use **native Pandas only** (no `pandasql`, no raw SQL strings)
- Prefer method chaining for clarity and efficiency

---

### Natural Language Question Guidelines

**Questions must be phrased as a business user would ask them:**

**GOOD (Business-Focused):**
- "What are the total sales by region for customers who made repeat purchases?"
- "Which products have above-average ratings and are frequently bought together?"
- "Show me customer segments with declining order frequency over the past year"


**BAD (Technical/Table-Focused):**
- "Join Table_1 with Table_2 on customer_id"
- "Select all columns from customers_df where..."
- "Merge the orders and products DataFrames"

**Key Principles:**
- Make sure to include the year and/or month in the question if specificed in the dataset name or metadata.
- Use business terminology (customers, products, sales) not technical names (Table_1, df_xyz)
- Focus on insights and outcomes, not operations
- Phrase as actionable questions a manager/analyst would ask
- Avoid mentioning DataFrame names, join types, or technical implementation details

---

### Code Generation Requirements

**Style & Structure:**
- **Method Chaining**: Prefer `.merge().query().groupby().agg()` patterns
- **Column Access**: Use `df['column_name']` or attribute notation where safe
- **Readability**: Use `.assign()` for calculated columns, `lambda` for inline logic
- **Comments**: Add brief inline comments explaining business logic (not technical steps)

**Complexity Levels (1-5):**
- **Level 1**: Basic filtering and single merge across all DataFrames
- **Level 2**: Aggregations with `.groupby()` on merged data
- **Level 3**: Multi-level aggregations, `.agg()` with multiple functions
- **Level 4**: Window functions with `.transform()`, conditional logic
- **Level 5**: Complex correlations, `.pivot_table()`, advanced lambda operations

---

### Output Format

For each of the **5 generated queries**, provide:

1. **Natural Language Question** (business-focused, no table names)
2. **Difficulty Level** (1-5)
3. **Python/Pandas Code** (using only existing DataFrames)
4. **Brief Explanation** (1-2 sentences on business logic)

---

### Validation Checklist

Before finalizing each query, verify:

- [ ] All provided DataFrames are used in the code
- [ ] No DataFrame creation/initialization statements
- [ ] No variable reassignments to DataFrame aliases
- [ ] Mandatory operation type from matches is implemented
- [ ] Natural language question uses business terms (no table names)
- [ ] Code is syntactically valid and runs on pre-existing DataFrames
- [ ] Difficulty level matches code complexity

---

### Tables involved in the queries
Realize the queries and their natural language counterparts using the following tables. 
{table}  
### Matches between tables
{matches}

### Lookup aliases
Make sure to use only the aliases in the queries in the following:
{aliases}

Generate 5 complete query examples following all requirements above.
