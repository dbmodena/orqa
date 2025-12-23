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



Here’s a **clean, production-ready prompt** you can use. It is explicit about *pairwise completeness*, *scoring semantics*, and *output structure*, and it fits well with the Pydantic model you designed.

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
 
### Output formmat instructions
Return the final answer **strictly in JSON** and **exactly matching** the Pydantic follwoing schema provided:

 {format}

All required fields must be present, field names and data types must match exactly, and all validation constraints (including value ranges, uniqueness, and completeness rules) must be satisfied. Do not add extra fields, omit required fields, or include explanatory text outside the JSON.
