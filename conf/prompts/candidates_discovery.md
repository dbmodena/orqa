# Analyze
Analyze the given dataset and identify columns for UNION, JOIN and JOIN-CORRELATION discovery operations over a data lake. 

{query_dataset_description}

### Task Instructions:

1. **UNION TASKS** (union_tasks):
   - Identify groups of columns that could be combined with similar columns from other datasets
   - Look for columns with similar names, types, or semantic meaning
   - Each entry: {{"columns": ["col1", "col2", "col3", "col4"]}}

2. **JOIN TASKS** (join_tasks):
   - Identify columns that could serve as join keys (unique identifiers or foreign keys)
   - Each entry: {{"columns": ["id_col1", "id_col2"]}}

3. **JOIN-CORRELATION TASKS** (join_correlation_tasks):
   - Identify pairs where one column is a join key and another correlates with it
   - Each entry: {{"join_column": "id_col", "correlation_column": "metric_col"}}
   - Examples: user_id with total_purchases, product_id with price
   - The correlation column MUST be numeric (e.g., amount, price, quantity, score, balance).
     Do not consider categorical columns (e.g. with boolean "yes/no", with categoricals "small/medium/large", ...)

### Notes

1. Do not focus on correctness of numerical values: for instance, 6,124.45 is as correct as 6124.45
2. For Union tasks is better to consider wider set of columns.
3. For Join tasks, ignore those columns with few unique value (<3).
4. Propose **at most 5** tasks for each task type.

### Critical Rules:

1. Your response must be a single JSON object (not an array)
2. The object must have exactly these 3 fields: "union_tasks", "join_tasks", "join_correlation_tasks"
3. Each field is an ARRAY of objects
4. DO NOT include a "task" field in any object
5. DO NOT wrap your response in an array
6. Column names must exactly match those in the CSV
