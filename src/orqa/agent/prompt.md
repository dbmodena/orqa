## Analyze
Analyze this CSV file and identify columns for UNION and JOIN operations.
            CSV Information:
            - Filename: {filename}
            - Rows: {numrows}
            - Columns: {numcolumns}
            - Notes: {notes}

            Column Details: {coldetails}
Sample Data (first 5 rows):
    {sample}

Task Instructions:

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

COLUMN SELECTION GUIDELINES:
- **INCLUDE**: Columns with semantic meaning (user_id, product_id, name, email, amount, date, etc.)
- **EXCLUDE**: Sequential index columns that are just row numbers (1, 2, 3, ..., N) with no business meaning
- Use your judgment: If a column only serves as a row counter and provides no informational value for data operations, skip it
- Meaningful identifiers (even if numeric) should be included as they enable joins

CRITICAL RULES:
1. Your response must be a SINGLE JSON OBJECT (not an array)
2. The object must have exactly these 3 fields: "union_tasks", "join_tasks", "join_correlation_tasks"
3. Each field is an ARRAY of objects
4. DO NOT include a "task" field in any object
5. DO NOT wrap your response in an array
6. Column names must exactly match those in the CSV

