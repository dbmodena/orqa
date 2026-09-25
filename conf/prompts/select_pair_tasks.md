# Analysis and Work Description
You are given TWO concrete datasets (Q and R) that an automated discovery step has flagged as related, together with the evidence for that relation (metadata similarity and schema-matching column pairs). Select which discovery operations genuinely make sense between them, and exactly which columns on BOTH sides.

### Task Instructions:

1. **UNION TASKS** (union_tasks):
   - Propose a union only when Q and R describe the same kind of records (same entity type, compatible granularity).
   - Format each entry as: {{"q_columns": ["q_col1", "q_col2"], "r_columns": ["r_col1", "r_col2"]}}.
   - The lists are ALIGNED: q_columns[i] and r_columns[i] must carry the same semantic meaning and compatible types. Both lists must have the same length.
   - Prefer a WIDE set of aligned columns; do not propose unions covering only one or two columns unless the tables are that narrow.

2. **JOIN TASKS** (join_tasks):
   - Propose a join only when Q and R share key columns whose VALUES plausibly overlap (identifiers, codes, names of the same real-world entities — not merely similar column names).
   - Format each entry as: {{"q_columns": ["q_key"], "r_columns": ["r_key"]}}.
   - The lists are ALIGNED join keys: q_columns[i] joins r_columns[i]. Both lists must have the same length.
   - Ignore columns with very few distinct values; focus on columns that are likely keys.

3. **JOIN-CORRELATION TASKS** (join_correlation_tasks):
   - Propose only when, AFTER joining Q and R on a key, correlating a numeric column of Q with a numeric column of R would be analytically meaningful.
   - Each entry: {{"q_key": "q_id", "r_key": "r_id", "q_target": "q_metric", "r_target": "r_metric"}}.
   - q_target and r_target MUST be numeric columns (amounts, counts, scores, rates). Never categorical or boolean columns.

### Notes

- Propose AT MOST 3 tasks per task type.
- It is CORRECT to return empty lists when an operation does not genuinely apply — a topically similar pair is not automatically unionable or joinable.
- Use the schema-matching evidence as a hint, not an obligation: a matched column pair is only worth using when the semantics truly line up.

### Critical Rules:

1. Your response must be a single JSON object (not an array).
2. The object must have exactly these 3 fields: "union_tasks", "join_tasks", "join_correlation_tasks".
3. Each field is an ARRAY of objects.
4. Column names must exactly match those of the respective dataset: q_-prefixed fields use Q's columns, r_-prefixed fields use R's columns.
5. DO NOT wrap your response in an array.

---

## Dataset Q

{q_dataset_description}

## Dataset R

{r_dataset_description}

## Relation evidence

- Metadata cosine similarity: {cosine_similarity}
- Schema-matching column pairs (q_col <-> r_col (score)):
{schema_matches}
