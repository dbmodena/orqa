### TABLE-LEVEL ANALYSIS
{table_analysis}

{planning_section}### OUTPUT CARDINALITY (NON-NEGOTIABLE)
Return a `queries` list containing EXACTLY ONE query object — the one
that answers the single question given to you in the Structured Plan
below. Do not generate three queries or any query other than the single
one this plan calls for.

### KEYWORD CONSTRAINTS
- Each table's keywords are its reverse-index retrieval keys from the table analysis: together they identify the table univocally among all portal tables, so copy them faithfully, in full — never dilute them into generic terms, and never truncate the list.
- Keywords help non-expert users understand which tables the query touches.

When generating the final output, include the following fields in each query:
- For each table, description, keywords, and translated_keywords copied from the table analysis
- Ensure the question is phrased as a non-expert user would ask it, keeping the distinctive table-identifying vocabulary (topic, entity type, agency, place, period) intact so a keyword extractor over the question still retrieves the right table.
Note: `difficulty`, `query_plan`, `question_keywords`, `translated_question_keywords`, `translated_question`, `detected_language`, `topic`, and `story` are NOT part of this schema — they are already decided by the query plan above and are attached to your output automatically. Do not attempt to produce them.
Return only valid JSON matching the expected schema.
