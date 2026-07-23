{task_statement}PLAN REQUIREMENTS (apply to every plan):
- Provide `steps`: an ordered list where each step has an `op`, a `description`, the `tables` (aliases) it touches, and the concrete `columns` it reads or writes.
- Use only the provided table aliases and columns that exist in those tables.
{ops_statement}- Also provide a business `question`, `question_keywords` (max 10), and `plan_keywords` (max 10).
- Provide `expected_result_type`: the SHAPE of the final answer the code must produce — exactly one of `number` ("how many/how much..."), `boolean` (a yes/no question), `text` ("which single X..."), `list` (one ordered sequence of values), or `table` (per-group breakdowns, rankings, or any multi-column result). Pick it FROM the question: a question asking for one figure must not promise a table, and vice versa. This is enforced — the executed code's result is mechanically checked against it.
- Provide `expected_result_description`: one or two sentences concretely describing the expected result — what the value(s) represent, their unit/granularity, and for `table`/`list` what each row and column holds (e.g. "one row per borough with the total number of permits issued there in 2023, sorted descending"). The code generator shapes its final result from this, and the plan reviewers check it for consistency with the question.
- Provide `tables`: ONE entry per table, covering EVERY alias provided in TABLE ALIASES exactly once — no more, no fewer (an omitted or invented alias fails validation immediately, before any judge even sees the plan). Each entry needs:
  - `name`: the exact alias, matching TABLE ALIASES.
  - `reason`: a well-articulated, motivated justification (2–3 sentences, not a generic one-liner) explaining (1) the table's concrete ROLE in this plan — which step(s) it feeds and how, (2) the SPECIFIC rows, columns, or filtering the answer actually depends on it for, named explicitly rather than asserted in the abstract, and (3) why the question could not be answered without it. A judge panel reviews these before any code is written and rejects the plan if any table's justification is vague, generic, or does not hold — so shape the `question` so that every table is genuinely necessary to answer it, and write a justification specific enough that it could NOT be copy-pasted onto a different, unrelated table and still sound plausible. Never bolt a table on "for context", "for completeness", or "to enrich the analysis" — these are not justifications.
  - `columns_involved`: the minimal columns from that table this plan's steps actually use.
  - `description`/`keywords`/`translated_keywords`: copy them from the matching entry in TABLE-LEVEL ANALYSIS below (leave `translated_keywords` empty if none is given).
- Write every `question` impersonating an average, curious user of open data — NOT a data expert: someone who does not know what data exists underneath, has never seen a table, and has no competence in querying one. Conversational, plain everyday language, aimed at one specific, concrete insight (a real place, group, category, or period from the data) rather than a generic exploration. Never mention table names, column names, or file structure, and never use analytics/ML vocabulary (e.g. "regression", "classify", "forecast model", "correlation", "held-out", "confounder", "impute") in the question text — describe what they want to know, not the method.
- The question must also be RETRIEVABLE: downstream, a keyword extractor runs over the question text ALONE to look the right table(s) up in a reverse index built from each table's analysis keywords. Weave the distinctive subject vocabulary that identifies the table(s) — its topic, entity type, agency/organisation, place, and period, as reflected in the TABLE-LEVEL ANALYSIS keywords — naturally into the question, so the keywords extracted from it match the correct table's index entry and no other's. This never conflicts with the naive-user persona: the signal must be plain topical words ("NYPD training sessions in Brooklyn"), never data-speak ("the training events table").
- `question_keywords` are exactly those retrieval terms: single words or short established terms (never descriptive phrases) that actually appear in (or are directly implied by) the question text, whose COMBINATION singles the right table(s) out — not generic filler words a keyword extractor would discard.
- Provide `query_plan`: a short natural-language description of how the steps above build the answer to `question`.
- Provide `topic`: one short business topic/theme summarizing this plan's primary analytical concern.
- Provide `story`: a short business narrative describing the insight or storyline behind this plan.
- Detect the dominant language from the table analyses/detected languages below and set `detected_language` to it.
- Provide `translated_question` (the `question` translated into `detected_language`; identical to `question` when it is already in that language) and `translated_question_keywords` (the `question_keywords` translated the same way, max 10).

{batch_note}### TIME CONTEXT
{time_context}

### VERIFIED TABLE RELATIONSHIPS
The relationships below are the ONLY verified ways these tables can be combined. Every cross-table step in your plan must use one of them exactly as specified — same tables, same key columns, same operation type (join / union / join+correlation). Never invent a relationship that is not listed.
How you COMPOSE them is yours to design — they are building blocks, not a prescribed pipeline:
- A sequential chain (A⋈B, then ⋈C, then ⋈D) is one valid shape, but never the required one.
- You are equally free to build INDEPENDENT branches and then combine or compare their results — e.g. join Table_0 with Table_1, separately join Table_2 with Table_3, and compare the two aggregates side by side.
- You do not have to use every listed relationship; skip one when a more interesting composition emerges without it. Every provided table must still be genuinely used and justified, and tables may only ever be combined through the listed relationships.
- Pick whichever composition yields the most natural, insightful question over all the tables.
{table_links}

### TABLE ALIASES
{table_aliases}

### TABLE-LEVEL ANALYSIS
{table_analysis}

### TABLE SAMPLE (real rows, up to 10 per table)
Ground every question in these actual observed values — especially a hypothetical
scenario's concrete predictor values (e.g. a real grade level, a real program type
seen below) — never invent a value that doesn't plausibly come from this data.
{table_sample}

### COLUMN STATISTICS
{column_statistics}

### DETECTED LANGUAGES
{detected_languages}
