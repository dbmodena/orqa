# Benchmark solver — Phase 2: table selection

You are answering an open-data question independently, without being told which table it belongs to. A reverse-index search over the question's own keywords retrieved the CANDIDATE tables below — **not every one of them is necessarily relevant**; retrieval is imperfect and some are false positives that merely share vocabulary with the question. Decide which candidate(s), and which of their columns, actually answer the question.

### Instructions

- Read the question, then each candidate's schema/samples/column stats below. A candidate is usable only if it genuinely contains the data the question asks about — matching on topic vocabulary alone is not enough.
- You may select ONE table, or a COMBINATION of tables — but only combine tables the relationships block below actually supports (a real join key, a real union alignment, or a real correlation), never two tables that merely seem thematically related. Never invent a relationship the evidence doesn't show.
- List every column, from every table you select, that the answer actually depends on — filters, join keys, and whatever gets aggregated or returned. Omit columns you won't use.
- Set `expected_result_type` from the question's own shape — a question asking for one figure ("how many/how much...") is `number`; yes/no is `boolean`; "which single X..." is `text`; one ordered sequence of values is `list`; per-group breakdowns, rankings, or any multi-column result is `table`.
- If NONE of the candidates can actually answer the question, set `no_viable_selection: true` and leave `tables` empty — a forced, unsupported guess is worse than admitting retrieval missed.

### Question
{question}

### Candidate tables
{candidates}

### Relationships between candidates
Only pairs with real evidence appear here; a candidate absent from every pair has no supported relationship to any other candidate in this list.
{relationships}
