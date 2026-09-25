{task_statement}PLAN REQUIREMENTS (apply to every plan):
- `steps`: an ordered list; each step has an `op`, a `description`, the `tables` (aliases) it touches, and the concrete `columns` it reads or writes.
- Use only the provided table aliases and columns that exist in those tables. Any OTHER column your plan needs must be declared as a derived column (below) — a column name that exists in no table and is declared nowhere fails validation before any judge sees the plan.

{ops_statement}- Also provide `question`, `question_keywords`, `plan_keywords`, `query_plan` (a short natural-language description of how the steps build the answer), `topic` (one short business theme), and `story` (a short business narrative behind the plan).
- `expected_result_type`: the SHAPE of the final answer — exactly one of `number` ("how many/how much"), `boolean` (yes/no), `text` ("which single X"), `list` (one ordered sequence), or `table` (per-group breakdowns, rankings, any multi-column result). Take it from the question: a question asking for one figure must not promise a table. This is mechanically enforced against the executed result.
- `expected_result_description`: one or two sentences concretely describing that result — what the value(s) represent, their unit/granularity, and for `table`/`list` what each row and column holds (e.g. "one row per borough with the total permits issued there in 2023, sorted descending"). The code generator shapes its final result from this.
- Together these two must be the NATURAL CONCLUSION of your steps — what the plan builds toward, not a shape chosen first and bolted on — and they must ACCOUNT FOR EVERY ANALYSIS the plan performs. If your steps compute two figures, the declared result has to hold both: fold them together (a `join`/`union` on a shared key, a correlation, a comparison or ranking, or a final step reporting on both — no mechanism is mandatory), or declare a result that honestly carries both (e.g. a two-row table). A plan that computes something the declared result never accounts for is rejected — the fix is to fold it in, widen the declaration, or drop the step that earns nothing, never to drop a branch.
- `tables`: ONE entry per table, covering EVERY alias in TABLE ALIASES exactly once — no more, no fewer (an omitted or invented alias fails validation before any judge sees the plan). Each entry needs:
  - `name`: the exact alias from TABLE ALIASES.
  - `reason`: 2–3 sentences, not a one-liner, covering (1) the table's concrete ROLE — which step(s) it feeds and how, (2) the SPECIFIC rows/columns/filtering the answer depends on it for, named explicitly, and (3) why the question could not be answered without it. A judge panel rejects vague, generic, or unfounded justifications, so shape the `question` so every table is genuinely necessary and write a justification specific enough it could NOT be pasted onto a different table and still sound plausible. "For context", "for completeness", "to enrich the analysis" are never justifications.
  - `columns_involved`: the minimal columns from that table this plan's steps actually use.
  - `description`/`keywords`/`translated_keywords`: copy from the matching entry in TABLE-LEVEL ANALYSIS (leave `translated_keywords` empty if none given).
- `detected_language`: the dominant language from the table analyses / DETECTED LANGUAGES below. Also provide `translated_question` and `translated_question_keywords` (identical to the originals when already in that language).

### DERIVED COLUMNS
A step often creates a column no table contains: an aggregate's result, a flag or bucket, a rank position, a renamed column. Declare each one in that step's `produces`, with what it was computed from:
- `produces`: `[{{"name": <new column>, "operation": <short label: "sum", "mean", "count", "flag", "bucket", "ratio", "year_extract", "rename", ...>, "sources": [<column it comes from>, ...]}}]`
- Write a source that is a real column of a table as `Alias.column` (e.g. `Table_0.amount`). Write a source an EARLIER step derived as its bare name (e.g. `total_spend`) — that column may itself come from several tables, so qualifying it would be wrong. Never guess an alias.
- `sources` may be empty ONLY for an operation that genuinely reads no column (`count`, `size`, `row_number`, `rank`, `literal`, `constant`). Every other operation must say what it was computed from.
- Wherever a step also declares an output in `params` (`output_column`, `new_column`, or the keys of `aggregations`), declare it in `produces` too — `params` carries the shape each op needs, `produces` carries the origin.

DECLARE ONCE. A derived column is declared only in the step that creates it. From that step onward it behaves exactly like a real column of the data: any later step may reference it by name in `columns` or `params` for any purpose — filtering, grouping, aggregating, sorting, ranking, joining, correlating, or as the origin of a further derived column. Never re-declare it in a later step, and never declare a name that a table already has.

### THE QUESTION IS GIVEN
Each plan's `question` is fixed by the FIXED QUESTIONS section — do not write or reword one. Everything a question must do (plain everyday words, the tables' distinctive vocabulary, a fixed-period scope, naming the specific program) was settled and checked before you were called. Your job is the plan: the steps, each table's role, the result declaration and the remaining metadata fields.

### DATA QUALITY / CLEANING
Tables are shown RAW — no bad-token conversion, numeric coercion, or null-row dropping. Beyond the usual `dtype`/`cardinality`/`top_values`, each column in COLUMN STATISTICS carries:
- `nan_count` / `null_ratio`: true missing values (empty cells, pandas' default NA tokens) — unambiguous, no decision needed.
- `bad_token_counts`: counts of this portal's known missing-value LITERALS (`"n/a"`, `"not available"`, `"(null)"`) still present as plain strings — NOT yet NaN, and visible in TABLE SAMPLE exactly as stored.
- `numeric_parseable_ratio`: for a text column, the fraction of values that would parse as a number once stripped of formatting (`"1,314"`, `"34.10%"`, `"$500"`). Near 1.0 flags a column that is numeric-in-disguise.
- `numeric_outliers`: values statistically far from the column's OWN spread (Tukey IQR fencing), with counts and examples per side.
- `numeric_pinned_extreme`: a value AT the column's min or max repeating far more than its spread predicts (hundreds of rows reading exactly `90` in an age column, exactly `0` in a dollar column) — the top-/bottom-coding signature in already-numeric data. Contrast `numeric_outliers`, which flags a RARE extreme; both can legitimately fire for the same value.
- `minority_value_groups`: rare/tail text values beyond `top_values`, grouped into structural shapes with counts and examples (values shaped `<#` might group `"<18"` and `"<16"`; a shape can also be a literal word repeated rarely).

**None of these five stats tell you whether what they found is noise or signal — that judgment is yours.** For every non-trivial entry on a column your plan uses, ask what it MEANS, not how rare or oddly-shaped it is:
- An explicit non-answer/placeholder ("no data recorded", even if absent from `bad_token_counts`) or a sentinel (a lone `-1`/`999`/`9999` far outside the distribution) → a `clean` step: `impute`/`drop_rows`/`drop_column`/`cast`. If not already in `bad_token_counts`, name it in that action's `treat_as_missing` list so the code generator treats it as missing too.
- A genuine, informative value — a censoring/top-coding convention (a bound recorded instead of a number), a qualitative label standing in for a range, or a rare-but-real extreme → discarding it is a loss. Add a `derive` step that PRESERVES it as its own queryable feature (`technique: "flag"` or `"bucket"`). This is often where the most interesting questions come from — the feature can legitimately be what the `question` is about.
- Rarity alone is never reason to clean something away, and a shape is never automatically dirty. An age column recording minors as `"<18"` is censored data, not corrupted — the move is an `is_minor` flag, not a `clean` step turning it into NaN. `numeric_pinned_extreme` flagging hundreds of exact `90`s is the SAME kind of genuine recorded bound, just already numeric — again a `flag`/`bucket`, not a silent discard. Treat any of it as noise only with a CONCRETE reason; repetition at a boundary is exactly what the stat measures, not evidence of error.
- **DANGER — `numeric_outliers` feeding a `sum`/"total":** never drop a numerically extreme value from a column a later step SUMS unless you have a CONCRETE reason it is wrong (impossible for the domain, a formatting artifact, a duplicated-digit typo). The Tukey fence alone is never that reason — it flags rarity, not error. A real, unusually large payment/grant is exactly what a total must include; dropping it understates the true total, which is a wrong answer, not a cleaner one. Round, plausible numbers within the column's own unit (grants of $400k–$600k where typical is $100k) are genuine — keep them.
- **DANGER — `numeric_pinned_extreme` feeding a `mean`/`sum`:** the mirror failure. Silently including a capped value with no `flag`/`bucket` step marking which rows were capped misrepresents the result as if it reflected the true unbounded distribution (a pinned ceiling biases a mean down, a pinned floor up). Either preserve the cap as its own feature, or — when the question is honestly about the capped population as recorded — say so explicitly in `expected_result_description`.
- **DANGER — `correlate` against a column with no variance in this plan's scope:** if one correlated column is CONSTANT across the plan's grouping, the coefficient is mathematically UNDEFINED (NaN), not weak, and the query is rejected. The common trap: deriving "average year" from a table already scoped to a single fixed period — every row shares that year, so there is zero spread. Before adding a `correlate` step, check BOTH columns can plausibly vary across the grouping; if a fixed scope guarantees one side is constant, correlate against something that varies, or drop the step for a comparison that doesn't need variance on both sides (a ratio, or the two figures side by side).

A `clean` step's `params` follows the `{{"actions": [...]}}` convention on the step schema; cite the exact literal tokens from the stats above in the step's `description`, never invented ones. If a `clean` step touches a column a later `join`/`union` relies on, place it BEFORE that step.

### DIFFICULTY IS GIVEN
Each plan's `difficulty` is fixed by its question (see FIXED QUESTIONS): it was judged on the question before you were called. Copy it into `difficulty` and nothing more — never shape, pad or trim the steps to reach a tier. Plan the analysis the question naturally needs, in as many steps as it takes.

### COMBINING METRICS ACROSS TABLES
When a plan blends figures from more than one table into a single output value (a `derive`/`aggregate` producing a "combined total" or blended score), the combination must be substantively meaningful. This does NOT apply to `correlate`: a coefficient is scale-invariant (normalized to [-1, 1] regardless of units), so correlating a COUNT against a dollar amount is sound and needs no unit reconciliation.
- NEVER sum, average, or blend raw values on incommensurate units/scales into one additive total (a COUNT of records + a SUM of an area + a SUM of a dollar amount) — the largest-magnitude term silently dominates. Check COLUMN STATISTICS: if magnitudes differ by orders of magnitude, or the business definition differs (count vs. measurement vs. money), report them as separate columns, combine via a dimensionally-sound rate/ratio/index ("complaints per action", "actions per square mile"), or normalize (z-score/min-max) if one blended index is genuinely the point.
- Combining figures over DIFFERENT periods isn't inherently wrong — a plan MAY compare, rank, or compute a ratio/trend BETWEEN periods when that comparison IS the insight. Not allowed: silently summing values from different periods into one undifferentiated total. Keep them as distinct, separately-labeled figures (or an explicit ratio/delta) in the steps and `expected_result_description`.
- Same-unit does NOT mean summable: plain counts from conceptually unrelated administrative processes (building-permit + parking-ticket + tree-planting counts) are as meaningless added as mismatched units. Ask whether the sum is one quantity a domain expert would recognize and name ("total complaints" across sub-categories of the SAME register is fine). A generic thematic label slapped on afterward ("combined civic activity") is a rationalization, not a shared referent.
- Sanity-check against a concrete row: if one term can be near-zero while the combined value barely changes because another dominates, the combination isn't meaningful — present the components separately or as a dimensionally-valid ratio.

{batch_note}{fixed_questions}### TIME CONTEXT
{time_context}

### DETECTED LANGUAGES
{detected_languages}

### VERIFIED TABLE RELATIONSHIPS
The relationships below are the ONLY verified ways these tables can be combined. Every cross-table step must use one of them exactly as specified — same tables, same key columns, same operation type (join / union / join+correlation). Never invent one that is not listed.
How you COMPOSE them is yours to design — they are building blocks, not a prescribed pipeline. A sequential chain (A⋈B, then ⋈C) is one valid shape, never the required one; you are equally free to build INDEPENDENT branches and then combine or compare their results. You do not have to use every listed relationship — skip one when a more interesting composition emerges without it, as long as every provided table is still genuinely used and justified, and tables are only ever combined through the listed relationships. Pick whichever composition yields the most natural, insightful question over all the tables.

A verified relationship is also EVIDENCE, not only a recipe. Its key columns tell you these tables are cut along the SAME dimensions and therefore describe comparable populations — that is knowledge about what is jointly ASKABLE. Use it that way: it can justify a question that compares or contrasts the tables even when the plan combines them in one cheap step, or brings a table in through a single lightweight `select`/`filter` rather than a join. The constraint above still holds — cross-table STEPS may only use listed relationships — but what you learn FROM a relationship is free to shape the question.
{table_links}

### TABLE ALIASES
{table_aliases}

### TABLE-LEVEL ANALYSIS
{table_analysis}

### TABLE METADATA (from the open-data portal)
Each table's own entry in the portal, keyed by alias. `title` is shared by every file of a dataset, while `resource_name` and `resource_description` describe this particular file and often carry its snapshot date or edition; `temporal_coverage` is the period the portal declares. Take a table's period, vintage and program from here when the analysis leaves them out or disagrees with it. Metadata describes the whole file, never a subset of its rows: a question narrowed to a place, unit or category that is only one of a column's values still needs a step filtering on it (see COLUMN STATISTICS).
{table_metadata}

### TABLE SAMPLE (real rows, up to 10 per table)
Ground every question in these actual observed values — especially a hypothetical scenario's concrete inputs (a real grade level, a real program type seen below) — never invent a value that doesn't plausibly come from this data.
{table_sample}

### COLUMN STATISTICS
{column_statistics}
