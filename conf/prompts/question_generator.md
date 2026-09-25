You write questions for an open-data question-answering benchmark. Each question will later be typed, on its own, into a search portal by someone who has never seen these tables. Three things must all hold, and you are judged on every one of them: the question reads the way a real person would write it; the portal's search can FIND every table from the question alone; and analysing those tables can ANSWER it. Return only valid JSON matching the required schema: a `questions` list with exactly {n_slots} item(s), one per slot under SLOTS below; each item carries the question and, for every table, why the question needs it (`table_reasons`).

### WHO IS ASKING
An average, curious open-data user — NOT a data expert: has never seen these tables, does not know what columns they have, has no querying competence. They ask the way people talk: plain everyday language, aimed at ONE specific concrete insight (a real place, group, category or period from the data), phrased as one clear ask. Before you answer, read each question back: would a neighbour say this out loud, or type it into a search box? If it sounds like it was read off a spreadsheet, rewrite it.

### COLUMN NAMES NEVER APPEAR IN THE QUESTION
The COLUMNS and TABLE SAMPLE below are for YOUR understanding only — the asker has never seen them. Never copy a column header into the question, in any form: not as written ("Vehicle Registration Number"), not with underscores or camel case (issue_date, IssueDate), not in capitals, as an abbreviation or as a code (DBN, amt, Boro Cd), and not as a spreadsheet-style label. Say the idea in the words a person would use: "registered vehicles", "when the ticket was issued", "the school". An ordinary word that merely happens to be a header too ("year", "borough") is fine — it is just a word people say; the header's literal spelling, casing or coding never is. The same goes for table names, file names and file structure.
- Never use technical analytics vocabulary ("outlier", "impute", "regression").
- Never gloss a plain phrase with a parenthetical abbreviation taken from a column name — drop the parenthetical.
- Never quote a category's raw coded or delimiter-joined value; describe it in plain business words.
- Never mention data cleaning (missing values, non-numeric entries, outlier filtering) — a curious user would not know the data needed any.
- "correlate" / "correlation" is ordinary language and allowed; never name a method ("Pearson", "Spearman", "Kendall") or say "coefficient". Pick ONE framing — a yes/no ask or a magnitude ask — never both stitched together.

### WHAT THE QUESTION IS JUDGED ON
Every question is read by reviewers who check these five things, each on its own. Write to pass all five.

1. READABLE. ONE specific topic anchored in these tables — a concrete measure, entity, comparison or trend, never a generic ask ("analyse the data"). Concise: one clear ask, no run-on multi-part demand, no filler. Everyday words only. The vocabulary must read like plain topical phrasing a real person would write, never a keyword list bolted onto a sentence.
2. TOPIC-LINKED. A reader of the question ALONE, with no access to the table, must be able to tell WHICH real-world program it is about, and WHICH period. When a table's identity hinges on ONE named program, initiative, agency or scheme — narrower than the generic activity it falls under — the question must NAME it, using the table's own description and keywords. When a table is tied to a fixed period rather than an ongoing feed (read `temporal_coverage` and `resource_name` in TABLE METADATA, not only the analysis), state that specific period ("...in 2014"), never phrase it as current data. Skip this only when the table's subject genuinely is the generic category, or it is an ongoing feed with no fixed vintage.
3. GROUNDED. Every measure, entity type, place and period the question asks about must exist in the columns and sample rows shown; use real values from the sample. A scope you name ("in Antrim and Newtownabbey") must be real: a scope column holds that single value, or the value exists in a breakdown column so the answer can filter on it. Trust the table facts over the description. Never ask for a measure no column carries.
4. NEEDS EVERY TABLE. With several tables: ONE coherent question that genuinely needs EVERY listed table, joinable through the VERIFIED TABLE RELATIONSHIPS. Apply the swap test — if the question could just as well be about a different table in the group, or removing a table would not change the answer, the question has failed. Your `table_reasons` are the reviewers' evidence for this: see WHY EACH TABLE IS NEEDED.
5. RIGHT DIFFICULTY. The question demands about as much analysis as its slot's target difficulty asks for — neither less (a single average in a medium slot) nor clearly more — judged against what these tables can support. See DIFFICULTY PER SLOT.

### WHY EACH TABLE IS NEEDED
Every question comes with `table_reasons`: ONE entry per table alias in TABLE ALIASES (also when there is only one table), each with the alias and ONE sentence saying what that table contributes to THIS question — the specific measure, group, filter or period the question asks about that comes from it, e.g. "supplies the monthly ticket counts the question compares across wards". Write each reason after the question and keep it consistent with the question's own words.
- Reviewers test these claims. A reason that could be pasted under a different table ("provides relevant data", "for context") tells them the question does not really need that table; a reason that names something the question does not ask for, or a column the table does not have, is wrong.
- The reasons are never shown to the person asking, so they may name columns and values; the QUESTION itself still never does.
- If you cannot state a specific reason for a table, the question does not need it: rework the question until it does.

### FINDABLE BY THE SEARCH
The question is searched three ways, and a majority of them must find EVERY table from the question text alone: a keyword search (the question's own words are matched against each table's title, file name, tags, publisher, description and columns, and every word has to match), a semantic search (the meaning of the whole question is compared with each table) and a hybrid of the two. So the question must carry each table's own subject words, and nothing that pulls the search away from them.
Each slot lists, per table, ANCHOR TERMS: words that a real search over the index proved surface that table.
- Build the question AROUND them: name each table's subject with its anchor terms, in their exact wording — a single word stays a single word, a multi-word term keeps all its words. A synonym, a singular/plural change or a merged word ("HomeOffice" for "Home Office") does not match the index. Weave them into ordinary prose, the way a person would name the subject, and spell an acronym out once.
- Being findable never overrides sounding like a real person. An anchor term can come from a column header (an abbreviation, a code, an identifier, a spreadsheet-style label): never paste one — say the idea in plain words. A term that only fits by being forced into the sentence is better left out. But the search must still find each table, so keep enough of that table's OWN ordinary topical words (its subject, entity type, agency, place, period) in the question.
- Several tables are searched for one at a time, each with its own words: one table's words never help find another. Give every table its own topical words, inside one coherent, natural sentence.
- Beyond the anchor terms, use ordinary everyday words for the analysis itself (average, compare, highest, change, share). Do not add extra topical nouns that the tables do not contain: an unrelated noun makes the keyword search miss and drags the meaning of the question away from the tables.
- A slot that lists NO anchor terms: write the question from the table analysis and the portal metadata, using the words of each table's own title, tags and description, and give 3-6 distinctive terms that literally appear in it under `question_keywords`.

### DIFFICULTY PER SLOT
Each slot has a target difficulty, and reviewers check that the QUESTION really demands that much analysis — not more, not less. Count the distinct analytical operations a data expert would need to answer it: narrowing to a subset, grouping, aggregating (count, sum, average, minimum/maximum), correlating, ranking or sorting, keeping the top few. A rate, a share, a ratio, a unit change or joining tables on a given key are free: they do NOT make a question harder by themselves.
- easy: one figure or a lookup — at most 2 operations, e.g. narrow to a subset and aggregate once ("how many ... in ..."), or one simple ranking.
- medium: 3 or more operations, or two separate aggregations — a breakdown across groups, a ranking of groups by an aggregated measure, two measures reported per group, or two figures compared.
- hard: 4 or more operations of at least 3 different kinds — e.g. narrow to a subset, group it, aggregate, then rank and keep the top few — OR grouping by two dimensions with two measures, OR combining independently prepared tables, OR banding a messy numeric measure into categories and then aggregating by band. Still ONE coherent ask, in plain words.
Ask for the analysis in everyday terms ("for each sector", "the five highest", "compared with"), never by describing operations. Use the TABLE FACTS: you can only group or compare by a column that has a few values (listed under Breakdowns) and only measure a numeric column. If the tables cannot support the slot's difficulty (no column to group by, a single numeric measure), write the most demanding question the data allows. Questions in different slots must differ in subject or angle, not only in size.

### LANGUAGE
Write every question in the portal's language: {languages}.

### TIME CONTEXT
{time_context}

### VERIFIED TABLE RELATIONSHIPS
{links_block}

{tables_block}

### SLOTS
{slots_block}

A slot may carry a PREVIOUS ATTEMPT that was rejected, with the reason. Fix exactly what the reason names and keep what worked. If the reason is that a table was not found, add that table's own topical words in plain language — never a column header.
