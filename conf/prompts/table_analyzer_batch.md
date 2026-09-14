Analyze ALL of the following tables and return a JSON object matching
the provided schema. Return exactly one analysis entry per table, in
the same order the tables are listed, echoing each table's alias
unchanged. Do not include any explanatory text outside the JSON.

IMPORTANT INSTRUCTIONS:
- Produce one analysis object per table under the 'tables' key.
- For each table, write a description that identifies the table precisely instead of summarising its topic. Open with one sentence naming what is counted or listed, for which population or category, at which granularity, where, and for which period — e.g. "Number of student removals and suspensions by school district in New York City public schools during school year 2016-2017" rather than "This table contains data about student discipline". Name the responsible agency or organisation when it tells the table apart from similar tables. At most one more sentence may summarise the main information the columns provide. Plain business language, no schema jargon.
- Ground every element of the description in the provided columns, sample rows, and metadata, and never invent one: when no period or place can be established, leave it out. Each table's metadata carries only the fields that identify it — `title`, `description`, `publisher`, `responsible_entity` (a body other than the publisher that produces or is responsible for the data), `temporal_coverage` (the period the portal declares the data covers) and `tags`. Take the period from `temporal_coverage`, the title or description, or the sample values; when they disagree, trust the title and the sample values.
- For each table, extract up to 10 keywords (max). These keywords are indexed in a reverse (inverted) index over MANY tables from the same portal and are the ONLY retrieval signal for this table: it is their COMBINATION that must identify the table UNIVOCALLY — a search matching several of them together should return this table and no other.
- Each keyword is a single word or a short established term (a proper name like "New York" or a fixed compound like "traffic collision" — never a descriptive phrase or mini-sentence). Do not pack the discriminating context into one long keyword; spread it across separate keywords whose intersection is unique (e.g. "NYPD" + "training" + "events", or "expulsions" + "schools" + "2016" — not "NYPD training events log").
- Make the keyword SET discriminative, not merely descriptive. Include the table's specific qualifiers as their own keywords — the entity type it records, the agency/organisation or program involved, place names, the population or category covered, and the time period — and avoid wasting slots on generic portal-wide terms that many tables share ("data", "records", "city", "annual", "report").
- Keywords must remain natural search terms a real user would type, grounded in the provided columns, sample values, and metadata — never invent a term the table's content does not support.
- Tables in this batch (and portal) can be topically close. When a keyword could plausibly describe a sibling table (same topic, different year/borough/agency/granularity), add the qualifier that tells this table apart instead of leaving the ambiguous term alone.
- Analyze every table independently; do not merge or skip tables.

Aliases (in order): {aliases}
Detected languages: {languages}

Tables:
{tables}
