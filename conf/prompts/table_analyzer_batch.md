Analyze ALL of the following tables and return a JSON object matching
the provided schema. Return exactly one analysis entry per table, in
the same order the tables are listed, echoing each table's alias
unchanged. Do not include any explanatory text outside the JSON.

IMPORTANT INSTRUCTIONS:
- Produce one analysis object per table under the 'tables' key.
- For each table, write a concise 2-3 sentence description: what the table represents, the real-world context and topic it belongs to, and what information it provides. When the columns, sample values, or metadata reveal a place or a time period, anchor the description with them — e.g. "Students expelled from New York City schools in 2016" rather than "Records about students". Ground every claim in the provided columns, sample rows, and metadata; plain business language, no schema jargon.
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
