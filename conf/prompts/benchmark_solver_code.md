# Benchmark solver — Phase 3: write the code

Table selection is already decided (Phase 2, below) — your only job here is to write **{kind}** code that answers the question using exactly those tables and columns. Do not second-guess or change the selection; do not reference any table other than the ones listed.

### Data quality — tables are RAW
Loaded exactly as stored: no bad-token→missing conversion, no numeric coercion, no null-row dropping has been applied upstream. If a column you use looks numeric but is stored as text, or shows a sentinel-looking value (`"n/a"`, `"not available"`, `-1`, `999`, …), handle it defensively rather than assuming it is already clean.

{kind_rules}

### Output
- `code`: the executable {kind} code. End in whatever this language's convention is for "the answer" (see the rules above).

### Question
{question}

### Expected result shape
{expected_result_type}

### Selected tables
{selected_tables}
