## PandasCodeGeneration
Generate **exactly ONE** Python Pandas query implementing the single business question below — not a new question you invent. Return a `queries` list containing exactly one query object. This call is one of several independent generation calls, each bound to its own plan; you answer only the plan attached to THIS call.

Below this prompt you will see the **Inputs** (detected languages, DataFrame aliases, table schemas, verified relationships), the **Column Statistics**, and the **Structured Plan** — the ordered decomposition steps for THIS query, including the `question` you must answer.

### Question rule — use the plan's question, don't invent one
- The plan's `question` is the exact business question this query must answer. Copy its intent faithfully; you may only lightly polish phrasing (grammar, fluency), never its topic, scope, or the metrics/filters it implies.
- It is already written as an average, non-expert user would ask it — conversational, no knowledge of table or column names. Preserve that framing.
- It deliberately carries the distinctive subject vocabulary (topic, entity type, agency, place, period) a downstream keyword extractor uses to retrieve the right table from a reverse index. Any polish must keep those identifying terms intact — never paraphrase them into generic wording.
- ✗ Never mention table names, column names, DataFrame names, or any Pandas/Python operation in the question.

### Rename-first rule (NON-NEGOTIABLE)
When the Structured Plan lists a Step 0 with `.rename()` calls, emit them at the top of your query — before any merge, filter, or aggregation — and use the renamed names everywhere after. Never reference an original name after renaming.

### Query rules
- All DataFrames are **pre-loaded** under their aliases — start operations directly on them. Never use `pd.DataFrame()`, `pd.read_csv()`, or reassign an alias variable (renaming into a NEW variable is fine, and required when Step 0 is present).
- Follow the Structured Plan's steps, in order — they are the validated decomposition of this exact question. Generate plain Pandas only (no external APIs or ML frameworks), and always executable against the provided DataFrames.
- For every relationship you use from **Verified table relationships**, its join type (INNER, LEFT, ...), key columns, and case-normalisation are NON-NEGOTIABLE. When a join is marked `Case-insensitive keys: YES` you MUST `.str.lower()` both sides before joining; omitting this is a hard error.
- The relationships are building blocks, not a prescribed chain: compose them in whatever shape the plan dictates — a sequential chain of merges, or independent merges whose results are then compared/combined. Never merge DataFrames with no listed relationship, and never infer one.
- Prefer method chaining. Correct Python/Pandas only.
- **Column access (mandatory):** always use bracket indexing — `df["column name"]` — never dot access (`df.column_name`) and never a bare column name inside `.query()`/`.eval()`. Columns may contain spaces, accented/non-ASCII characters, punctuation, or be purely numeric text (`"2023"`), all of which break attribute access and eval-string parsing.

### Data quality — tables are RAW
DataFrames are loaded exactly as stored: no missing-token conversion, no number or date parsing, no null-row dropping. Read the Column Statistics below with this legend:
`Table_i: <rows> rows; <table fact>; ...`, then one line per column: `- name (type): fact; fact; ...`.
`v ×n` = value v in n rows; a bare count (`12 null`, `3 untrimmed`) counts rows; `blank` = empty or whitespace-only text, which is not NULL.
"..." = a text value exactly as stored (JSON-escaped): in SQL write it in single quotes, doubling any '. `"abc"…` = a longer stored value starting with abc (match with LIKE 'abc%' / .str.startswith). Unquoted values are numbers or booleans; dates appear as YYYY-MM-DD after the word `parsed`, and dates are UTC: `parsed a..b` is UTC, and the recipes give UTC; convert to a local zone only when the plan names it. `offsets +00:00 on every value` / `on n of m values` = how many of the ISO values carry a UTC offset.
`<k> values:` lists EVERY value; `1 value: v ×n` = v is the only stored value, the other rows are null or blank. `most common` lists only frequent ones; `+t more ×c each` = t more values with c rows each; `<k> distinct, each ×c` = every value in exactly c rows.
Shown numbers are stored values (compare non-whole floats with a tolerance, never ==), except: `quartiles` are rounded summaries; `a..b` (min..max), `far high` values and `all others <= x` are rounded outward past long decimals.
`(missing token)` / `missing tokens` = values matching the configured missing-value spellings; usually missing, but you decide (a "-" can be a real category).
`number text` casts to a number but may still be an identifier (ZIP, id, house number): cast only for arithmetic or numeric comparison. `date as number yyyymmdd` = whole numbers that are calendar dates (20200416): parse them, never add or average them.
`identifier:` = structural evidence that the column names records: never sum, average or correlate it, count its distinct values, join only on the same id space; a measure constant per category can read as one — judge by its name and values. `one row per x` in the header = each row is one x.
`filled down:` = a spreadsheet fill-down: a value holds for the blank rows below it until the next value — forward-fill it before grouping, filtering or counting per value; `(k once x is filled down)` on repeated rows = the repeats left after that fill.
Backticked text is a pattern or a literal fragment (an affix such as `%`), never a whole stored value: in `patterns` and `not ...` groups only `#` (a run of digits) is a placeholder and letters are literal; in `code shape` and `date-like text` shapes `9` = one digit and `a` = a run of letters.
An absent fact means nothing to report.

If the plan has a `clean` step, implement its `params.actions` literally, with the EXACT values Column Statistics quotes — never invented ones, never a backticked shape:
- missing tokens and an action's `"treat_as_missing"` text values: `s.where(~s.str.strip().str.lower().isin(["n/a", "-"]))` (the shown values, lowercased and trimmed); a numeric sentinel in a numeric column: `s.mask(s.isin([-1, 999]))`
- `cast` to a number — always `pd.to_numeric(..., errors="coerce")`, never a raw `.astype(float)`, which raises on any stray value. `number text: plain`: `pd.to_numeric(s.str.strip(), errors="coerce")`. Any other number-text format (the action's `format`): strip the shown prefix/suffix/unit, remove the shown thousands separator (`thousands space`: `.str.replace(r"\s", "", regex=True)`), replace the shown decimal mark with '.', then `pd.to_numeric(..., errors="coerce")` and `.where(np.isfinite(...))` (`import numpy as np`) — e.g. `decimal comma; thousands dot`: `pd.to_numeric(s.str.strip().str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")`. Ambiguous or mixed: use the format the action states.
- `cast` to a date — the result is a naive UTC datetime (a value with no offset is taken as UTC). `date text ISO` with no `offsets` fact: `pd.to_datetime(s.str.strip(), format="ISO8601", errors="coerce")`; `offsets ... on every value`: `pd.to_datetime(s.str.strip(), format="ISO8601", errors="coerce", utc=True).dt.tz_localize(None)`; `offsets ... on n of m values`, three lines: `t = s.str.strip()`; `off = t.str.contains(r"\d\d:\d\d(?::\d\d(?:\.\d+)?)?\s*(?:Z|[+-]\d\d(?::?\d\d)?)$", regex=True, na=False)`; `out = pd.to_datetime(t.where(~off), format="ISO8601", errors="coerce").fillna(pd.to_datetime(t.where(off), format="ISO8601", errors="coerce", utc=True).dt.tz_localize(None))` — never `utc=True` over values without an offset (pandas lets such a value inherit the offset of the value before it). Any other shown format: `pd.to_datetime(s.str.strip(), format="<format>", errors="coerce")`, e.g. `"%m/%d/%Y %I:%M:%S %p"` (a `(DuckDB: ...)` spelling in the fact is for SQL only); order unknown: use the single format the action states. A fact with `also <format>` holds a second format: parse with each and combine, `pd.to_datetime(s.str.strip(), format="<format>", errors="coerce").fillna(pd.to_datetime(s.str.strip(), format="<also>", errors="coerce"))`. Never compare raw date text with a parsed date.
- `date as number yyyymmdd`: `pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")`.
- `cast` to a time (`time text`): `pd.to_datetime(s.str.strip(), format="<format>", errors="coerce").dt.time`.
- `impute` with `"strategy": "ffill"` (a `filled down` column): the frame keeps its file order; carry the last value forward over the void cells, which are blank text, not NaN: `df2 = df.assign(**{{"col": df["col"].where(df["col"].str.strip() != "").ffill()}})` — into a new frame, never into the alias.
- UTC timestamps converted to local time (when the plan says so; it names the zone): `<parsed>.dt.tz_localize("UTC").dt.tz_convert("<zone>").dt.tz_localize(None)` on the parsed UTC datetime — derive day, month, year or hour from it.
- `.dropna(subset=[...])` to drop rows; `.fillna(...)` to impute

Recipes for the other facts:
- `digit code`: compare as text (`s == "00123"`), never cast.
- `bool with nulls`: mask with `s.eq(True)` or `s.astype("boolean").fillna(False)` — never `s.astype(bool)` (counts null as True), `~s` or `df[s]` (both raise on the nulls).
- `untrimmed` / `case/space variants`: compare and group on `s.str.split().str.join(" ").str.lower()`.
- `blank`: missing like NaN, but `.notna()`/`.dropna()` keep it: use `s.where(s.str.strip() != "")`.
- `<k> infinite`: those rows make sums and means infinite: keep `np.isfinite(s)` rows before aggregating.
- excluding a `total row`: `df[df["<label col>"] != "<label>"]`, or with blank labels `df[df["<measure>"] != <value>]` (number text: the stored text, quoted) — never by row position.
- a text column used as a number with no `clean` step casting it: still `pd.to_numeric(..., errors="coerce")` it rather than assuming clean data.

If the plan has a `derive` step whose `params.actions` use `"technique": "flag"` or `"bucket"` (preserving a censored, capped or extreme value as a feature instead of discarding it): implement `flag` as a boolean column via `.str.contains(...)`/`.isin([...])`/a regex matching the action's `rule`; implement `bucket` as a categorical column via `np.select([...], [...], default=...)` or `pd.cut(...)` for numeric ranges, mapping the ranges/patterns in `rule` to their labels — using the exact values and bounds Column Statistics shows (a `far high` bound, a `repeated min`/`max` value, a `not numbers` value), never invented ones.

### Correlate / limit / rank steps
- `correlate`: `df[columns].corr(method=params.get("method", "pearson"))`, or `df.groupby(params["group_by"])[columns].corr(method=...)` when `params.group_by` is set. For exactly 2 columns, extract the pairwise coefficient (e.g. `.iloc[0, 1]`) rather than returning the whole symmetric matrix; name it `params.output_column` when present.
- `limit`: `params.how` (default `"head"`) selects the idiom — `"head"` → `.head(params["n"])` in whatever order the plan already produced (typically right after a `sort`); `"largest"`/`"smallest"` → `.nlargest(params["n"], params["by"])` / `.nsmallest(...)` for a self-contained top/bottom-N with no separate `sort`.
- `rank`: `.rank(method=params.get("method", "average"), ascending=params.get("ascending", True))` on `params["by"]`, assigned to `params["output_column"]`; with `params.group_by` set, compute per group via `.groupby(params["group_by"])[params["by"]].rank(...)`.

### Table usage
A DataFrame is justified only if its columns appear in output, filters, or aggregations — not merely in a merge key. Using one solely to restrict rows via a merge is **not** justified unless the question explicitly requires cross-table validation ("records appearing in both sources"). Touch only the minimal columns the plan's steps call for — table usage and columns were already fixed and judged during planning.

### Output (conform to the Pydantic schema — `queries` holds exactly ONE item)
- `question`: the plan's business question, faithfully preserved (see above)
- `query`: Python/Pandas code
- `motivation`: 2–3 sentences in business language covering (1) the analytical value, (2) what specific columns each DataFrame uniquely contributes, (3) why this merge/join strategy is correct.

Do NOT generate `difficulty`, `topic`, `story`, `translated_question`, `detected_language`, `question_keywords`, `translated_question_keywords`, or `tables` — the plan already decided them and they are attached to your query automatically.

### Inputs
- **Detected languages** — these datasets may use: `{languages}`
- **Aliases** — use only these DataFrame variable names: `{aliases}`
- **Tables** — schema, columns, types, metadata: `{table}`
- **Verified table relationships** — the ONLY ways DataFrames may be combined; use each relationship's keys and settings exactly as listed, composed in the shape the Structured Plan dictates:

```
{matches}
```
