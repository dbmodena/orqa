# Benchmark (MCP server)

*(formerly `orqa.dataset_search` — renamed because the module now hosts
the whole benchmark harness, not just dataset discovery.)*

MCP server for one city's dataset-retrieval benchmark (retrieval-only
for now — no code-writing, execution or result reporting). It serves
two things, both configured by the city's workflow yaml:

1. **Dataset discovery** — a keyword-based reverse index over the city's
   normalized datasets metadata
   (`<data_path>/metadata/normalized_metadata.json`). Given keywords
   extracted from a natural language question, it finds the crawled CSV
   files needed to answer it.
2. **Benchmark questions** — the questions produced by the
   `generate-statements` step, read fresh at server startup from
   `candidates_discovery/generated_queries.json` or
   `generated_queries_semantic.json` — whichever
   `tasks.candidates_discovery.method` (`blend` or `semantic`) selects,
   see `queries_path` in `src/conf/config.py` — organized by id and
   phrased in the portal's target language, served as a list the
   answering agent works through one at a time.

## Target language

Questions, keywords and metadata are in the portal's language, taken
from `tasks.query_generation.languages` in the workflow yaml: Spanish
for valencia/madrid, Italian for bologna/modena, French for paris,
English for nyc/uk/canada. **Search keywords must be translated into
that language** before calling `search_datasets` — the reverse index
only matches the portal's language (accent-insensitively).

## Benchmark folder

All run artifacts live under `<data_path>/benchmark/<kind>/`, adjacent
to `candidates_discovery/` and `metadata/`, where `<kind>` is the
programming language selected by `tasks.query_generation.kind`
(`benchmark/pandas` or `benchmark/sql`) — a label carried over from the
question source, not something this retrieval-only server writes code
for:

```
benchmark/
└── pandas/                      # or sql/
    └── questions_todo.json      # one entry per flat question id (st_5_0):
                                 #   fetched: bool (has get_question handed it out)
```

The list is synced at server startup from the generated queries (new
questions are added not yet fetched; already-fetched ones survive
restarts and re-syncs) — `fetched` is set by `get_question`, the only
progress tracked for now.

## Reverse index backends

Two interchangeable backends implement the reverse index, selected with
`tasks.mcp_search.backend` in the city's workflow yaml. Both rank with
field-weighted BM25 (title > tags > columns > publisher > description)
and fold accents for the multilingual portals, and both are created
automatically at server startup when missing and recreated whenever the
normalized metadata changes:

- `elasticsearch` (default) — the index lives in an Elasticsearch index
  named `orqa-<provider>-<city>` on the cluster configured by
  `tasks.mcp_search.elasticsearch_url` (the `ELASTICSEARCH_URL` env
  variable takes precedence). A local dev instance is enough:

  ```sh
  docker run -d --name es -p 9200:9200 \
    -e discovery.type=single-node -e xpack.security.enabled=false \
    docker.elastic.co/elasticsearch/elasticsearch:8.17.0
  ```

- `builtin` — pure-Python index materialized in
  `<data_path>/index/metadata_index.json`; no external infrastructure.

## Running the server

The server is a workflow step: the city selects the data and its
workflow yaml, where the port is configured (`tasks.mcp_search.port`).

```sh
# stdio transport (the MCP client spawns this command)
uv run src/main.py --country usa --city nyc --steps mcp_search --mode stdio

# streamable HTTP on the port from the city's workflow yaml
uv run src/main.py --country spain --city valencia --steps mcp_search --mode port

# only (re)build the index and exit
uv run src/main.py --country usa --city nyc --steps mcp_search --mode build
```

In `port` mode clients connect to `http://<host>:<port>/mcp`.

Example MCP client configuration (stdio) — the server is city-agnostic,
the `--country`/`--city` flags select which collection it serves:

```json
{
  "mcpServers": {
    "orqa-benchmark": {
      "command": "uv",
      "args": [
        "run", "src/main.py",
        "--country", "spain", "--city", "valencia",
        "--steps", "mcp_search", "--mode", "stdio"
      ]
    }
  }
}
```

## Tools

Retrieval benchmark loop: **fetch** a question → **search** its
candidate tables (ranked + related on the fly) → **evaluate** the table
selection against the hidden ground truth. That's the whole loop for
now — there is no code-writing, execution or result-reporting tool.

- `get_question(question_id?)` — the next question **not yet fetched**
  (or a specific one by id): just `{id, question}`, nothing else — no
  keywords, difficulty or status. Called without arguments it marks the
  question fetched and hands out the next unfetched one on every call.
- `list_questions(fetched?)` — every question as just `{id, question}`;
  `fetched` optionally filters to already-fetched (`true`) or
  not-yet-fetched (`false`).
- `search_datasets(keywords, top_k?, only_available?, with_relationships?, matcher?)` —
  ranked datasets (BM25 over keywords, in the portal's target language)
  **plus**, by default, the Valentine join/union/correlation
  relationship between every pair of ranked results whose CSV is on
  disk, computed on the fly (same shape as `compare_tables`, condensed
  to best join + union ratio + top correlation). Set
  `with_relationships=false` to skip that and get just the ranking.
- `evaluate_tables(question_id, resource_ids)` — check the dataset
  resource_ids retrieved for a question against its hidden ground-truth
  tables: set precision/recall/F1 over resource_ids, plus which ones
  are correct/missing/extra. This is the terminal step of the loop.

Dataset inspection:

- `get_dataset_metadata(resource_id)` — full normalized metadata
  (description, tags, column names/types, URLs).
- `preview_dataset(resource_id, n_rows?, how?)` — rows of the CSV.
  Deterministic: `how="sample"` (default) is a seed-fixed random sample
  (workflow yaml `seed`), identical across calls and sibling workflows;
  `how="head"` returns the first rows.
- `compare_tables(resource_id_a, resource_id_b, matcher?, top_k?)` —
  the full-detail version of search_datasets' on-the-fly relationships,
  for one deliberate pair (or a table outside the top-k ranking): ranked
  column matches (join key / union candidates), joined row count on the
  best key, the fraction of unionable columns, and Pearson correlations
  between matched numeric columns. Read the results as pandas `merge` /
  `concat` / `corr` feasibility.
- `index_info()` — source, target language, statement kind, seed,
  dataset count, datasets folder, question counts (`total`, `fetched`,
  `unfetched`).

## CLI (quick index queries without an MCP client)

```sh
python -m orqa.benchmark --data-dir data --list-sources
python -m orqa.benchmark --data-dir data --source socrata/nyc taxi license expiration
```
