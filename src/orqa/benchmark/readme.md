# Benchmark (round-trip solver)

The benchmark harness for one city's open-data retrieval-and-query
benchmark. Given ONLY a question — never the hidden ground truth — an
internal agent independently retrieves candidate tables, relates them via
Valentine matching, decides which to use, writes and executes code, and the
result is scored against the hidden ground truth two ways: table/column
selection accuracy, and result value/dtype accuracy. This is the
round-trip/cycle-consistency check from the Text-to-SQL benchmark-
construction literature (independently re-derive an answer from the
question alone and check it against the ground truth), applied to orqa's
own generated benchmark.

*(Formerly an MCP server that exposed the same retrieval primitives as
tools for an external client to drive by hand — fetch a question, search,
evaluate the table selection — with no code-writing, execution or result
reporting. Retired: this module now runs the whole loop itself.)*

## Questions

The questions are the ones produced by the `generate-statements` step, read
fresh from `candidates_discovery/generated_queries.json` or
`generated_queries_semantic.json` — whichever `tasks.candidates_discovery.method`
(`blend` or `semantic`) selects, see `queries_path` in `src/conf/config.py` —
organized by id and phrased in the portal's target language
(`tasks.query_generation.languages` in the workflow yaml).

## The solver loop

For each not-yet-solved question (`orqa.benchmark.solve.solve_one_question`):

1. **Keywords** (`orqa.agent.agents.BenchmarkSolver.BenchmarkSolverAgent.generate_keywords`)
   — an LLM call extracts retrieval keywords from the question, in the
   question's own detected language (the reverse index only matches text
   in the language it was indexed in).
2. **Retrieve** — the top `tasks.benchmark_solver.top_k_tables` candidates
   from the reverse index (`orqa.benchmark.index`), then **relate** them:
   Valentine join/union/correlation matching (`tasks.benchmark_solver.matcher`)
   between every pair among them, capped to a fast row/column sample —
   a quick screen, not the figure that feeds the final answer.
3. **Select** (`select_tables`) — an LLM call decides which candidate(s) —
   not every one is necessarily relevant — and which of their columns
   actually answer the question, and the answer's expected shape.
4. **Code** (`write_code`) — an LLM call writes Pandas/SQL code (per
   `tasks.query_generation.kind`) using exactly the tables/columns Phase 3
   selected.
5. **Validate + execute** — the code is checked and executed through the
   same `PandasValidator`/`SQLValidator` the main generation pipeline uses
   (sandboxed: separate process, timeout, memory limit), against the FULL,
   uncapped table (a capped read here would make even correct code produce
   a systematically wrong count/sum). The hidden reference code is executed
   the same way, against its own original ground-truth table(s).
6. **Score** — table/column selection against
   `orqa.benchmark.questions.evaluate_table_retrieval`'s precision/recall/F1,
   and (when both sides executed successfully) result value/dtype against
   `orqa.benchmark.solve.compare_results`'s diagnostic checklist (dtype
   match, shape match, exact match, and a numeric difference magnitude for
   scalar answers — independent facts, not one collapsed verdict, so a
   failure's actual character stays visible).

No phase retries itself — a benchmark solver measures raw capability
against retrieved, not-guaranteed-relevant context; looping any phase until
it succeeds would measure the loop, not the question's actual difficulty.

## Reverse index backend

Selected by `tasks.mcp_search.backend` in the city's workflow yaml. Both
rank with field-weighted BM25 (title > tags > columns > publisher >
description) and fold accents for the multilingual portals; both are
created automatically when missing and recreated whenever the normalized
metadata changes:

- `builtin` (default) — pure-Python index materialized in
  `<data_path>/index/metadata_index.json`; no external infrastructure.
- `elasticsearch` — the index lives in an Elasticsearch index named
  `orqa-<provider>-<city>` on the cluster configured by
  `tasks.mcp_search.elasticsearch_url` (the `ELASTICSEARCH_URL` env
  variable takes precedence).

## Running it

A workflow step: the city selects the data and its workflow yaml.

```sh
uv run src/main.py --country usa --city nyc --steps solve-benchmark
uv run src/main.py --country spain --city valencia --steps solve-benchmark
```

## Results

All run artifacts live under `<data_path>/benchmark/<kind>/`, adjacent to
`candidates_discovery/` and `metadata/`, where `<kind>` is the programming
language selected by `tasks.query_generation.kind` (`benchmark/pandas` or
`benchmark/sql`):

```
benchmark/
└── pandas/                      # or sql/
    └── solver_results.json      # one entry per flat question id (st_5_0):
                                  #   solved: bool
                                  #   result: the full solve_one_question
                                  #     output once solved — keywords,
                                  #     detected_language, candidates,
                                  #     table_selection, code,
                                  #     table_evaluation, execution,
                                  #     result_evaluation, token usage
```

The list is synced at step startup from the generated queries (new
questions are added not yet solved; already-solved ones survive restarts
and re-syncs) — `solve_benchmark` persists incrementally after every
question, so an interrupted run resumes without re-spending LLM calls on
work already done.

## CLI (quick index queries without running the solver)

```sh
python -m orqa.benchmark --data-dir data --list-sources
python -m orqa.benchmark --data-dir data --source socrata/nyc taxi license expiration
```
