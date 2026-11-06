# OrQA AI Agent Instructions

This repository implements the OrQA workflow for generating benchmark datasets from open data sources, using Python and Apache Solr.

## What this repo is

- `src/` contains the Python application and package source.
- `src/main.py` is the main workflow CLI entrypoint.
- `conf/workflow/*.yaml` defines supported targets and workflow stages.
- `src/orqa/agent/` contains LLM agent logic and prompt-related code.
- `src/solr/` provides Solr integration and local Solr control.
- `tests/` contains unit tests built with Python's `unittest`.

## Important conventions

- The package is source-layout-based: Python code is under `src/` and imported from there.
- Workflows are executed by running `python src/main.py --country <country> [--city <city>] --steps <steps...>`.
- Valid steps are:
  - `crawl`
  - `clean`
  - `normalize-metadata`
  - `index` (only with `tasks.candidates_discovery.method: blend`)
  - `candidates-discovery`
  - `embed-search-metadata`
  - `generate-query-candidates`
  - `generate-statements`
- Targets are fixed in `src/main.py` and include `canada`, `uk`, `italy/modena`, `italy/bologna`, `spain/madrid`, `spain/valencia`, `france/paris`, and `usa/nyc`.

## Runtime setup

- `DATADIR` must be set to the repository's data base path before running the workflow.
- Every `src/main.py` run appends what it prints and logs to `<write_path>/log/orqa.log` (e.g. `data/bologna/log/orqa.log`): 3 files of 25 MB in rotation, the oldest overwritten when all are full, set by the workflow yaml's top-level `logging` block (`src/orqa/utils/run_log.py`).
- `generate-statements` also writes a per-run trace to `<write_path>/log/generation/<run>/` (`latest/` points to the newest): `events.jsonl`, `groups.jsonl`, and `llm/<group>.txt` with every LLM call as sent and received (user message, parameters, reasoning, raw answer, errors; each system prompt once in `llm/system/`), set by `tasks.query_generation.trace.llm_io` (`all`/`failed`/`off`, `src/orqa/utils/generation_trace.py`).
- `src/main.py` also loads `.env` from the repository root via `dotenv`.
- Apache Solr is expected for search/indexing; `src/solr/solr.py` uses `SOLR_URL`, `SOLR_HOME`, `SOLR_PATH`, `SOLR_BIN`, and related environment variables.
- Retrieval for uk, nyc, bologna, paris and valencia (`tasks.search.backend: solr`) queries LakeGen's live Solr cores at `SOLR_URL` (`uk`, `nyc`, `bologna_v2`, `paris_v2`, `valencia_v2`) through `src/solr/client.py`, READ-ONLY. That Solr instance is shared with LakeGen: never update, delete, reload, commit or post to any core, and never run `src/solr/solr.py` `create`/`delete`/`post`/`reload`/`refresh` against it, since its domain names are LakeGen's core names.

## Build / test commands

Preferred commands for local development:

- Install dependencies: follow `README.md` or use `pyproject.toml` with a Python environment.
- Run the workflow CLI:
  - `python src/main.py --country canada --steps crawl index`
  - `python src/main.py --country italy --city bologna --steps crawl clean normalize-metadata`
- Run tests:
  - `python -m unittest discover -s tests`

## What AI agents should focus on first

- `src/main.py` for workflow entry and target resolution.
- `conf/workflow/*.yaml` for execution configuration and dataset targets.
- `src/orqa/agent/` for LLM prompt construction, structured outputs, and pipeline behavior.
- `src/solr/` for dependency management around Solr and how the local search backend is started/configured.
- `README.md` for environment setup, Solr installation pointers, and general project intent.

## Notes for code changes

- Avoid changing workflow target names or step names without updating `src/main.py` and any dependent docs.
- Preserve the source-layout import path semantics when editing or adding modules.
- If adding new workflow stages, update `STEP_CHOICES` and `_step_callable_path` in `src/main.py`.
- Keep data path resolution logic consistent with `DATADIR` and `TargetSpec.relative_data_path`.

## Helpful links

- README: `README.md`
- Package config: `pyproject.toml`
- Workflow CLI: `src/main.py`
- Agent code: `src/orqa/agent/`
- Solr integration: `src/solr/`
- Benchmark retrieval (LakeGen's Solr cores, read-only; builtin index; gate retrievers): `docs/benchmark-retrieval.md`
- Retired Elasticsearch backend and how to reintegrate it: `legacy/elasticsearch/README.md`
- Structured LLM output (budgets, truncation retry, coercion, loop bounds): `docs/structured-output.md`
- Question stage inputs (speakable anchors, table identity, shared rubric): `docs/question-stage-inputs.md`
- LLM models and named deployments (dedicated OCI endpoints, per-endpoint output caps): the `deployments` section of `conf/llm/litellm.yaml`, resolved by `LLMClient._resolve_model_spec` in `src/orqa/agent/llm_client/LLMClient.py`
- Tests: `tests/test_main_cli.py`
