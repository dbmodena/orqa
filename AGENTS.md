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
  - `index`
  - `candidates-discovery`
  - `generate-statements`
- Targets are fixed in `src/main.py` and include `canada`, `uk`, `italy/modena`, `italy/bologna`, `spain/madrid`, `spain/valencia`, `france/paris`, and `usa/nyc`.

## Runtime setup

- `DATADIR` must be set to the repository's data base path before running the workflow.
- `src/main.py` also loads `.env` from the repository root via `dotenv`.
- Apache Solr is expected for search/indexing; `src/solr/solr.py` uses `SOLR_URL`, `SOLR_HOME`, `SOLR_PATH`, `SOLR_BIN`, and related environment variables.

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
- Tests: `tests/test_main_cli.py`
