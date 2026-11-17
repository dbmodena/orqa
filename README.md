# OrQA

**OrQA** (Open Data Retrieval and Question Answering) generates benchmark datasets for retrieval and tabular question
answering on Open Data portals: it crawls a portal, finds tables that can be joined or unioned, and generates questions
with the pandas code that answers them, checked by LLM judges and by the portal's own search.

---

### Requirements

- Python 3.13 and the packages of `pyproject.toml`, e.g. with [uv](https://docs.astral.sh/uv/):

  ```sh
  uv sync
  ```

- A `.env` file at the repository root (loaded by `src/main.py`). `DATADIR` is the base folder of all the data:

  ```sh
  DATADIR=/path/to/your/data/directory
  ```

### Data layout

Each target writes under its own folder of `DATADIR`. The folder comes from the target in `src/main.py` and the
workflow yaml's `flat_layout` (e.g. `<DATADIR>/uk/` or `<DATADIR>/ods/bologna/`); inside it:

```
datasets/<format>/       the crawled tables
metadata/                the portal's metadata, raw and normalized
candidates_discovery/    candidate join/union groups and the generated questions
log/                     orqa.log of every run; log/generation/<run>/ traces of generate-statements
```

### Run the workflow

The workflow configuration of each target is in `conf/workflow/*.yaml`. Run one or more steps, in order:

```sh
python src/main.py --country uk --steps crawl clean normalize-metadata
python src/main.py --country italy --city bologna --steps generate-statements
```

Targets: `uk`, `italy/bologna`, `spain/valencia`, `france/paris`, `usa/nyc`.
Steps: `crawl`, `clean`, `normalize-metadata`, `build-solr-core`, `index`, `candidates-discovery`,
`generate-query-candidates`, `generate-statements`.

Two browser views run from `src/`: `uvicorn web_app:app --port 8000` (the generated benchmark) and
`uvicorn run_monitor:app --port 8001` (the runs in progress).

### LLM models

The models every stage calls are configured in `conf/llm/litellm.yaml` (LiteLLM model strings; OCI credentials are read
from `.env`). The dedicated OCI AI cluster is addressed through the `deployments` section of that file, over OCI's
OpenAI-compatible route with the Generative AI API key: each deployment names the endpoint OCID (`DEDICATE_OCI_MODEL` /
`_MODEL_2` / `_MODEL_3` in `.env`), the base URL (`DEDICATE_OCI_BASE_URL`) and the key (`DEDICATED_API_KEY`), and may cap
the answer length with `max_output_tokens`. A deployment name can be used wherever a model is referenced (`model`,
`fallback_models`, `judge_profiles.*`). See `docs/structured-output.md`.

### Apache Solr

OrQA runs its own Solr for the portals LakeGen is scored on: one core per portal, holding the documents LakeGen
searches, which the question stage's retrievability gate queries. Every target needs its core. Their vectors are
`bge-m3`, embedded on CPU by OrQA's own Ollama; questions are embedded by the same server, as LakeGen embeds them.

Prerequisites: Java (tested on OpenJDK 17), Solr (tested on 9.10.1) and Ollama (tested on 0.35, at
`/usr/local/bin/ollama`). In `.env`:

```sh
ORQA_SOLR_HOME=/path/with/room/for/the/indexes/solr   # required
ORQA_SOLR_PORT=8984                                   # default
SOLR_PATH=/opt/solr                                   # default
ORQA_OLLAMA_MODELS=/path/to/ollama/models             # required
ORQA_OLLAMA_PORT=11435                                # default
```

```sh
python src/solr/cli.py start                             # Solr and the embedding server; also: stop, status
python src/main.py --country uk --steps build-solr-core  # builds the core uk_build
python src/solr/cli.py publish uk                        # uk_build -> uk; the replaced core stays as uk_previous
python src/solr/cli.py reembed uk --model bge-m3         # new vectors for an existing core, into uk_build
```

Neither server starts at boot; `start` them again after a reboot. `start` pulls no model; `build-solr-core` and
`reembed` pull a missing one into OrQA's Ollama. Every core uses the configset `conf/solr/portal/`
(`python src/solr/cli.py sync-config` copies a changed one to the server and reloads the cores).

The code is in `src/solr/`: `instance.py` (the server's files, process and cores; every write first checks that the
server's Solr home is `ORQA_SOLR_HOME`, so OrQA never writes to LakeGen's shared Solr on port 8983), `embedder.py` (the
Ollama server), `cli.py` (the commands above), `client.py` (the read-only HTTP client), `index.py` (a portal's core as
retrieval queries it) and `build.py` (the `build-solr-core` step). The gate's retriever panel is
`src/orqa/retrieval/panel.py`.

The search side of the benchmark (the Solr cores and the retrievability gate's retrievers) is documented in [docs/benchmark-retrieval.md](docs/benchmark-retrieval.md). The retired
Elasticsearch backend, and how to bring it back, is in [legacy/elasticsearch/README.md](legacy/elasticsearch/README.md).
