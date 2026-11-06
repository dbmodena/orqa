# <img src="static/images/clipart58856.png" alt="Image Alt Text" style="width: 50px; vertical-align: middle;"> OrQA

**OrQA** (Open Data Retrieval and Question Answering) is a workflow for generating new benchmark datasets for retrieval and tabular question answering model evaluation on Open Data.

The workflow is composed of four main stages:

1. Crawling data and metadata from the desired Open Data endpoint  
2. Searching for candidate related tables  
3. Evaluating the previously found pairs  
4. Generating questions and corresponding SQL queries

All scripts needed to run your own experiments are located in the `scripts` folder.

---

### 🧰 Requirements

- Python environment
Install the required Python packages via Conda:

```sh
$ conda env create -f env.yaml
```

- Define a "DATADIR" environment variable path. This will be the base path for OrQA and all the stages.

```sh
$ export DATADIR=/path/to/your/data/directory
```

### OrQA Files Organization

OrQA organizes its data under the environment variable path "DATADIR". 

DATADIR/open_data/ckan/<tag>/datasets/<format> --> here will be stored all the crawled datasets, with the given format.

                            /metadata/ --> here will be saved metadata JSON files about datasets fetched in the initial crawling stage.
                            /log/ --> logfiles for all the stages are here.
                            /BLEND/index.db --> here will be saved the BLEND index after creation.

-  Apache Solr Installation
A local installation of Apache Solr search engine allows us to simulate a CKAN Open Data endpoint against which we can make query.

After you have downloaded an Apache Solr release (OrQA is tested on [9.10][https://www.apache.org/dyn/closer.lua/solr/solr/9.10.0/solr-9.10.0.tgz?action=download]),
you can load the metadata fetched during the crawling stage to the Apache Solr server with the Python class "src/solr/solr.py:Solr".

Do not run `src/solr/solr.py` against the shared Solr instance: its domain names (`uk`, `nyc`, `bologna`, `paris`, `valencia`) are the names
of LakeGen's live cores, and `create`, `delete`, `post`, `reload` or `refresh` would destroy or overwrite them. Retrieval does not need it:
for uk, nyc, bologna, paris and valencia, OrQA queries LakeGen's own cores read-only at `$SOLR_URL` (default `http://localhost:8983/solr`).

- LLM models

The models every stage calls are configured in `conf/llm/litellm.yaml` (LiteLLM model strings; OCI credentials are read from `.env`).
The dedicated OCI AI cluster is addressed through the `deployments` section of that file, over OCI's OpenAI-compatible route with the
Generative AI API key: each deployment names the endpoint OCID (`DEDICATE_OCI_MODEL` / `_MODEL_2` / `_MODEL_3` in `.env`), the base URL
(`DEDICATE_OCI_BASE_URL`) and the key (`DEDICATED_API_KEY`), and may cap the answer length with `max_output_tokens`. A deployment name can
be used wherever a model is referenced (`model`, `fallback_models`, `judge_profiles.*`). See `docs/structured-output.md`.

---

### Run the Workflow

In conf/workflow there are the workflow configuration files for UK and Canada (.yaml). You can customize them in order to modify, for instance,
the number of datasets fetched during the initial crawling stage and their sizes.

To run a workflow, you can call:

```sh
$ python main.py <canada | uk>
```

The benchmark's search side (LakeGen's Solr cores, the builtin index for the
portals without one, the retrievability gate's retrievers) is documented in
[docs/benchmark-retrieval.md](docs/benchmark-retrieval.md). The retired
Elasticsearch backend, and how to bring it back, is in
[legacy/elasticsearch/README.md](legacy/elasticsearch/README.md).

