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

---

### Run the Workflow

In conf/workflow there are the workflow configuration files for UK and Canada (.yaml). You can customize them in order to modify, for instance,
the number of datasets fetched during the initial crawling stage and their sizes.

To run a workflow, you can call:

```sh
$ python main.py <canada | uk>
```

