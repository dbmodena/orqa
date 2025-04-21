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

OrQA is built on top of [Ollama](https://ollama.com/download) and [LiteLLM](https://docs.litellm.ai/docs/).  
You will need to manually install Ollama before running the scripts.

Install the required Python packages via Conda:

```sh
$ conda env create -f environment.yml
```

---

### 🚀 Starting the Services

Before running the evaluation and generation scripts, start the Ollama server:

```sh
$ ollama serve 
```

Then, launch LiteLLM:

```sh
(orqa) $ litellm --config litell_config.yml 
```

---

### 🧪 Run the Workflow

Use the following commands to create a new dataset from the first 1000 available packages on the Canadian Open Data portal:

```sh
(orqa) $ python orqa_0_open_data_crawler.py CAN 0 1000 https://open.canada.ca/data/api/action
(orqa) $ python orqa_1_create_blend_index.py CAN 0 1000
(orqa) $ python orqa_2_search_candidates.py CAN 0 1000
(orqa) $ python orqa_3_evaluation.py CAN 0 1000
(orqa) $ python orqa_4_generate_questions.py CAN 0 1000
```

---

### ⚙️ Customization

At this stage, customization of the workflow—such as selecting different models for question generation—is not yet available via command-line arguments or external config files. These settings must be hardcoded directly into the scripts.

---

In the `dataset` folder there is a first dataset version generated with OrQA workflow: this dataset contains 1,000 questions created from Canadian and UK Open Data tables.
