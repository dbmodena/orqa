# <img src="static/images/clipart58856.png" alt="Image Alt Text" style="width: 50px; vertical-align: middle;"> OrQA

OrQA (Open Data retrieval and Question Answering) is a workflow to generate new benchmark datasets for systems evaluation on retrival and tabular question answering on Open Data portals. The workflow is composed by four main stages: 
1. Crawling of data and metadata from the desired Open Data endpoint;
2. Search of candidate related tables;
3. Evaluation of the previously found pairs;
4. Generation of questions and relative SQL queries.

Into the 'scripts' folder you can find all the coded steps to run your own experiments.

OrQA has been built upon [Ollama](https://ollama.com/download) and [LiteLLM](https://docs.litellm.ai/docs/): you will need to manually install Ollama before running the scripts.

Install the required packages,
```sh
$ conda env create -f environment.yml
```


Before running the evaluation and generation scripts, you need to start Ollama server,
```sh
$ ollama serve 
```

and launch LiteLLM,
```sh
(orqa) $ litellm --config litell_config.yml 
```

With the following commands you will be able to create a new questions dataset from the first 1000 available packages on the Canadian Open Data portal. 

```sh
(orqa) $ python orqa_0_open_data_crawler.py CAN 0 1000 https://open.canada.ca/data/api/action
(orqa) $ python orqa_1_create_blend_index.py CAN 0 1000
(orqa) $ python orqa_2_search_candidates.py CAN 0 1000
(orqa) $ python orqa_3_evaluation.py CAN 0 1000
(orqa) $ python orqa_4_generate_questions.py CAN 0 1000
```

Customisations of the workflow, like the model used for question generation, are not available as program arguments or as external configuration file yet, and have to be hardcoded into the scripts. 
