import json
from pathlib import Path
from .utils import load_datasets_metadata, load_dataset_info,save_json,load_json
from conf import OrQAConfig
from .agent.agent import GenerateResponseAgent




def generate_response(cfg:OrQAConfig):
    #cfg.llm_config_path.joinpath("litellm.yaml")
    #cfg.datasets_path
    #cfg.statement_generation.queries_path
    
    return 0