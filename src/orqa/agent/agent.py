import os
import yaml
from pathlib import Path
from prompting import load_prompt
from bs4 import BeautifulSoup
import pandas as pd
import polars as pl
import json
from ai import LLMClient
import re
import time
import importlib
import inspect
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, Any, Type,Set

class agent:
    def __init__(self,prompt_path:Path,config_path:Path,metadata:Path):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.prompt_path= prompt_path
        self.client = LLMClient(self.config_path)
        self.json_file = metadata
        #self.matching_model = _load_response_model

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config

    def _load_response_model(self,model) -> Optional[Type[BaseModel]]:
            """
            Dynamically load Pydantic model from config.
            
            Returns:
                Pydantic model class or None if not specified
            """
            ## matching_model or response_model
            response_model_config = self.config.get(model)
            
            if not response_model_config:
                return None
            
            module_name = response_model_config.get("module")
            class_name = response_model_config.get("class")
            
            if not module_name or not class_name:
                raise ValueError(
                    "response_model must specify both 'module' and 'class'"
                )
            
            try:
                # Import the module
                module = importlib.import_module(module_name)
                
                # Get the class from the module
                model_class = getattr(module, class_name)
                
                # Verify it's a Pydantic model
                if not issubclass(model_class, BaseModel):
                    raise TypeError(
                        f"{class_name} is not a Pydantic BaseModel"
                    )
                
                return model_class
                
            except ImportError as e:
                raise ImportError(
                    f"Could not import module '{module_name}': {e}"
                )
            except AttributeError as e:
                raise AttributeError(
                    f"Could not find class '{class_name}' in module '{module_name}': {e}"
                )
    
    def collect_pydantic_models(self,
        root_model: Type[BaseModel],
        collected: Set[Type[BaseModel]] | None = None
    ) -> Set[Type[BaseModel]]:
        """
        Recursively collect all Pydantic models referenced by a root model.
        """
        if collected is None:
            collected = set()

        if root_model in collected:
            return collected

        collected.add(root_model)

        for field in root_model.model_fields.values():
            annotation = field.annotation
            origin = getattr(annotation, "__origin__", None)

            if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                self.collect_pydantic_models(annotation, collected)

            elif origin in (list, set, tuple, dict):
                for arg in getattr(annotation, "__args__", []):
                    if inspect.isclass(arg) and issubclass(arg, BaseModel):
                        self.collect_pydantic_models(arg, collected)

        return collected

    def pydantic_models_to_string(self,root_model: Type[BaseModel]) -> str:
        models = self.collect_pydantic_models(root_model)

        # Stable order: dependencies first
        ordered = sorted(models, key=lambda m: m.__name__)

        blocks = []
        for model in ordered:
            source = inspect.getsource(model)
            blocks.append(source.strip())

        return "\n\n".join(blocks)


    def create_table_prompt(self,csv_info: dict):
        """Creates a prompt piece with data information"""
        return load_prompt(self.prompt_path, section="Table", **csv_info)

    def create_structured_prompt(self,model:str):
        """Create a detailed prompt for CSV analysis"""
        return  load_prompt(self.prompt_path, section="Pydantic", **{"format":model})

    def create_analysis_prompt(self,csv_info: dict,section_str: str= None,model=None) -> str:
        """Create a detailed prompt for CSV analysis"""
        table = self.create_table_prompt(csv_info)
        prompt = ""
        if section_str:
            prompt =  load_prompt(self.prompt_path, section=section_str, **{"table":table})
        else:
            prompt =  load_prompt(self.prompt_path, **{"table":table})
        if model is not None:
            prompt +=f"\n {self.create_structured_prompt(self.pydantic_models_to_string(model))}"
        return prompt
    def create_matching_prompt(self,csv_infos: dict,task,section_str: str= None, model=None) -> str:
        """Create a detailed prompt for CSV analysis"""
        table = ""
        prompt = ""
        for info in csv_infos:
            table += self.create_table_prompt(info)+"\n" 
        if section_str:
            prompt = load_prompt(self.prompt_path, section=section_str, **{"table":table,"task":task})
        else:
            prompt =  load_prompt(self.prompt_path, **{"table":table})
        if model is not None:
            prompt +=f"\n {self.create_structured_prompt(self.pydantic_models_to_string(model))}"
        return prompt

    def analyze(self,csv_file:Path):
        try:
            csv_info,column_typings = self.load_csv_info(csv_file)
            if csv_info["numrows"] == 0 or csv_info["numcolumns"] == 0:
                return 0,{}
            print(f"Loading CSV: {csv_file.name}")
            model = self._load_response_model("response_model")
            prompt = self.create_analysis_prompt(csv_info,"Analyze",model)
            print(prompt)
            result,tokens = self.client.complete_analysis(prompt,schema=csv_info["columns"],column_typings=column_typings,reply_model=model)
            print("="*60)
            print("ANALYSIS RESULTS")
            print("="*60)
            return  csv_info["id"],{"csv":csv_file.name,"result":result,"token_usage":tokens}
        except FileNotFoundError as e:
            print(f"Error: '{e}'")
            print("\nTo test this code, create a sample CSV first:")
            return 0,{}
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            return 0,{}


    def match_datasets(self, task:str,datasets:[Path]):
        try:
            csv_info = self.load_datasets_info(datasets)
            model = self._load_response_model("matching_model")
            prompt = self.create_matching_prompt(csv_info,task,"Match",model)
            print(prompt)
            result,tokens = self.client.complete_match(prompt,reply_model=model)
            print("="*60)
            print("Matching RESULTS")
            print("="*60)
            #return {"result":"","token_usage":0}
            return  {"result":result,"token_usage":tokens}
        except FileNotFoundError as e:
            print(f"Error: '{e}'")
            print("\nTo test this code, create a sample CSV first:")
            return {"result":"","token_usage":0}
        except Exception as e:
            print(f"\n❌ Matching failed: {e}")
            return {"result":"","token_usage":0}

    def clean_html_notes(self,html_text):
        """
        Clean HTML from notes field and extract plain text.
        """
        if not html_text:
            return ""
        
        # Parse HTML
        soup = BeautifulSoup(html_text, 'html.parser')
        
        # Extract text and clean up whitespace
        text = soup.get_text(separator=' ', strip=True)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove &nbsp; and other HTML entities that might remain
        text = text.replace('\xa0', ' ')
        
        return text.strip()

    def extract_id_from_filename(self,filename):
        """
        Extract ID from filename format: [data_title_]id__uuid.csv
        Handles cases where data and title might be absent.
        """
        print(filename)
        # Remove .csv extension
        name = filename.replace('.csv', '')
        
        # Split by '__' to get the UUID part
        parts = name.split('__')
        if len(parts) >= 2:
            return parts[-1]  # Return the UUID (last part)
        return None

    def find_json_metadata_by_resource(self,json_data, file_id):
        """
        Find matching JSON object by searching for the ID in the resources array.
        Returns the parent object's metadata including cleaned notes.
        """
        for obj in json_data:
            # Check if any resource has matching ID
            resources = obj.get('resources', [])
            for resource in resources:
                if resource.get('id') == file_id:
                    # Found match - return parent object's metadata
                    return {
                        'title': obj.get('title', 'N/A'),
                        'notes': self.clean_html_notes(obj.get('notes', '')),
                        'organization': obj.get('organization', {}).get('title', 'N/A'),
                        'tags': [tag.get('display_name', '') for tag in obj.get('tags', [])],
                        'metadata_created': obj.get('metadata_created', 'N/A'),
                        'metadata_modified': obj.get('metadata_modified', 'N/A'),
                        'license': obj.get('license_title', 'N/A'),
                        'url': obj.get('url', 'N/A'),
                    }
        return None

    def extract_dataset_info(self,csv_path:Path):
        schema_df = pl.read_csv(csv_path, n_rows=0)
        # Get first 20 columns (or all if fewer)
        columns_to_read = schema_df.columns[:20]
        # Read CSV with only the selected columns
        df = pl.read_csv(csv_path, columns=columns_to_read)
        column_typings = {}
        coldetails = ""
        for col in df.columns:
            coldetails += f"\n- {col}:"
            coldetails += f"\n  Type: {df[col].dtype}"
            coldetails += f"\n  Unique values: {df[col].n_unique()}"
            coldetails += f"\n  Null count: {df[col].null_count()}"
            column_typings[col]=df[col].dtype.is_numeric() #isinstance(df[col].dtype, pl.datatypes.NumericType)
            # pl.datatypes.is_numeric(df[col].dtype)
        
        # Return dictionary with keys matching load_prompt parameters
        info = {
            "id":csv_path.name,
            "filename": csv_path.name,
            "numrows": len(df),
            "numcolumns": len(df.columns),
            "coldetails": coldetails,
            "sample": json.dumps(df.head(5).to_dicts(), indent=2),
            "columns": df.columns,

        }
        return column_typings,info

    def load_datasets_info(self, csv_paths:[Path]) -> dict:
        infos = []
        for path in csv_paths:
            # Build detailed column information string
            _ ,info = self.extract_dataset_info(path)
            info["metadata"] = "None"
            infos.append(info)
        return infos

    def load_csv_info(self, csv_path: Path) -> dict:
        """
        Load CSV and extract relevant information for the LLM.
        Returns a dict ready to be unpacked as kwargs for load_prompt.
        """
        # Build detailed column information string
        column_typings,info = self.extract_dataset_info(csv_path)
        
        try:
            if self.json_file and self.json_file.exists():
                    with open(self.json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        # Extract ID from CSV filename
                        file_id = self.extract_id_from_filename(csv_path.name)
                        
                        if file_id:
                            # Search for ID in resources
                            print(file_id)
                            metadata = self.find_json_metadata_by_resource(json_data, file_id)
                            if metadata:
                                info['metadata'] = f"notes:{metadata['notes']}"
                                info["id"] = file_id
                            else:
                                info['metadata'] = "None"
                                #info['metadata'] = None
                                #info['note'] = f"No metadata found for resource ID: {file_id}"
                        else:
                             info['metadata'] = "None"
            else:
                    info['metadata'] = "None"
        except Exception as e: 
            info['metadata'] = "None"
            return info
        return info,column_typings


if __name__ == "__main__":
   

    data = {}
    documents = [
            r"D:\uk_small\uk_small\datasets\csv\2011-03-31-Organogram-(Senior)__c529c8da-0e24-4778-8c0a-968ff209b5b5.csv",
            r"D:\uk_small\uk_small\datasets\csv\2011-09-30-Organogram-(Junior)__344ad9e9-a77c-4b18-a4aa-46980878a062.csv", 
            r"D:\uk_small\uk_small\datasets\csv\2011-09-30-Organogram-(Senior)__13a28011-59e7-4aa7-a41a-d3afed9552f5.csv",
            r"D:\uk_small\uk_small\datasets\csv\2016-July-transactions__a27f90fc-e4eb-4d06-b376-f84b6b17188d.csv",
            r"D:\uk_small\uk_small\datasets\csv\2017-03-31-Organogram-(Junior)__b08128ce-2293-4ff9-881b-57f9236b4984.csv",
            r"D:\uk_small\uk_small\datasets\csv\2017-03-31-Organogram-(Senior)__f9c9ee5a-3adb-4670-8961-6cdd6c0ac6b5.csv",
            r"D:\uk_small\uk_small\datasets\csv\2019-February-Return-(Forestry-Commission-England)__139700c2-fd54-4f40-bbf1-3f6e80308518.csv",
            r"D:\uk_small\uk_small\datasets\csv\2019-February-return__5cea3557-6fb5-4998-822e-cb5cf57731fa.csv",
            r"D:\uk_small\uk_small\datasets\csv\2019-July-Return-Forestry-England__a1c25c75-ddd4-4b3c-b4fd-e950ef148bd0.csv",
            r"D:\uk_small\uk_small\datasets\csv\2019-Q1-transactions__ed3a0fd2-9185-43b2-89ea-c1558168355f.csv",
            r"D:\uk_small\uk_small\datasets\csv\2020-June-Return---Forestry-England__1d790441-f671-4e38-b92e-5196c762ea45.csv",
            r"D:\uk_small\uk_small\datasets\csv\2023-12-31-Organogram-(Senior)__86b38dcc-8047-4e0e-ac56-d80d3cf913f9.csv"
        ]

    my_agent = agent(Path("prompt.md"),Path("config.yaml"),Path(r"D:\uk_small\uk_small\metadata\metadata.json"))
    file_path="results.json"
    for document in documents:
            id,result=my_agent.analyze(Path(document))
            data[id]=result
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            time.sleep(100)

