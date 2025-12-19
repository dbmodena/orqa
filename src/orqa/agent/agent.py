import os
from pathlib import Path
from prompting import load_prompt
from bs4 import BeautifulSoup
import pandas as pd
import polars as pl
import json
from ai import LLMClient
import re
import time

class agent:
    def __init__(self,prompt_path:Path,config_path:Path,metadata:Path):
        self.config_path = config_path
        self.prompt_path= prompt_path
        self.client = LLMClient(self.config_path)
        self.json_file = metadata


    def create_analysis_prompt(self,csv_info: dict, prompt_path: Path,section_str: str= None) -> str:
        """Create a detailed prompt for CSV analysis"""
        if section_str:
            return  load_prompt(prompt_path, section=section_str, **csv_info)
        else:
            return  load_prompt(prompt_path, **csv_info)

    def analyze(self,csv_file:Path):
        try:
            csv_info,column_typings = self.load_csv_info(csv_file)
            if csv_info["numrows"] == 0 or csv_info["numcolumns"] == 0:
                return 0,{}
            print(f"Loading CSV: {csv_file.name}")
            prompt = self.create_analysis_prompt(csv_info,self.prompt_path,"Analyze")
            result,tokens = self.client.complete(prompt,schema=csv_info["columns"],column_typings=column_typings)
            print("="*60)
            print("ANALYSIS RESULTS")
            print("="*60)
            #print(result)
            #print(tokens)
            return  csv_info["id"],{"csv":csv_file.name,"result":result,"token_usage":tokens}
        except FileNotFoundError as e:
            print(f"Error: '{e}'")
            print("\nTo test this code, create a sample CSV first:")
            return 0,{}
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            return 0,{}

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

    def load_csv_info(self, csv_path: Path) -> dict:
        """
        Load CSV and extract relevant information for the LLM.
        Returns a dict ready to be unpacked as kwargs for load_prompt.
        """
        schema_df = pl.read_csv(csv_path, n_rows=0)
    
        # Get first 20 columns (or all if fewer)
        columns_to_read = schema_df.columns[:20]
        # Read CSV with only the selected columns
        df = pl.read_csv(csv_path, columns=columns_to_read)
        # Build detailed column information string
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

