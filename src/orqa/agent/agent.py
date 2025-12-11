import os
from pathlib import Path
from prompting import load_prompt
import pandas as pd
import json
from ai import LLMClient


def create_analysis_prompt(csv_info: dict, prompt_path: Path,section_str: str= None) -> str:
    """Create a detailed prompt for CSV analysis"""
    if section_str:
        return  load_prompt(prompt_path, section=section_str, **csv_info)
    else:
        return  load_prompt(prompt_path, **csv_info)


def load_csv_info(csv_path: Path) -> dict:
    """
    Load CSV and extract relevant information for the LLM.
    Returns a dict ready to be unpacked as kwargs for load_prompt.
    """
    df = pd.read_csv(csv_path)
    
    # Build detailed column information string
    coldetails = ""
    for col in df.columns:
        coldetails += f"\n- {col}:"
        coldetails += f"\n  Type: {df[col].dtype}"
        coldetails += f"\n  Unique values: {df[col].nunique()}"
        coldetails += f"\n  Null count: {df[col].isnull().sum()}"
    
    # Return dictionary with keys matching load_prompt parameters
    info = {
        "filename": csv_path.name,
        "numrows": len(df),
        "numcolumns": len(df.columns),
        "coldetails": coldetails,
        "sample": json.dumps(df.head(5).to_dict(orient='records'), indent=2)
    }
    
    return info

# === USAGE EXAMPLE ===

if __name__ == "__main__":
    # Example: analyze a CSV file
    csv_file = Path("your_data.csv")  # Replace with your CSV path
    config_path = Path("config.yaml")
    prompt_path=Path("prompt.md")
    # Create sample CSV
    df = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 
                'david@example.com', 'eve@example.com'],
        'order_date': pd.date_range('2024-01-01', periods=5),
        'amount': [100.50, 250.00, 75.25, 300.00, 150.75]
    })
    df.to_csv(csv_file, index=False)
    try:
    ### initialize the LLM client
        client = LLMClient(config_path)
        csv_info = load_csv_info(csv_file)
        print(f"Loading CSV: {csv_file.name}")
        # Create prompt
        #schema = CSVAnalysisResult.model_json_schema()
        #schema = Result.model_json_schema()
        prompt = create_analysis_prompt(csv_info,prompt_path,"Analyze")
        result = client.complete(prompt)
        print("="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        print(result)
        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file.name}' not found")
        print("\nTo test this code, create a sample CSV first:")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")