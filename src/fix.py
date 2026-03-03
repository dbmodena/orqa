import json
import re

input_path = r"D:\orqa\socrata\nyc\candidates_discovery\query_candidates.json"
output_path = r"D:\orqa\socrata\nyc\candidates_discovery\query_candidates_fixed.json"

def add_suffixes_to_merge(code):
    pattern = r"pd\.merge\s*\(\s*Table_(\d+)\s*,\s*Table_(\d+)(.*?)\)"
    
    def replacer(match):
        left = match.group(1)
        right = match.group(2)
        middle = match.group(3)
        
        # Se già contiene suffixes, non duplicare
        if "suffixes=" in middle:
            return match.group(0)
        
        return (
            f"pd.merge(Table_{left}, Table_{right}"
            f"{middle}, suffixes=('_T{left}', '_T{right}'))"
        )
    
    return re.sub(pattern, replacer, code, flags=re.DOTALL)

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "PANDAS_matches" in item:
        fixed_matches = []
        for code in item["PANDAS_matches"]:
            fixed_matches.append(add_suffixes_to_merge(code))
        item["PANDAS_matches"] = fixed_matches

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("File corretto salvato come:", output_path)