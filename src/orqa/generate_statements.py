import json
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
from statement_generator.StatementClient import LLMClientStatementGenerator
from statement_generator.prompting import _load_prompt, DatasetDescription
from pathlib import Path
import polars as pl
import time
from utils import load_datasets_metadata, load_dataset_info
import tempfile
load_dotenv()
# ============================================================================
# CARICAMENTO
# ============================================================================

def load_tasks(tasks_file):
    """Carica tasks_results.json e indicizza per (Q, R, task)"""
    tasks = {}
    with open(tasks_file) as f:
        for line in f:
            if line.strip():
                spec = json.loads(line)
                tasks[(spec["Q"], spec["R"], spec["task"])] = spec
    return tasks

def load_candidates(candidates_file):
    """Carica final_generation_candidates.json"""
    with open(candidates_file) as f:
        return json.load(f)

def make_match(task_spec, df_q, df_r, alias_q, alias_r):
    """Converte task spec in una stringa che descrive il match tra le tabelle"""
    task = task_spec["task"]
    
    # Dizionario per tracciare le colonne coinvolte
    involved_columns = {alias_q: set(), alias_r: set()}
    
    # UNION
    if task == "U":
        columns = task_spec.get("q_columns", [])
        if columns:
            q_cols = set(df_q.columns)
            r_cols = set(df_r.columns)
            missing_in_q = [col for col in columns if col not in q_cols]
            missing_in_r = [col for col in columns if col not in r_cols]
            
            if missing_in_q or missing_in_r:
                print(f"UNION skipped: columns not found")
                if missing_in_q:
                    print(f"   Missing in {alias_q}: {missing_in_q}")
                if missing_in_r:
                    print(f"   Missing in {alias_r}: {missing_in_r}")
                return None
            
            col_str = ", ".join(columns)
            involved_columns[alias_q].update(columns)
            involved_columns[alias_r].update(columns)
        else:
            common_cols = set(df_q.columns) & set(df_r.columns)
            if not common_cols:
                print(f"UNION skipped: no common columns between {alias_q} and {alias_r}")
                return None
            col_str = "All columns"
            involved_columns[alias_q].update(common_cols)
            involved_columns[alias_r].update(common_cols)
        
        return {
            "description": f"UNION: {alias_q} ∪ {alias_r} ON {col_str}",
            "columns": {k: list(v) for k, v in involved_columns.items()}
        }
    
    # JOIN
    elif task == "J":
        q_keys = task_spec["q_join_keys"] if "q_join_keys" in task_spec else [task_spec["q_join_key"]]
        r_keys = [df_r.columns[pos] for pos in task_spec["r_join_keys_pos"]] if "r_join_keys_pos" in task_spec else [df_r.columns[task_spec["r_join_key_pos"]]]
        
        involved_columns[alias_q].update(q_keys)
        involved_columns[alias_r].update(r_keys)
        
        join_conditions = [f"{alias_q}.{q_keys[i]} = {alias_r}.{r_keys[i]}" for i in range(len(q_keys))]
        join_str = " AND ".join(join_conditions)
        
        return {
            "description": f"JOIN: {alias_q} ⋈ {alias_r} ON {join_str}",
            "columns": {k: list(v) for k, v in involved_columns.items()}
        }
    
    # JOIN-CORRELATION
    elif task == "JC":
        q_key = df_q.columns[task_spec["q_key"]] if isinstance(task_spec["q_key"], int) else task_spec["q_key"]
        r_key = df_r.columns[task_spec["r_key"]] if isinstance(task_spec["r_key"], int) else task_spec["r_key"]
        q_target = df_q.columns[task_spec["q_target"]] if isinstance(task_spec["q_target"], int) else task_spec["q_target"]
        r_target = df_r.columns[task_spec["r_target"]] if isinstance(task_spec["r_target"], int) else task_spec["r_target"]
        
        involved_columns[alias_q].update([q_key, q_target])
        involved_columns[alias_r].update([r_key, r_target])
        
        return {
            "description": f"JOIN-CORRELATION: {alias_q} ⋈ {alias_r} ON {alias_q}.{q_key} = {alias_r}.{r_key} AND {alias_q}.{q_target} = {alias_r}.{r_target}",
            "columns": {k: list(v) for k, v in involved_columns.items()}
        }
    
    return None

def process_all(candidates_file, tasks_file, csv_folder, output_file="queries/matches.json"):
    """
    Processa tutti i path e genera i matches.
    """
    csv_folder = Path(csv_folder)
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    print("Caricamento...")
    tasks = load_tasks(tasks_file)
    candidates = load_candidates(candidates_file)
    print(f"✓ {len(tasks)} tasks, {len(candidates)} datasets")
    
    all_results = []
    
    for dataset_id, path_groups in candidates.items():
        for group in path_groups:
            for path in group["paths"]:
                matches = path["datasets"]
                operations = path["operation_type"]
                
                aliases = {f"Table_{i}": matches[i] for i in range(len(matches))}
                
                # Traccia colonne aggregate per tutte le tabelle
                all_columns = {f"Table_{i}": set() for i in range(len(matches))}
                
                path_matches = []
                for i in range(len(matches) - 1):
                    for op in operations:
                        key = (matches[i], matches[i+1], op)
                        if key in tasks:
                            task_spec = tasks[key]
                            df_q = pd.read_csv(csv_folder / f"{matches[i]}.csv", low_memory=False, sep=",")
                            df_r = pd.read_csv(csv_folder / f"{matches[i+1]}.csv", low_memory=False, sep=",")
                            match_result = make_match(task_spec, df_q, df_r, f"Table_{i}", f"Table_{i+1}")
                            
                            if match_result is not None:
                                path_matches.append(match_result["description"])
                                # Aggrega le colonne coinvolte
                                for table_alias, cols in match_result["columns"].items():
                                    all_columns[table_alias].update(cols)
                
                if path_matches and len(path_matches) >= (len(aliases)-1):
                    all_results.append({
                        "dataset_id": dataset_id,
                        "aliases": aliases,
                        "matches": path_matches,
                        "columns_by_table": {k: list(v) for k, v in all_columns.items()}
                    })
                else: 
                    print("Filtered the match because it's not as exhaustive.")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Salvati {len(all_results)} risultati in {output_path}")
    return all_results



def prepare_queries(candidates_file,tasks_file,csv_folder,output_dir):
    results = process_all(
        candidates_file=candidates_file,
        tasks_file=tasks_file,
        csv_folder=csv_folder,
        output_file=f"{output_dir}/matches.json"
    )

import sys

def create_statements(csv_folder, output_dir, kind="PANDAS", max_cols=15, metadata=None):
    client = LLMClientStatementGenerator(Path("statement_generator/litellm.yaml"))
    with open(f"{output_dir}/matches.json", 'r', encoding='utf-8') as f:
        all_matches = json.load(f)
    
    descriptor = DatasetDescription()
    output_file = Path(output_dir) / f"validated_queries({kind}).json"
    
    # Initialize or load existing results
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = {"successful": [], "failed": [], "total_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    
    total_matches = len(all_matches)
    
    for idx, result in enumerate(all_matches):
        start_time = time.time()
        
        # Build dataframes and descriptions
        dataframes = []
        table_names = []
        table_descriptions = []
        aliases = result["aliases"]
        involved_columns = result["columns_by_table"]
        present_metadata = "present"
        
        for alias, dataset in aliases.items():
            specific_cols = involved_columns[alias]
            df = pd.read_csv(f"{csv_folder}/{dataset}.csv", sep=",")

            # Puts a max number of columns
            if len(df.columns) > max_cols:
                other_cols = [c for c in df.columns if c not in specific_cols]
                cols_to_keep = list(specific_cols) + other_cols[:max_cols - len(specific_cols)]
                df = df[cols_to_keep]
            
            dataframes.append(df)
            table_names.append(alias)
            metadata = datasets_metadata.get(dataset, "")
            if metadata is not None:
                present_metadata = "absent"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                df.to_csv(tmp_file.name, index=False)
                tmp_path = tmp_file.name

            try:
                dataset_info, _ = load_dataset_info(Path(tmp_path))
                table_descriptions.append(
                    f"### Dataset Alias: {alias} \n{descriptor.update(dataset, df.shape[0], df.shape[1], metadata, dataset_info['columns_details'], df.head(3))}"
                )
            finally:
                os.unlink(tmp_path)  

        prompt = _load_prompt(
            "statement_generator/prompt.md",
            "SQLGeneration",
            matches="\n".join(result["matches"]),
            table="\n".join(table_descriptions),
            aliases=aliases
        )
        
        queries_result, usage = client.complete(prompt, dataframes, table_names, typology=kind)
        elapsed_time = time.time() - start_time
        
        # Update total usage
        all_results["total_usage"]["prompt_tokens"] += usage["prompt_tokens"]
        all_results["total_usage"]["completion_tokens"] += usage["completion_tokens"]
        all_results["total_usage"]["total_tokens"] += usage["total_tokens"]
        
        # Prepare result entry
        entry = {
            "match_index": idx,
            "aliases": result["aliases"],
            "matches": result["matches"],
            "generation_time_seconds": round(elapsed_time, 2),
            "usage": usage,
            "metadata": present_metadata,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Check if generation was successful
        if queries_result["queries"]:
            entry["queries"] = queries_result["queries"]
            entry["status"] = "success"
            all_results["successful"].append(entry)
        else:
            entry["status"] = "failed"
            entry["error"] = "No queries generated after max retries"
            all_results["failed"].append(entry)
        
        # Save incrementally after each match
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Update terminal message (overwrites previous line)
        progress = idx + 1
        success_count = len(all_results["successful"])
        failed_count = len(all_results["failed"])
        total_tokens = all_results["total_usage"]["total_tokens"]
        
        # Clear line and print updated stats
        sys.stdout.write("\r" + " " * 120)  # Clear line
        sys.stdout.write(
            f"\r[{progress}/{total_matches}] "
            f"✓ Success: {success_count} | "
            f"✗ Failed: {failed_count} | "
            f"🎫 Tokens: {total_tokens:,} | "
            f"⏱️ Last: {elapsed_time:.1f}s"
        )
        sys.stdout.flush()
    
    # New line after completion
    print()
    print(f"\n💾 Results saved to {output_file}")
    return all_results

if __name__ == "__main__":
    datasets_metadata = load_datasets_metadata(Path(r"D:\uk\metadata\metadata.json"))
    #prepare_queries(r"D:\uk\candidates_discovery\final_generation_candidates.json",r"D:\uk\candidates_discovery\tasks_results.json",r"D:\uk\datasets\csv","queries")
    create_statements(r"D:\uk\datasets\csv","queries","SQL",10,datasets_metadata)
    