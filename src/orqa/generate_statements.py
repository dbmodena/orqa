"""
Generatore Query Semplificato
Usa final_generation_candidates.json + tasks_results.json
"""
import json
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
from statement_generator.StatementClient import LLMClientStatementGenerator
from statement_generator.prompting import _load_prompt, DatasetDescription
from pathlib import Path
#import pandas as pd
import polars as pl
import json
import time
from utils import load_datasets_metadata, load_dataset_info

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
    
    # UNION
    if task == "U":
        columns = task_spec.get("q_columns", [])
        if columns:
            q_cols = set(df_q.columns)
            r_cols = set(df_r.columns)
            missing_in_q = [col for col in columns if col not in q_cols]
            missing_in_r = [col for col in columns if col not in r_cols]
            
            # Se mancano colonne in uno dei due DataFrame, ritorna None
            if missing_in_q or missing_in_r:
                print(f"⚠️ UNION skipped: columns not found")
                if missing_in_q:
                    print(f"   Missing in {alias_q}: {missing_in_q}")
                if missing_in_r:
                    print(f"   Missing in {alias_r}: {missing_in_r}")
                return None
            
            col_str = ", ".join(columns)
        else:
            # Se non ci sono colonne specificate, usa tutte le colonne comuni
            common_cols = set(df_q.columns) & set(df_r.columns)
            if not common_cols:
                print(f"⚠️ UNION skipped: no common columns between {alias_q} and {alias_r}")
                return None
            col_str = "tutte le colonne comuni"
        
        return f"UNION: {alias_q} ∪ {alias_r} ON {col_str}"
    
    # JOIN
    elif task == "J":
        q_keys = task_spec["q_join_keys"] if "q_join_keys" in task_spec else [task_spec["q_join_key"]]
        r_keys = [df_r.columns[pos] for pos in task_spec["r_join_keys_pos"]] if "r_join_keys_pos" in task_spec else [df_r.columns[task_spec["r_join_key_pos"]]]
        
        # Crea le coppie di join
        join_conditions = [f"{alias_q}.{q_keys[i]} = {alias_r}.{r_keys[i]}" for i in range(len(q_keys))]
        join_str = " AND ".join(join_conditions)
        
        return f"JOIN: {alias_q} ⋈ {alias_r} ON {join_str}"
    
    # JOIN-CORRELATION
    elif task == "JC":
        q_key = df_q.columns[task_spec["q_key"]] if isinstance(task_spec["q_key"], int) else task_spec["q_key"]
        r_key = df_r.columns[task_spec["r_key"]] if isinstance(task_spec["r_key"], int) else task_spec["r_key"]
        q_target = df_q.columns[task_spec["q_target"]] if isinstance(task_spec["q_target"], int) else task_spec["q_target"]
        r_target = df_r.columns[task_spec["r_target"]] if isinstance(task_spec["r_target"], int) else task_spec["r_target"]
        
        return f"JOIN-CORRELATION: {alias_q} ⋈ {alias_r} ON {alias_q}.{q_key} = {alias_r}.{r_key} AND {alias_q}.{q_target} = {alias_r}.{r_target}"
    
    return f"UNKNOWN TASK: {task}"

def process_all(candidates_file, tasks_file, csv_folder, output_file="queries/matches.json"):
    """
    Processa tutti i path e genera i matches.
    
    Args:
        candidates_file: path a final_generation_candidates.json
        tasks_file: path a tasks_results.json  
        csv_folder: cartella con i CSV
        output_file: file JSON di output
    """
    # Setup
    csv_folder = Path(csv_folder)
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    # Carica dati
    print("Caricamento...")
    tasks = load_tasks(tasks_file)
    candidates = load_candidates(candidates_file)
    print(f"✓ {len(tasks)} tasks, {len(candidates)} datasets")
    
    # Risultati finali
    all_results = []
    
    for dataset_id, path_groups in candidates.items():
        for group in path_groups:
            for path in group["paths"]:
                matches = path["datasets"]
                operations = path["operation_type"]
                
                # Crea aliases
                aliases = {f"Table_{i}": matches[i] for i in range(len(matches))}
                # Trova i match
                path_matches = []
                for i in range(len(matches) - 1):
                    for op in operations:
                        key = (matches[i], matches[i+1], op)
                        if key in tasks:
                            task_spec = tasks[key]
                            df_q = pd.read_csv(csv_folder / f"{matches[i]}.csv", low_memory=False,sep=",")
                            df_r = pd.read_csv(csv_folder / f"{matches[i+1]}.csv", low_memory=False,sep=",")
                            match_str = make_match(task_spec, df_q, df_r, f"Table_{i}", f"Table_{i+1}")
                            if match_str is not None:
                               path_matches.append(match_str)
                
                # Salva risultato per questo path
                if path_matches:
                    all_results.append({
                        "dataset_id": dataset_id,
                        "aliases": aliases,
                        "matches": path_matches
                    })
    
    # Salva JSON
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

def create_statements(csv_folder,output_dir,kind="PANDAS",metadata=None):
    client = LLMClientStatementGenerator(Path("statement_generator/litellm.yaml"))
    with open(f"{output_dir}/matches.json", 'r', encoding='utf-8') as f:
        all_matches = json.load(f)
    
    descriptor = DatasetDescription()
    output_file = Path(output_dir) / "validated_queries.json"
    
    # Initialize or load existing results
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = {"successful": [], "failed": [], "total_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    
    for idx, result in enumerate(all_matches[:50]):
        print(f"\n{'='*60}")
        print(f"Processing match {idx + 1}/{len(all_matches)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Build dataframes and descriptions
        dataframes = []
        table_names = []
        #table_descriptions = []
        aliases = result["aliases"]
        for alias, dataset in aliases.items():
            df = pd.read_csv(f"{csv_folder}/{dataset}.csv",sep=",")
            print(df.head())
            dataframes.append(df)
            table_names.append(alias)
            #metadata = datasets_metadata[dataset]
            #dataset_info,_ = load_dataset_info(Path(f"{csv_folder}/{dataset}.csv"))
            #table_descriptions.append(
            #    descriptor.update(alias, df.shape[0], df.shape[1], '', '', df.head(3))
            #)
        # Generate prompt and queries
        prompt = _load_prompt(
            "statement_generator/prompt.md",
            "PandasCodeGeneration",
            matches=result["matches"],
            table=""
            #table="\n".join(table_descriptions)
        )
        print(dataframes)
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
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        # Check if generation was successful
        if queries_result["queries"]:
                entry["queries"] = queries_result["queries"]
                entry["status"] = "success"
                all_results["successful"].append(entry)
                print(f"✓ Success: {len(queries_result['queries'])} queries generated in {elapsed_time:.2f}s")
        else:
                entry["status"] = "failed"
                entry["error"] = "No queries generated after max retries"
                all_results["failed"].append(entry)
                print(f"✗ Failed: No queries generated in {elapsed_time:.2f}s")
            
            # Save incrementally after each match
        with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print(f"💾 Results saved to {output_file}")
    return all_results

if __name__ == "__main__":
    datasets_metadata = load_datasets_metadata(r"D:\uk\metadata\metadata.json",None)
    prepare_queries(r"D:\uk\candidates_discovery\final_generation_candidates.json",r"D:\uk\candidates_discovery\tasks_results.json",r"D:\uk\datasets\csv","queries")
    create_statements(r"D:\uk\datasets\csv","queries",datasets_metadata)
    