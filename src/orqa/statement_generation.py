import json
from pathlib import Path
import pandas as pd
from .agent.agent import StatementGenerationAgent
from .utils import load_datasets_metadata, load_dataset_info,save_json,load_json
from conf import OrQAConfig
from dataclasses import dataclass, field


def load_tasks(tasks_file: Path) -> dict:
    """Load tasks_results.json and index by (Q, R, task)."""
    tasks = {}
    with open(tasks_file) as f:
        for line in f:
            if line.strip():
                spec = json.loads(line)
                tasks[(spec["Q"], spec["R"], spec["task"])] = spec
    return tasks


# ── Match builders ────────────────────────────────────────────────────────────

def _make_union_match(task_spec, df_q, df_r, alias_q, alias_r):
    q_columns = task_spec.get("q_columns", [])
    r_columns = task_spec.get("r_columns", [])
    involved = {alias_q: set(q_columns), alias_r: set(r_columns)}
    col_str = f"{q_columns} / {r_columns}" if q_columns or r_columns else "All columns"
    return {
        "description": f"UNION: {alias_q} ∪ {alias_r} ON {col_str}",
        "pandas_expr": f"concat([{alias_q}[{q_columns}], {alias_r}[{r_columns}]], ignore_index=True)",
        "columns": {k: list(v) for k, v in involved.items()},
    }


def _make_join_match(task_spec, df_r, alias_q, alias_r):
    q_keys = task_spec.get("q_columns", [])
    r_keys = task_spec.get("r_columns", [])
    involved = {alias_q: set(q_keys), alias_r: set(r_keys)}
    conditions = " AND ".join(f"{alias_q}.{q_keys[i]} = {alias_r}.{r_keys[i]}" for i in range(len(q_keys)))
    return {
        "description": f"JOIN: {alias_q} ⋈ {alias_r} ON {conditions}",
        "pandas_expr": f"merge(left={alias_q}, right={alias_r}, left_on={q_keys}, right_on={r_keys}, suffixes=('_{alias_q}', '_{alias_r}'))",
        "columns": {k: list(v) for k, v in involved.items()},
    }


def _make_join_correlation_match(task_spec, df_q, df_r, alias_q, alias_r):
    q_key = task_spec["q_key"]
    r_key = task_spec["r_key"]
    q_target = task_spec["q_target"]
    r_target = task_spec["r_target"]
    involved = {alias_q: {q_key, q_target}, alias_r: {r_key, r_target}}
    return {
        "description": (
            f"JOIN-CORRELATION: {alias_q} ⋈ {alias_r} "
            f"ON {alias_q}.{q_key} = {alias_r}.{r_key} "
            f"AND {alias_q}.{q_target} = {alias_r}.{r_target}"
        ),
        "pandas_expr": (
            f"pd.merge({alias_q}, {alias_r}, left_on=['{q_key}', '{q_target}'], right_on=['{r_key}', '{r_target}'],suffixes=('_{alias_q}', '_{alias_r}'))"
        ),
        "columns": {k: list(v) for k, v in involved.items()},
    }

def make_match(task_spec, df_q, df_r, alias_q, alias_r):
    task = task_spec["task"]
    builder = _MATCH_BUILDERS.get(task)
    if builder is None:
        return None
    if task == "J":
        return builder(task_spec, df_r, alias_q, alias_r)
    return builder(task_spec, df_q, df_r, alias_q, alias_r)


_MATCH_BUILDERS = {"U": _make_union_match, "J": _make_join_match, "JC": _make_join_correlation_match}


# ── Match processing ──────────────────────────────────────────────────────────

def process_path(path: dict, tasks: dict, csv_folder: Path) -> dict | None:
    datasets = path["datasets"]
    operations = path["operation_type"]
    aliases = {f"Table_{i}": datasets[i] for i in range(len(datasets))}
    all_columns = {f"Table_{i}": set() for i in range(len(datasets))}
    path_matches = []
    path_pandas = []

    for i in range(len(datasets) - 1):
        pair_matches = []
        pair_pandas = []

        df_i = pd.read_csv(csv_folder / f"{datasets[i]}.csv", low_memory=False)
        df_i1 = pd.read_csv(csv_folder / f"{datasets[i + 1]}.csv", low_memory=False)

        for op in operations:
            key = (datasets[i], datasets[i + 1], op)
            reversed_key = False
            if key not in tasks:
                key = (datasets[i + 1], datasets[i], op)
                reversed_key = True
            if key not in tasks:
                continue

            if reversed_key:
                result = make_match(tasks[key], df_i1, df_i, f"Table_{i + 1}", f"Table_{i}")
            else:
                result = make_match(tasks[key], df_i, df_i1, f"Table_{i}", f"Table_{i + 1}")

            if result:
                pair_matches.append(result["description"])
                pair_pandas.append(result["pandas_expr"])
                for alias, cols in result["columns"].items():
                    all_columns[alias].update(cols)

        if not pair_matches:
            print(f"Filtered: no match found for Table_{i} ↔ Table_{i + 1}.")
            return None

        path_matches.extend(pair_matches)
        path_pandas.extend(pair_pandas)

    return {
        "aliases": aliases,
        "SQL_matches": path_matches,
        "PANDAS_matches": path_pandas,
        "columns_by_table": {k: list(v) for k, v in all_columns.items()},
    }


def process_all_candidates(candidates_file: Path, tasks_file: Path,
                            csv_folder: Path, output_file: Path) -> list[dict]:
    """Process all candidate paths and write matches.json."""
    print("Loading...")
    tasks = load_tasks(tasks_file)
    with open(candidates_file) as f:
        candidates = json.load(f)
    print(f"✓ {len(tasks)} tasks, {len(candidates)} datasets")

    results = []
    for dataset_id, path_groups in candidates.items():
        for group in path_groups:
            for path in group["paths"]:
                record = process_path(path, tasks, csv_folder)
                if record:
                    results.append({"dataset_id": dataset_id, **record})

    save_json(results, output_file)
    print(f"✓ Saved {len(results)} results to {output_file}")
    return results



# ── Statement generation ──────────────────────────────────────────────────────

def _build_match_inputs(
    match: dict, csv_folder: Path, datasets_metadata: dict, extension: str
) -> tuple[list[Path], list[str], list[dict], set]:
    """Unpack a match record into agent inputs: (dataset_paths, aliases, metadatas, involved_cols).
    Candidate for utils.py if reused elsewhere.
    """
    dataset_paths, metadatas = [], []
    aliases = {}
    involved_cols={}
    for alias, dataset in match["aliases"].items():
        dataset_paths.append(csv_folder / f"{dataset}.{extension}")
        aliases[f"{alias}"]=dataset 
        metadatas.append(datasets_metadata.get(dataset))
        involved_cols[alias]=match["columns_by_table"].get(alias, [])
    return dataset_paths, aliases, metadatas, involved_cols


import sys

def create_statements(
    config_path: Path, csv_folder: Path, candidates_path: Path, output_path: Path,
    kind: str = "PANDAS", max_cols: int = 15,
    datasets_metadata: Path = None, extension: str = "csv", bad_tokens: list = []
) -> dict:
    datasets_metadata = datasets_metadata or {}

    all_matches = load_json(candidates_path)
    output_file = output_path
    results = load_json(output_file) if output_file.exists() else {}

    # Ensure the kind layer exists, preserving other kinds if present
    if kind not in results:
        results[kind] = {}

    successes = 0
    failures = 0
    total = len(all_matches)

    for idx, match in enumerate(all_matches):
        agent = StatementGenerationAgent(config_path, kind, bad_tokens)
        sys.stdout.write(
            f"\r[{idx + 1}/{total}]  ✅ Successes: {successes}   ❌ Failures: {failures}   "
        )
        dataset_paths, aliases, metadatas, involved_cols = _build_match_inputs(
            match, csv_folder, datasets_metadata, extension
        )
        content = agent.generate_statements(
            dataset_paths, aliases, kind, match[f"{kind}_matches"], involved_cols, metadatas,
            max_cols, sample_size=5,
        )

        result = content["result"]
        tokens = content["token_usage"]
        status = "success" if result['queries'] != [] else "failure"

        if result['queries'] != []:
            successes += 1
        else:
            failures += 1

        # Save under the kind layer
        results[kind][str(idx)] = {"status": status, "data": result, "tokens": tokens, "tables": aliases}
        save_json(results, output_file)
        sys.stdout.flush()

    print(f"\nResults saved to {output_file}")
    return results

# ── Entry point ───────────────────────────────────────────────────────────────
from orqa.candidates_generation import generate_random_walks

def generate_statements(cfg:OrQAConfig):
    metadata = load_datasets_metadata(
        cfg.metadata_path.joinpath("metadata.json"),
        None,  # [s[3 if len(s) == 4 else 2] for s in seed_datasets],
        source=cfg.source,
    )
    
    #cfg.statement_generation.threshold
    #cfg.statement_generation.join_score
    
    #cfg.statement_generation.union_score
    ### first we generate the random walks
    #generate_random_walks(cfg)
    #print(cfg.candidates_discovery.candidates_path)
    #cfg.candidates_discovery.proposed_tasks_path
    #process_all_candidates(cfg.candidates_discovery.candidates_path, cfg.candidates_discovery.tasks_results_path, cfg.datasets_path, cfg.statement_generation.query_candidates_path)
    create_statements(cfg.llm_config_path.joinpath("litellm.yaml"),cfg.datasets_path, cfg.statement_generation.query_candidates_path,cfg.statement_generation.queries_path, cfg.statement_generation.kind,cfg.statement_generation.max_cols, datasets_metadata=metadata,bad_tokens=cfg.statement_generation.bad_tokens)
    