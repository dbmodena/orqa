import json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from .agent.agent import StatementGenerationAgent
from .utils import load_datasets_metadata, load_dataset_info,save_json,load_json

load_dotenv()


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
    columns = task_spec.get("q_columns", [])
    involved = {alias_q: set(), alias_r: set()}

    if columns:
        missing_q = [c for c in columns if c not in df_q.columns]
        missing_r = [c for c in columns if c not in df_r.columns]
        if missing_q or missing_r:
            print(f"UNION skipped: columns not found")
            if missing_q: print(f"  Missing in {alias_q}: {missing_q}")
            if missing_r: print(f"  Missing in {alias_r}: {missing_r}")
            return None
        involved[alias_q].update(columns)
        involved[alias_r].update(columns)
        col_str = ", ".join(columns)
    else:
        common = set(df_q.columns) & set(df_r.columns)
        if not common:
            print(f"UNION skipped: no common columns between {alias_q} and {alias_r}")
            return None
        involved[alias_q].update(common)
        involved[alias_r].update(common)
        col_str = "All columns"

    return {
        "description": f"UNION: {alias_q} ∪ {alias_r} ON {col_str}",
        "columns": {k: list(v) for k, v in involved.items()},
    }


def _make_join_match(task_spec, df_r, alias_q, alias_r):
    q_keys = task_spec["q_join_keys"] if "q_join_keys" in task_spec else [task_spec["q_join_key"]]
    r_keys = (
        [df_r.columns[pos] for pos in task_spec["r_join_keys_pos"]]
        if "r_join_keys_pos" in task_spec
        else [df_r.columns[task_spec["r_join_key_pos"]]]
    )
    involved = {alias_q: set(q_keys), alias_r: set(r_keys)}
    conditions = " AND ".join(f"{alias_q}.{q_keys[i]} = {alias_r}.{r_keys[i]}" for i in range(len(q_keys)))
    return {
        "description": f"JOIN: {alias_q} ⋈ {alias_r} ON {conditions}",
        "columns": {k: list(v) for k, v in involved.items()},
    }


def _make_join_correlation_match(task_spec, df_q, df_r, alias_q, alias_r):
    def resolve(key, df):
        v = task_spec[key]
        return df.columns[v] if isinstance(v, int) else v

    q_key, r_key = resolve("q_key", df_q), resolve("r_key", df_r)
    q_target, r_target = resolve("q_target", df_q), resolve("r_target", df_r)
    involved = {alias_q: {q_key, q_target}, alias_r: {r_key, r_target}}
    return {
        "description": (
            f"JOIN-CORRELATION: {alias_q} ⋈ {alias_r} "
            f"ON {alias_q}.{q_key} = {alias_r}.{r_key} "
            f"AND {alias_q}.{q_target} = {alias_r}.{r_target}"
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
    """Process a single candidate path into a match record, or None if incomplete."""
    datasets = path["datasets"]
    operations = path["operation_type"]
    aliases = {f"Table_{i}": datasets[i] for i in range(len(datasets))}
    all_columns = {f"Table_{i}": set() for i in range(len(datasets))}
    path_matches = []

    for i in range(len(datasets) - 1):
        for op in operations:
            key = (datasets[i], datasets[i + 1], op)
            if key not in tasks:
                continue
            df_q = pd.read_csv(csv_folder / f"{datasets[i]}.csv", low_memory=False)
            df_r = pd.read_csv(csv_folder / f"{datasets[i + 1]}.csv", low_memory=False)
            result = make_match(tasks[key], df_q, df_r, f"Table_{i}", f"Table_{i + 1}")
            if result:
                path_matches.append(result["description"])
                for alias, cols in result["columns"].items():
                    all_columns[alias].update(cols)

    if not path_matches or len(path_matches) < len(aliases) - 1:
        print("Filtered: match not exhaustive.")
        return None

    return {
        "aliases": aliases,
        "matches": path_matches,
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
    aliases = ""
    involved_cols: set = set()
    for alias, dataset in match["aliases"].items():
        dataset_paths.append(csv_folder / f"{dataset}.{extension}")
        aliases +=f"\n'{alias}':{dataset}" 
        metadatas.append(datasets_metadata.get(dataset))
        involved_cols.update(match["columns_by_table"].get(alias, []))
    return dataset_paths, aliases, metadatas, involved_cols


def create_statements(
    config_path: Path, csv_folder: Path, output_dir: Path,
    kind: str = "PANDAS", max_cols: int = 15,
    datasets_metadata: dict = None, extension: str = "csv",
) -> dict:
    datasets_metadata = datasets_metadata or {}
    csv_folder, output_dir = Path(csv_folder), Path(output_dir)

    agent = StatementGenerationAgent(config_path, kind)
    all_matches = load_json(output_dir / "matches.json")
    output_file = output_dir / f"validated_queries({kind}).json"
    results = load_json(output_file) if output_file.exists() else {}

    for match in all_matches:
        dataset_paths, aliases, metadatas, involved_cols = _build_match_inputs(
            match, csv_folder, datasets_metadata, extension
        )
        print( match["matches"])
        content = agent.generate_statements(
            dataset_paths, aliases, kind, match["matches"], involved_cols, metadatas,
            max_cols, sample_size=5,
        )
        results = content["result"]
        save_json(results, output_file)

    print(f"\nResults saved to {output_file}")
    return results

def generate_statements(
    config_path: Path, candidates_file: Path, tasks_file: Path,
    csv_folder: Path, metadata_file: Path, output_dir: Path,
    kind: str = "SQL", max_cols: int = 10, skip_matching: bool = False,
) -> dict:
    """Run the full pipeline: match building + statement generation."""
    if not skip_matching:
        process_all_candidates(candidates_file, tasks_file, csv_folder, output_dir / "matches.json")
    return create_statements(
        config_path, csv_folder, output_dir, kind=kind,
        max_cols=max_cols, datasets_metadata=load_datasets_metadata(metadata_file),
    )

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CSV_FOLDER = Path(r"D:\uk\datasets\csv")
    OUTPUT_DIR = Path(r"C:\Users\39377\Documents\GitHub\orqa\src\orqa\queries")
    METADATA_FILE = Path(r"D:\uk\metadata\metadata.json")
    CANDIDATES_FILE = Path(r"D:\uk\candidates_discovery\final_generation_candidates.json")
    TASKS_FILE = Path(r"D:\uk\candidates_discovery\tasks_results.json")
    CONFIG_PATH = Path(r"C:\Users\39377\Documents\GitHub\orqa\conf\llm\litellm.yaml")
    datasets_metadata = load_datasets_metadata(METADATA_FILE)

    # Step 1 — build matches (comment out if already done)
    # process_all_candidates(CANDIDATES_FILE, TASKS_FILE, CSV_FOLDER, OUTPUT_DIR / "matches.json")

    # Step 2 — generate statements
    create_statements(CONFIG_PATH,CSV_FOLDER, OUTPUT_DIR, kind="SQL", max_cols=10, datasets_metadata=datasets_metadata)