import json
from pathlib import Path

import pandas as pd
from valentine import valentine_match
from valentine.algorithms import Coma

DOCUMENT_TYPE = "csv"
THRESHOLD = 0.5
MAX_WORKERS = 12


### calcs the Valentine averages
def calc_avg_valentine_scores(
    scores,
    q_columns: list = None,
    r_columns: list = None,
    q_key: str = None,
    q_target: str = None,
    r_key: str = None,
    r_target: str = None,
):
    averageAll = sum(scores.values()) / len(scores)
    specificAvg = 0
    filteredValues = []
    if q_columns is not None and r_columns is not None:
        filteredValues = [
            value
            for ((_, col1), (_, col2)), value in scores.items()
            if col1 in q_columns and col2 in r_columns
        ]
        specificAvg = sum(filteredValues) / len(filteredValues) if filteredValues else 0
    elif q_columns is not None:
        filteredValues = [
            value
            for ((_, col1), (_, col2)), value in scores.items()
            if col1 in q_columns and col2 in q_columns
        ]
        specificAvg = sum(filteredValues) / len(filteredValues) if filteredValues else 0
    elif (
        q_key is not None
        and q_target is not None
        and r_key is not None
        and r_target is not None
    ):
        filteredValues = [
            value
            for ((_, col1), (_, col2)), value in scores.items()
            if (col1 == q_key and col2 == r_key)
            or (col1 == q_target and col2 == r_target)
        ]
        specificAvg = sum(filteredValues) / len(filteredValues) if filteredValues else 0
    return averageAll, specificAvg


def prepare_datasets(dataset_dir, Q, R):
    try:
        l_path = dataset_dir.joinpath(f"{Q}.{DOCUMENT_TYPE}")
        r_path = dataset_dir.joinpath(f"{R}.{DOCUMENT_TYPE}")
        l_table = pd.read_csv(l_path, thousands=",").dropna(axis=1, how="all")
        r_table = pd.read_csv(r_path, thousands=",").dropna(axis=1, how="all")
        return l_table, r_table
    except Exception as e:
        print(e)
        return None


def apply_valentine_matcher(
    dataset_dir,
    task,
    Q,
    R,
    q_columns: list = None,
    r_indices: list = None,
    q_key: str = None,
    q_target: str = None,
    r_key: str = None,
    r_target: str = None,
) -> tuple | None:
    try:
        l_table, r_table = prepare_datasets(dataset_dir, Q, R)
        matcher = Coma(use_instances=True)
        matches = valentine_match(l_table, r_table, matcher)
        print("= L Table" + "=" * 50)
        print(l_table.columns)
        print("= R Table" + "=" * 50)
        print(r_table.columns)

        print(matches)
        if task == "U":
            r_columns = r_table.columns[r_indices].tolist()
            avg, avg_q = calc_avg_valentine_scores(matches, q_columns, r_columns)
            print(f"average overall {avg}")
            print(f"average score columns involved {avg_q}")
            return matches, avg, avg_q
        elif task == "MJ":
            avg, avg_q = calc_avg_valentine_scores(matches, q_columns)
            print(f"average overall {avg}")
            print(f"average score columns involved {avg_q}")
            return matches, avg, avg_q
        elif task == "JC":
            r_key_label = r_table.columns[r_key]
            r_target_label = r_table.columns[r_target]
            print(r_key_label)
            print(r_target_label)
            avg, avg_q = calc_avg_valentine_scores(
                matches,
                q_key=q_key,
                q_target=q_target,
                r_key=r_key_label,
                r_target=r_target_label,
            )
            print(f"average overall {avg}")
            print(f"average score columns involved {avg_q}")
            return matches, avg, avg_q
    except Exception as e:
        print(e)


def find_matches(matches_path: Path, dataset_dir: Path):
    try:
        with open(matches_path) as f:
            data = json.load(f)
            # Add all nodes first
            for entry in data:
                print(f"Analyzing the overlap between {entry['Q']} and {entry['R']}")
                if entry["task"] == "U":
                    print("smell ya later")
                    # apply_valentine_matcher(dataset_dir,"U",entry["Q"], entry["R"], entry["q_columns"])
                elif entry["task"] == "MJ":
                    print("coffee break!!!")
                    # apply_valentine_matcher(dataset_dir,"MJ",entry["Q"], entry["R"], entry["q_join_keys"], entry["r_join_keys_pos"])
                elif entry["task"] == "JC":
                    apply_valentine_matcher(
                        dataset_dir,
                        "JC",
                        entry["Q"],
                        entry["R"],
                        q_key=entry["q_key"],
                        q_target=entry["q_target"],
                        r_key=entry["r_key"],
                        r_target=entry["r_target"],
                    )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # table_id = "2019-March-return__3f436d14-4e17-476c-a3e4-66d18e7f6c90"
    # operation = "JC"
    path_matches = Path(MATCHES_JSON)
    path_dataset = Path(DATASET_DIR)
    find_matches(path_matches, path_dataset)
