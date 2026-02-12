import time
import pprint
from tqdm import tqdm
from functools import lru_cache
import os
import sys
import json
from pathlib import Path
import polars as pl

sys.path.append("./src")
from orqa.schema_matching.valentine_matcher import instantiate_matcher, schema_matching
from orqa.utils import remove_null_columns


SEED = 42


@lru_cache(64)
def _load_dataframe(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    return df.sample(min(1000, df.height), seed=SEED)


def load_dataframe(path: Path) -> pl.DataFrame:
    return _load_dataframe(path)


def evaluate_matches(
    blend_matches_json: Path, datasets_path: Path, statistics_path: Path
):
    print("Loading BLEND matches JSON file...")
    with open(blend_matches_json, "r") as file:
        blend_matches = [json.loads(line) for line in file.readlines()]
    print("Done.")

    print(f"Loaded {len(blend_matches)} matches.")

    records = []
    matcher_name = "coma"
    stats_filename = statistics_path / f"schema_matching_{matcher_name}.csv"

    matcher = instantiate_matcher(matcher_name, use_instances=False)

    for blend_match in tqdm(blend_matches, desc="Scanning BLEND matches"):
        Q_name = blend_match["Q"]
        R_name = blend_match["R"]
        task = blend_match["task"]

        Q = load_dataframe(datasets_path / f"{Q_name}.csv")
        R = load_dataframe(datasets_path / f"{R_name}.csv")

        print("\n" + f" Q: {Q_name} - R: {R_name} - Task: {task} ".center(150, "-"))
        print(f"Q schema: {Q.columns}")
        print(f"R schema: {R.columns}")

        q_columns = r_columns = None
        q_key = r_key = None
        q_target = r_target = None

        match task:
            case "U":
                q_columns = blend_match["q_columns"]
            case "J" | "MJ":
                try:
                    q_columns = blend_match["q_join_keys"]
                    r_columns_pos = blend_match["r_join_keys_pos"]
                except KeyError:
                    q_columns = [blend_match["q_join_key"]]
                    r_columns_pos = [blend_match["r_join_key_pos"]]

                r_columns = [R.columns[i] for i in r_columns_pos]
            case "JC":
                q_key = blend_match["q_key"]
                q_target = blend_match["q_target"]

                print(blend_match)
                r_key_pos = blend_match["r_key"]
                r_target_pos = blend_match["r_target"]
                r_key = R.columns[r_key_pos]
                r_target = R.columns[r_target_pos]

        if q_key is None:
            print(f"Q projection: {q_columns}")
            print(f"R projection: {r_columns}")
        else:
            print(f"Q key: {q_key} - Q target: {q_target}")
            print(f"R key: {r_key} - R target: {r_target}")

        try:
            match_t = time.time()
            matches, global_avg, spec_avg = schema_matching(
                matcher,
                task,
                Q.to_pandas(),
                R.to_pandas(),
                q_columns,  # ty: ignore
                r_columns,
                q_key,
                r_key,
                q_target,
                r_target,
            )
            match_t = time.time() - match_t
        except Exception as exc:
            print(f"Error with Q={Q_name}, R={R_name}: {exc}")
            raise exc
            # continue

        print(f"Global SM score: {global_avg:.2f}")
        print(f"Specific SM score: {spec_avg:.2f}")
        pprint.pprint(matches)

        blend_match |= {
            "#Q_schema": len(Q.columns),
            "#R_schema": len(R.columns),
            "matcher": matcher_name,
            "#Q_req_columns": len(q_columns) if q_columns else None,
            "#R_req_columns": len(r_columns) if r_columns else None,
            f"sm_{matcher_name}_matches": [(m[0][1], m[1][1]) for m in matches],
            f"sm_{matcher_name}_n_matches": len(matches),
            f"sm_{matcher_name}_global_avg": global_avg,
            f"sm_{matcher_name}_spec_avg": spec_avg,
            f"sm_{matcher_name}_time(s)": match_t,
        }

        records.append(
            {
                "Q": Q_name,
                "R": R_name,
                "#Q_schema": len(Q.columns),
                "#R_schema": len(R.columns),
                "task": task,
                "matcher": matcher_name,
                "sm_global_avg": global_avg,
                "sm_spec_avg": spec_avg,
                "#Q_req_columns": len(q_columns) if q_columns else None,
                "#R_req_columns": len(r_columns) if r_columns else None,
                "matches": len(matches),
                "time(s)": match_t,
            }
        )

        if len(records) >= 10:
            stats = pl.DataFrame(records, orient="row")
            include_header = not stats_filename.exists()
            with open(stats_filename, "a") as file:
                stats.write_csv(file, include_header=include_header, float_precision=5)
            records.clear()

    stats = pl.DataFrame(records, orient="row")
    include_header = not stats_filename.exists()
    with open(stats_filename, "a") as file:
        stats.write_csv(file, include_header=include_header, float_precision=5)

    with open(blend_matches_json.parent / "enriched_tasks_results.json", "w") as file:
        for blend_match in blend_matches:
            file.write(json.dumps(blend_match) + "\n")


def main():
    orqa_data_path = Path(os.environ["DATADIR"]) / "orqa" / "ckan" / "uk"
    assert orqa_data_path.exists()

    blend_matches_json = orqa_data_path / "candidates_discovery" / "tasks_results.json"
    datasets_path = orqa_data_path / "datasets" / "csv"
    statistics_path = orqa_data_path / "statistics"
    statistics_path.mkdir(exist_ok=True)

    evaluate_matches(blend_matches_json, datasets_path, statistics_path)


if __name__ == "__main__":
    main()
