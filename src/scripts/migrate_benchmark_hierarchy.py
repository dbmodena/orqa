"""
One-shot migration to the shared queries/results hierarchy:

    kind -> single_table | multi_table -> query id -> question number -> content

Converts, for every city found under the data directory:

- candidates_discovery/generated_queries.json from the legacy
  {model: {kind: {idx: {"data": {"queries": [...]}, ...}}}} shape: the
  entry-level fields (status, tokens, tables, ...) move under "_meta"
  (together with the model name) and each query becomes a numbered key.
- benchmark/<kind>/results/<qid>.json per-question files into a single
  benchmark/<kind>/results.json with the same hierarchy (minus the kind
  level, which is already the folder name); the old folder is kept as
  results.bak/.

Already-migrated files are detected and skipped, so the script is safe
to re-run. Originals are backed up next to the new files (*.bak).

    python src/scripts/migrate_benchmark_hierarchy.py [--data-dir DATADIR]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

KINDS = {"PANDAS", "SQL"}
SECTIONS = {"single_table", "multi_table"}
META_KEY = "_meta"


def _section_for(query_id: str) -> str:
    return "single_table" if str(query_id).startswith("st_") else "multi_table"


def _is_new_shape(payload: dict) -> bool:
    return bool(payload) and all(
        kind in KINDS and isinstance(sections, dict)
        and set(sections) <= SECTIONS
        for kind, sections in payload.items()
    )


def migrate_generated_queries(filepath: Path) -> bool:
    """Convert one generated_queries.json in place. Returns True if converted."""
    payload = json.loads(filepath.read_text(encoding="utf-8"))
    if _is_new_shape(payload):
        print(f"  {filepath}: already migrated, skipping")
        return False

    new: dict = {}
    for model, kinds in payload.items():
        for kind, entries in kinds.items():
            for query_id, entry in entries.items():
                data = entry.get("data") or {}
                queries = data.get("queries") or []
                meta = {
                    "model": model,
                    **{k: v for k, v in entry.items() if k != "data"},
                }
                extra = {k: v for k, v in data.items() if k != "queries"}
                if extra:
                    meta["result_extra"] = extra
                group = {META_KEY: meta, **{str(n): q for n, q in enumerate(queries)}}

                section = new.setdefault(kind, {}).setdefault(_section_for(query_id), {})
                if query_id in section:
                    print(
                        f"  {filepath}: WARNING id {query_id!r} ({kind}) exists for "
                        f"several models; keeping the first, dropping {model!r}"
                    )
                    continue
                section[query_id] = group

    backup = filepath.with_suffix(".json.bak")
    shutil.copy2(filepath, backup)
    filepath.write_text(
        json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  {filepath}: migrated (backup at {backup.name})")
    return True


def migrate_results_dir(results_dir: Path) -> bool:
    """Merge benchmark/<kind>/results/*.json into ../results.json."""
    results_filepath = results_dir.parent / "results.json"
    merged = (
        json.loads(results_filepath.read_text(encoding="utf-8"))
        if results_filepath.exists()
        else {}
    )

    for qfile in sorted(results_dir.glob("*.json")):
        entry = json.loads(qfile.read_text(encoding="utf-8"))
        flat_id = entry.get("id", qfile.stem)
        query_id = entry.get("query_id") or flat_id.rsplit("_", 1)[0]
        qnum = entry.get("question_number") or flat_id.rsplit("_", 1)[1]
        section = entry.get("section") or _section_for(query_id)
        entry.setdefault("section", section)
        entry.setdefault("query_id", query_id)
        entry.setdefault("question_number", qnum)
        merged.setdefault(section, {}).setdefault(str(query_id), {})[str(qnum)] = entry

    results_filepath.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    backup_dir = results_dir.parent / "results.bak"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    results_dir.rename(backup_dir)
    print(f"  {results_filepath}: merged {sum(len(g) for s in merged.values() for g in s.values())} "
          f"answered questions (old folder kept as {backup_dir.name}/)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATADIR", "").strip(),
        help="Base OrQA data directory (defaults to the DATADIR env variable).",
    )
    args = parser.parse_args()
    if not args.data_dir:
        parser.error("DATADIR is not set and --data-dir was not given.")
    data_dir = Path(args.data_dir)

    print("Migrating generated_queries.json files:")
    for filepath in sorted(data_dir.glob("**/candidates_discovery/generated_queries.json")):
        migrate_generated_queries(filepath)

    print("Migrating benchmark results folders:")
    for results_dir in sorted(data_dir.glob("**/benchmark/*/results")):
        if results_dir.is_dir():
            migrate_results_dir(results_dir)

    print("Done.")


if __name__ == "__main__":
    main()
