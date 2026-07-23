"""
Quick command-line access to the dataset reverse index, without going
through an MCP client:

    python -m orqa.benchmark --source socrata/nyc taxi license expiration
    python -m orqa.benchmark --list-sources
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from orqa.benchmark.index import Catalog


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the crawled open data datasets by keywords."
    )
    parser.add_argument("keywords", nargs="*", help="Keywords to search for.")
    parser.add_argument(
        "--data-dir",
        help="Base OrQA data directory (defaults to the DATADIR env variable).",
    )
    parser.add_argument(
        "--source",
        help='Restrict to one source, e.g. "socrata/nyc".',
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--only-available",
        action="store_true",
        help="Only return datasets whose CSV file exists on disk.",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List the discovered sources and exit.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or os.environ.get("DATADIR", "").strip()
    if not data_dir:
        parser.error("DATADIR is not set and --data-dir was not given.")

    catalog = Catalog(Path(data_dir))

    if args.list_sources:
        for source in catalog.sources:
            print(f"{source}\t{len(catalog.index(source))} datasets")
        return

    if not args.keywords:
        parser.error("Provide at least one keyword (or use --list-sources).")

    results = catalog.search(
        args.keywords, args.source, args.top_k, args.only_available
    )
    print(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
