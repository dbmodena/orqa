"""
Quick command-line access to the dataset reverse index, without going
through an MCP client:

    python -m orqa.benchmark --source socrata/nyc taxi license expiration
    python -m orqa.benchmark --list-sources

Also runs the retrievability report (verification for
``orqa.agent.utility.retrievability_gate``): re-runs the retriever panel
against every question already generated for a portal, and prints
per-retriever, per-table-count coverage numbers.

    python -m orqa.benchmark --retrievability-report --country uk
    python -m orqa.benchmark --retrievability-report --country uk --no-llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from orqa.benchmark.index import Catalog


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the crawled open data datasets by keywords, "
        "or run the retrievability report."
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
    parser.add_argument(
        "--retrievability-report",
        action="store_true",
        help="Re-run the retriever panel over every already-generated "
        "question for --country/--city and print coverage stats.",
    )
    parser.add_argument("--country", help="Workflow target country (see src/main.py TARGETS).")
    parser.add_argument("--city", help="Workflow target city, when the country is city-scoped.")
    parser.add_argument(
        "--queries",
        help="Path to a generated_queries.json-shaped file (defaults to "
        "the workflow's own tasks.query_generation.queries_path).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the llm_keywords retriever (no BenchmarkSolverAgent call "
        "per question) for a free run.",
    )
    parser.add_argument(
        "--top-ks",
        default="10,20",
        help="Comma-separated K values to report FullCoverage@K for (default: 10,20).",
    )
    args = parser.parse_args()

    if args.retrievability_report:
        if not args.country:
            parser.error("--retrievability-report requires --country.")
        _run_retrievability_report(args)
        return

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


def _load_cfg(country: str, city: str | None):
    """Resolve and load an ``OrQAConfig`` the same way ``src/main.py`` does,
    reusing its TARGETS table rather than duplicating it here.
    """
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import main as main_module  # src/main.py

    spec = main_module.resolve_target(country, city)
    return main_module.load_cfg(spec)


def _iter_report_questions(queries_path: Path):
    """Yield ``(kind, table_count, question_text, gold_resource_ids)`` for
    every generated question in ``queries_path``.
    """
    from orqa.benchmark.questions import iter_questions
    from orqa.utils import dataset_id_to_resource_id, load_json

    payload = load_json(queries_path)
    for kind in payload:
        for _section, _query_id, _qnum, question, meta in iter_questions(payload, kind):
            question_text = question.get("question") or ""
            tables_map = meta.get("tables") or {}
            gold_ids = [dataset_id_to_resource_id(str(v)) for v in tables_map.values()]
            if not question_text or not gold_ids:
                continue
            yield kind, len(gold_ids), question_text, gold_ids


def _run_retrievability_report(args) -> None:
    cfg = _load_cfg(args.country, args.city)

    from orqa.benchmark.index import load_index

    search_index = load_index(cfg)
    if search_index is None:
        print("No reverse index available for this portal (check tasks.mcp_search).")
        return

    # The SAME panel the retrievability gate votes with — same retrievers,
    # same AND keyword search, same RRF — so this report measures what the
    # gate enforces instead of a differently configured retriever.
    from orqa.statement_generation import _build_gate_index, _build_retrieval_panel
    from orqa.benchmark.solr_index import SolrDatasetIndex

    gate_index = _build_gate_index(cfg, search_index)
    panel = _build_retrieval_panel(
        cfg, search_index,
        keyword_index=gate_index if gate_index is not search_index else None,
        universe=gate_index.resource_ids() if isinstance(gate_index, SolrDatasetIndex) else None,
        require_enabled=False,
        with_keywords=not args.no_llm,
    )

    queries_path = Path(args.queries) if args.queries else cfg.statement_generation.queries_path
    if not queries_path.exists():
        print(f"No queries file found at {queries_path}.")
        return

    top_ks = [int(k.strip()) for k in args.top_ks.split(",") if k.strip()]

    # Per (retriever, table_count): running totals for FullCoverage@K / Hit@1.
    hits: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"n": 0, **{f"full@{k}": 0 for k in top_ks}, "hit@1": 0}
    )
    n_questions = 0

    for kind, n_tables, question_text, gold_ids in _iter_report_questions(queries_path):
        n_questions += 1
        rankings = panel.rank(question_text)
        for retriever, ranking in rankings.items():
            key = (retriever, n_tables)
            ranks = panel.ranks(ranking, gold_ids)
            hits[key]["n"] += 1
            for k in top_ks:
                if all(r <= k for r in ranks.values()):
                    hits[key][f"full@{k}"] += 1
            if any(r == 1 for r in ranks.values()):
                hits[key]["hit@1"] += 1

    print(f"Retrievability report — {queries_path}")
    print(f"{n_questions} question(s), retrievers: {panel.retriever_names}\n")

    def _print_table(title: str, hits: dict[tuple[str, int], dict]) -> None:
        print(f"── {title} " + "─" * max(0, 60 - len(title)))
        for (retriever, n_tables), stats in sorted(hits.items()):
            n = stats["n"] or 1
            cov = "  ".join(f"Full@{k}={stats[f'full@{k}'] / n:.2f}" for k in top_ks)
            print(
                f"  {retriever:16s} tables={n_tables:<2d} n={stats['n']:<5d} "
                f"{cov}  Hit@1={stats['hit@1'] / n:.2f}"
            )
        print()

    _print_table("Coverage", hits)


if __name__ == "__main__":
    main()
