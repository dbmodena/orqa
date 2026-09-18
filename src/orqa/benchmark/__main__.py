"""
Quick command-line access to the dataset reverse index, without going
through an MCP client:

    python -m orqa.benchmark --source socrata/nyc taxi license expiration
    python -m orqa.benchmark --list-sources

Also runs the retrievability report (verification for
``orqa.agent.utility.retrievability_gate``): re-runs the retriever panel and
the distinguishing-facet check against every question already generated for
a portal, and prints per-retriever, per-table-count coverage numbers.

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
    """Yield ``(kind, table_count, question_text, gold_resource_ids,
    stored_retrieval)`` for every generated question in ``queries_path``.
    """
    from orqa.benchmark.questions import iter_questions
    from orqa.utils import dataset_id_to_resource_id, load_json

    payload = load_json(queries_path)
    for kind in payload:
        for _section, _query_id, _qnum, question, _meta in iter_questions(payload, kind):
            question_text = question.get("question") or ""
            tables_map = question.get("tables") or {}
            gold_ids = [dataset_id_to_resource_id(str(v)) for v in tables_map.values()]
            if not question_text or not gold_ids:
                continue
            yield kind, len(gold_ids), question_text, gold_ids, question.get("retrieval")


def _run_retrievability_report(args) -> None:
    cfg = _load_cfg(args.country, args.city)

    from orqa.agent.utility.retrievability_gate import build_contract
    from orqa.benchmark.families import FamilyIndex, missing_facets
    from orqa.benchmark.index import load_index
    from orqa.benchmark.retrieval_panel import RetrieverPanel, make_llm_keyword_extractor
    from orqa.utils import load_json

    search_index = load_index(cfg)
    if search_index is None:
        print("No reverse index available for this portal (check tasks.mcp_search).")
        return

    with open(cfg.normalized_metadata_filepath, encoding="utf-8") as file:
        raw_records = json.load(file)
    family_index = FamilyIndex(raw_records, cfg.datasets_path)

    keyword_extractor = None
    if not args.no_llm:
        try:
            keyword_extractor = make_llm_keyword_extractor(cfg.llm_config_path / "litellm.yaml")
        except Exception as exc:
            print(f"(llm_keywords retriever unavailable: {exc}; continuing without it)")

    hybrid_index = search_index if hasattr(search_index, "semantic_search_many") else None
    panel = RetrieverPanel(
        search_index, hybrid_index=hybrid_index,
        keyword_extractor=keyword_extractor, family_index=family_index,
    )

    queries_path = Path(args.queries) if args.queries else cfg.statement_generation.queries_path
    if not queries_path.exists():
        print(f"No queries file found at {queries_path}.")
        return

    top_ks = [int(k.strip()) for k in args.top_ks.split(",") if k.strip()]
    contract_cfg = cfg.mcp_search.retrieval_contract

    # Per (retriever, table_count): running totals for FullCoverage@K / Hit@1.
    family_hits: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"n": 0, **{f"full@{k}": 0 for k in top_ks}, "hit@1": 0}
    )
    file_hits: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"n": 0, **{f"full@{k}": 0 for k in top_ks}, "hit@1": 0}
    )
    facet_total = 0
    facet_satisfied = 0
    accepted_larger = 0
    accepted_total = 0
    n_questions = 0

    for kind, n_tables, question_text, gold_ids, stored_retrieval in _iter_report_questions(queries_path):
        n_questions += 1
        rankings = panel.rank(question_text)
        for retriever, ranking in rankings.items():
            fkey = (retriever, n_tables)
            family_ranks = panel.family_ranks(ranking, gold_ids)
            resource_ranks = panel.resource_ranks(ranking, gold_ids)

            family_hits[fkey]["n"] += 1
            file_hits[fkey]["n"] += 1
            for k in top_ks:
                if all(r <= k for r in family_ranks.values()):
                    family_hits[fkey][f"full@{k}"] += 1
                if all(r <= k for r in resource_ranks.values()):
                    file_hits[fkey][f"full@{k}"] += 1
            if any(r == 1 for r in family_ranks.values()):
                family_hits[fkey]["hit@1"] += 1
            if any(r == 1 for r in resource_ranks.values()):
                file_hits[fkey]["hit@1"] += 1

        # Facet coverage: recompute this group's contract fresh (metadata
        # facets only — no data-scope pass, to keep the report a fast,
        # read-only pass over disk).
        plan_tables = [{"alias": f"Table_{i}", "resource_id": rid} for i, rid in enumerate(gold_ids)]
        contract = build_contract(
            plan_tables, family_index, search_index.get,
            top_k_per_table=contract_cfg.top_k_per_table,
            max_top_k=contract_cfg.max_top_k,
            min_agreement=contract_cfg.min_agreement,
            max_residual_siblings=contract_cfg.max_residual_siblings,
            single_table_top_k=contract_cfg.single_table_top_k,
        )
        for table in contract.tables:
            if not table.facets:
                continue
            facet_total += 1
            if not missing_facets(question_text, table.facets):
                facet_satisfied += 1

        if stored_retrieval:
            for table_info in (stored_retrieval.get("tables") or {}).values():
                accepted_total += 1
                if len(table_info.get("accepted_table_ids") or []) > 0:
                    accepted_larger += 1

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

    _print_table("Family-level coverage", family_hits)
    _print_table("File-level coverage", file_hits)

    if facet_total:
        print(f"Facet coverage: {facet_satisfied}/{facet_total} ({facet_satisfied / facet_total:.2%})")
    else:
        print("Facet coverage: no table in this file has a distinguishing facet.")

    if accepted_total:
        print(
            f"Accepted set larger than gold: {accepted_larger}/{accepted_total} "
            f"({accepted_larger / accepted_total:.2%}) — requires a queries file "
            "generated under the retrieval contract (stores a 'retrieval' block "
            "per query); older files report 0/0."
        )
    else:
        print(
            "Accepted set larger than gold: not available — this queries file "
            "carries no stored 'retrieval' block (generated before the "
            "retrieval contract, or the contract was disabled for that run)."
        )


if __name__ == "__main__":
    main()
