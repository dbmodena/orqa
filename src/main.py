from __future__ import annotations
from conf import OrQAConfig

import argparse
import importlib
#import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

STEP_CHOICES = (
    "crawl",
    "clean",
    "normalize-metadata",
    "index",
    "candidates-discovery",
    "generate-query-candidates",
    "generate-statements",
    "mcp_search",
)

MCP_MODE_CHOICES = ("stdio", "port", "build")


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    country: str
    city: str | None
    backend: str
    workflow_path: Path
    relative_data_path: Path


TARGETS = (
    TargetSpec(
        target_id="canada",
        country="canada",
        city=None,
        backend="ckan",
        workflow_path=_PROJECT_ROOT / "conf" / "workflow" / "canada.yaml",
        relative_data_path=Path("open_data", "ckan", "canada_small"),
    ),
    TargetSpec(
        target_id="uk",
        country="uk",
        city=None,
        backend="ckan",
        workflow_path=_PROJECT_ROOT / "conf" / "workflow" / "uk.yaml",
        relative_data_path=Path("orqa", "ckan", "uk"),
    ),
    TargetSpec(
        target_id="modena",
        country="italy",
        city="modena",
        backend="ckan",
        workflow_path=_PROJECT_ROOT / "conf" / "workflow" / "modena.yaml",
        relative_data_path=Path("open_data", "ckan", "modena"),
    ),
    TargetSpec(
        target_id="bologna",
        country="italy",
        city="bologna",
        backend="ods",
        workflow_path=_PROJECT_ROOT / "conf" / "workflow" / "bologna.yaml",
        relative_data_path=Path("orqa", "ods", "bologna"),
    ),
    TargetSpec(
        target_id="madrid",
        country="spain",
        city="madrid",
        backend="ckan",
        workflow_path=_PROJECT_ROOT / "conf" / "workflow" / "madrid.yaml",
        relative_data_path=Path("orqa", "ckan", "madrid"),
    ),
    TargetSpec(
        target_id="valencia",
        country="spain",
        city="valencia",
        backend="ckan",
        workflow_path=_PROJECT_ROOT / "conf" / "workflow" / "valencia.yaml",
        relative_data_path=Path("orqa", "ckan", "valencia"),
    ),
    TargetSpec(
        target_id="paris",
        country="france",
        city="paris",
        backend="ods",
        workflow_path=_PROJECT_ROOT / "conf" / "workflow" / "paris.yaml",
        relative_data_path=Path("orqa", "ods", "paris"),
    ),
    TargetSpec(
        target_id="nyc",
        country="usa",
        city="nyc",
        backend="socrata",
        workflow_path=_PROJECT_ROOT / "conf" / "workflow" / "nyc.yaml",
        relative_data_path=Path("orqa", "socrata", "nyc"),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OrQA workflow steps for a specific country or city target.",
        epilog=(
            "Examples:\n"
            "  python src/main.py --country canada --steps crawl index\n"
            "  python src/main.py --country italy --city bologna --steps "
            "crawl clean normalize-metadata"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--country",
        required=True,
        help="Country target to run, such as canada, uk, italy, spain, france, or usa.",
    )
    parser.add_argument(
        "--city",
        help="City target when the selected country is city-scoped.",
    )
    parser.add_argument(
        "--steps",
        required=True,
        nargs="+",
        choices=STEP_CHOICES,
        help="One or more workflow steps to execute, in the order provided.",
    )
    parser.add_argument(
        "--mode",
        choices=MCP_MODE_CHOICES,
        default="stdio",
        help=(
            "Transport for the mcp_search step: 'stdio' for clients that "
            "spawn the server, 'port' to listen on the port set in the "
            "city's workflow yaml, 'build' to only materialize the index "
            "and exit."
        ),
    )
    return parser


def resolve_target(country: str, city: str | None) -> TargetSpec:
    normalized_country = country.strip().lower()
    normalized_city = city.strip().lower() if city is not None else None

    matches = [spec for spec in TARGETS if spec.country == normalized_country]
    if not matches:
        known_countries = ", ".join(sorted({spec.country for spec in TARGETS}))
        raise ValueError(
            f"Unknown country {country!r}. Supported countries: {known_countries}."
        )

    if normalized_city is None:
        target = next((spec for spec in matches if spec.city is None), None)
        if target is not None:
            return target

        available_cities = ", ".join(sorted(spec.city for spec in matches if spec.city))
        raise ValueError(
            f"Country {country!r} requires --city. Supported cities: {available_cities}."
        )

    target = next((spec for spec in matches if spec.city == normalized_city), None)
    if target is None:
        if any(spec.city is None for spec in matches):
            raise ValueError(
                f"Country {country!r} does not accept --city. Remove --city {city!r}."
            )

        available_cities = ", ".join(sorted(spec.city for spec in matches if spec.city))
        raise ValueError(
            f"Unknown city {city!r} for country {country!r}. "
            f"Supported cities: {available_cities}."
        )

    return target


def resolve_data_path(spec: TargetSpec) -> Path:
    data_dir = os.environ.get("DATADIR", "").strip()
    if not data_dir:
        raise RuntimeError(
            "DATADIR is not set. Define DATADIR to the base directory used for OrQA data."
        )
    return Path(data_dir) / spec.relative_data_path


def load_cfg(spec: TargetSpec, out=None) -> OrQAConfig:
    from dotenv import load_dotenv

    dotenv_path = _PROJECT_ROOT / ".env"
    print(f"Loading .env file from {dotenv_path}", file=out or sys.stdout)
    load_dotenv(dotenv_path)

    if not spec.workflow_path.exists():
        raise FileNotFoundError(f"Workflow config not found: {spec.workflow_path}")

    data_path = resolve_data_path(spec)
    data_path.mkdir(parents=True, exist_ok=True)

    conf_module = importlib.import_module("conf")
    return conf_module.load_config(spec.workflow_path, data_path)


def _import_callable(dotted_path: str):
    module_name, attr_name = dotted_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _step_callable_path(step: str, spec: TargetSpec, cfg) -> str:
    if step == "crawl":
        return f"orqa.crawling:crawl_{spec.target_id}"
    if step == "clean":
        cleaning_functions = {
            "ckan": "orqa.cleaning:ckan_cleaning",
            "socrata": "orqa.cleaning:socrata_cleaning",
            "ods": "orqa.cleaning:ods_cleaning",
        }
        return cleaning_functions[spec.backend]

    # The workflow yaml (tasks.candidates_discovery.method) decides which
    # discovery pipeline runs and which artifact lineage all steps use.
    discovery_method = cfg.candidates_discovery.method
    if step == "index" and discovery_method != "blend":
        raise ValueError(
            "The 'index' step builds the BLEND index, but this workflow "
            "sets tasks.candidates_discovery.method: semantic. Set the "
            "method to 'blend' to use the classical pipeline."
        )
    if step == "candidates-discovery":
        if discovery_method == "semantic":
            # embeddings + HNSW + Valentine
            return "orqa.embedding_discovery.pipeline:candidates_discovery"
        # BLEND index + SLOTH/Valentine verification
        return "orqa.candidates_generation:candidates_discovery"

    step_paths = {
        "normalize-metadata": "orqa.normalize_metadata:normalize_metadata",
        "index": "orqa.indexing:create_blend_index",
        "generate-query-candidates": "orqa.query_candidates:generate_query_candidates",
        "generate-statements": "orqa.statement_generation:generate_statements",
        "mcp_search": "orqa.benchmark.server:run_mcp_search",
    }
    return step_paths[step]


def run_steps(
    cfg, spec: TargetSpec, steps: Sequence[str], mcp_mode: str = "stdio"
) -> None:
    for step in steps:
        step_callable = _import_callable(_step_callable_path(step, spec, cfg))
        if step == "mcp_search":
            step_callable(cfg, mode=mcp_mode)
        else:
            step_callable(cfg)


def main(argv: Sequence[str] | None = None) -> int:
    #logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        spec = resolve_target(args.country, args.city)
    except ValueError as exc:
        parser.error(str(exc))

    # In stdio mode stdout carries the MCP JSON-RPC stream, so all
    # diagnostics must go to stderr.
    out = (
        sys.stderr
        if "mcp_search" in args.steps and args.mode == "stdio"
        else sys.stdout
    )

    try:
        cfg = load_cfg(spec, out=out)

        print(f" LOADED PATHS ".center(100, "="), file=out)
        print(f"Discovery method: {cfg.candidates_discovery.method}", file=out)
        print(f"Root data directory: {cfg.data_path}", file=out)
        print(f"Prompts folder: {cfg.prompts_path}", file=out)
        print(f"Skills folder: {cfg.skills_path}", file=out)
        print(f"LLM configurations folder: {cfg.llm_config_path}", file=out)
        print(f"Crawled datasets folder: {cfg.crawled_datasets_path}", file=out)
        print(f"Processed datasets folder: {cfg.datasets_path}", file=out)

        run_steps(cfg, spec, args.steps, mcp_mode=args.mode)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
