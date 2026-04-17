from __future__ import annotations

import argparse
import importlib
import os
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
    "generate-statements",
    "generate-response",
)


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


def load_cfg(spec: TargetSpec):
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")

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


def _step_callable_path(step: str, spec: TargetSpec) -> str:
    if step == "crawl":
        return f"orqa.crawling:crawl_{spec.target_id}"
    if step == "clean":
        cleaning_functions = {
            "ckan": "orqa.cleaning:ckan_cleaning",
            "socrata": "orqa.cleaning:socrata_cleaning",
            "ods": "orqa.cleaning:ods_cleaning",
        }
        return cleaning_functions[spec.backend]

    step_paths = {
        "normalize-metadata": "orqa.normalize_metadata:normalize_metadata",
        "index": "orqa.indexing:create_blend_index",
        "candidates-discovery": "orqa.candidates_generation:candidates_discovery",
        "generate-statements": "orqa.statement_generation:generate_statements",
        "generate-response": "orqa.statement_judge:generate_response",
    }
    return step_paths[step]


def run_steps(cfg, spec: TargetSpec, steps: Sequence[str]) -> None:
    for step in steps:
        step_callable = _import_callable(_step_callable_path(step, spec))
        step_callable(cfg)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        spec = resolve_target(args.country, args.city)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        cfg = load_cfg(spec)
        run_steps(cfg, spec, args.steps)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
