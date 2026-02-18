import os
import sys
from pathlib import Path

from conf import load_config
from orqa.candidates_generation import candidates_discovery
from orqa.cleaning import ckan_cleaning, socrata_cleaning
from orqa.crawling import crawl_canada, crawl_modena, crawl_nyc, crawl_uk
from orqa.indexing import create_blend_index


def canada():
    canada_yaml_path = Path(
        os.path.dirname(__file__), "..", "conf", "workflow", "canada.yaml"
    )
    data_path = Path(os.environ["DATADIR"], "open_data", "ckan", "canada_small")

    data_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(canada_yaml_path, data_path)

    crawl_canada(cfg)
    create_blend_index(cfg)


def uk():
    uk_yaml_path = Path(os.path.dirname(__file__), "..", "conf", "workflow", "uk.yaml")
    data_path = Path(os.environ["DATADIR"], "orqa", "ckan", "uk")

    data_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(uk_yaml_path, data_path)

    # crawl_uk(cfg)
    # cleaning(cfg)
    # create_blend_index(cfg)
    candidates_discovery(cfg)


def modena():
    modena_yaml_path = Path(
        os.path.dirname(__file__), "..", "conf", "workflow", "modena.yaml"
    )
    data_path = Path(os.environ["DATADIR"], "open_data", "ckan", "modena")

    data_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(modena_yaml_path, data_path)

    # crawl_modena(cfg)
    # create_blend_index(cfg)
    candidates_discovery(cfg)


def nyc():
    nyc_yaml_path = Path(
        os.path.dirname(__file__), "..", "conf", "workflow", "nyc.yaml"
    )
    data_path = Path(os.environ["DATADIR"], "orqa", "socrata", "nyc")

    data_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(nyc_yaml_path, data_path)

    # crawl_nyc(cfg)
    # socrata_cleaning(cfg)
    create_blend_index(cfg)
    candidates_discovery(cfg)


def main():
    accepted = ["canada", "uk", "nyc", "modena"]
    assert len(sys.argv) == 2, f"Usage is: python main.py <{' | '.join(accepted)}>"

    match sys.argv[1]:
        case "canada":
            canada()
        case "uk":
            uk()
        case "nyc":
            nyc()
        case "modena":
            modena()
        case _:
            raise ValueError(f"Usage is: python main.py <{' | '.join(accepted)}>")


if __name__ == "__main__":
    main()
