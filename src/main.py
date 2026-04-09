import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from conf import load_config
from orqa.cleaning import ckan_cleaning, socrata_cleaning

from orqa.crawling import crawl_canada, crawl_modena, crawl_nyc, crawl_uk, crawl_bologna, crawl_paris, crawl_madrid
from orqa.indexing import create_blend_index

from orqa.candidates_generation import candidates_discovery
from orqa.statement_generation import generate_statements
from orqa.statement_judge import generate_response

load_dotenv()

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
    # candidates_discovery(cfg)


def modena():
    modena_yaml_path = Path(
        os.path.dirname(__file__), "..", "conf", "workflow", "modena.yaml"
    )
    data_path = Path(os.environ["DATADIR"], "open_data", "ckan", "modena")

    data_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(modena_yaml_path, data_path)

    # crawl_modena(cfg)
    # create_blend_index(cfg)
    # candidates_discovery(cfg)


def nyc():
    nyc_yaml_path = Path(
        os.path.dirname(__file__), "..", "conf", "workflow", "nyc.yaml"
    )
    data_path = Path(os.environ["DATADIR"], "orqa", "socrata", "nyc")

    data_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(nyc_yaml_path, data_path)

    # crawl_nyc(cfg)
    # socrata_cleaning(cfg)
    # create_blend_index(cfg)
    # candidates_discovery(cfg)
    #generate_statements(cfg)
    # generate_response(cfg)


def bologna():
    bologna_yaml_path = Path(
        os.path.dirname(__file__), "..", "conf", "workflow", "bologna.yaml"
    )
    data_path = Path(os.environ["DATADIR"], "orqa", "ods", "bologna")

    data_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(bologna_yaml_path, data_path)
    crawl_bologna(cfg)

def madrid():
    madrid_yaml_path = Path(
        os.path.dirname(__file__), "..", "conf", "workflow", "madrid.yaml"
    )
    data_path = Path(os.environ["DATADIR"], "orqa", "ckan", "madrid")

    data_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(madrid_yaml_path, data_path)
    crawl_madrid(cfg)

def paris():
    paris_yaml_path = Path(
        os.path.dirname(__file__), "..", "conf", "workflow", "paris.yaml"
    )
    data_path = Path(os.environ["DATADIR"], "orqa", "ods", "paris")

    data_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(paris_yaml_path, data_path)
    crawl_paris(cfg)

def main():
    accepted = ["canada", "uk", "nyc", "modena","bologna","paris","madrid"]
    assert len(sys.argv) == 2, f"Usage is: python main.py <{' | '.join(accepted)}>"

    match sys.argv[1]:
        case "canada":
            canada()
        case "uk":
            uk()
        case "nyc":
            nyc()
        case "bologna":
            bologna()
        case "paris":
            paris()
        case "madrid":
            madrid()
        case _:
            raise ValueError(f"Usage is: python main.py <{' | '.join(accepted)}>")


if __name__ == "__main__":
    main()
