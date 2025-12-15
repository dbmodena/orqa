import sys
import os
from pathlib import Path

from conf import load_config

from orqa.crawling import crawl_canada, crawl_uk
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
    data_path = Path(os.environ["DATADIR"], "open_data", "ckan", "uk_small")

    data_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(uk_yaml_path, data_path)

    crawl_uk(cfg)
    create_blend_index(cfg)


def main():
    assert len(sys.argv) == 2, "Usage is: python main.py <canada | uk>"

    match sys.argv[1]:
        case "canada":
            canada()
        case "uk":
            uk()
        case _:
            raise ValueError("Usage is: python main.py <canada | uk>")


if __name__ == "__main__":
    main()
