import os
from pathlib import Path

from conf import load_config

from orqa.crawling import crawl_canada
from orqa.indexing import create_blend_index


def canada():
    canada_yaml_path = Path(os.path.dirname(__file__), "..", "conf", "canada.yml")
    data_path = Path(os.environ["DATADIR"], "open_data", "ckan", "canada_small")

    data_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(canada_yaml_path, data_path)

    crawl_canada(cfg)
    create_blend_index(cfg)


def main():
    canada()


if __name__ == "__main__":
    main()
