import argparse
import os
import subprocess
from pathlib import Path


class Solr:
    def __init__(self, domain: str):
        if "SOLR_PATH" not in os.environ:
            raise OSError("env variable SOLR_PATH not found.")

        self._solr_path = Path(os.environ["SOLR_PATH"])
        if not self._solr_path.exists():
            raise FileNotFoundError(f"Path {self._solr_path} not exist.")

        self.domain = domain
        self._collection_conf_path = self._solr_path.joinpath(
            "server", "solr", self.domain, "conf"
        )

    def _exec(self, *args):
        subprocess.run(list(args))

    def start(self):
        self._exec("solr", "start", "--user-managed")

    def stop(self):
        self._exec("solr", "stop")

    def create_cluster(self):
        self._exec("solr", "create", "-c", self.domain)

    def delete_cluster(self):
        self._exec("solr", "delete", "-c", self.domain)

    def create_cluster_with_metadata(
        self,
        new_managed_schema_xml: Path,
        new_solrconfig_xml: Path,
        metadata_path: Path,
    ):
        self.start()
        self.create_cluster()
        self.stop()

        managed_schema_path = self._collection_conf_path.joinpath("managed-schema.xml")
        solrconfig_path = self._collection_conf_path.joinpath("solrconfig.xml")

        os.remove(managed_schema_path)
        os.remove(solrconfig_path)

        self._exec("cp", new_managed_schema_xml, managed_schema_path)
        self._exec("cp", new_solrconfig_xml, solrconfig_path)

        self.start()
        self.post_metadata(metadata_path)
        # self.stop()

    def post_metadata(self, metadata_path: Path):
        self._exec("solr", "post", "-c", self.domain, str(metadata_path.resolve()))


# --- domain helpers -----------------------------------------------------------

DOMAINS = ("nyc", "uk")

DOMAIN_CONFIG = {
    "nyc": {
        "data_subdir": "orqa/socrata/nyc",
        "metadata_file": "metadata/flat_metadata.json",
    },
    "uk": {
        "data_subdir": "orqa/ckan/uk",
        "metadata_file": "metadata/metadata.json",
    },
}


def _conf_dir(domain: str) -> Path:
    return (Path(__file__) / ".." / ".." / ".." / "conf" / "solr" / domain).resolve()


def _metadata_path(domain: str) -> Path:
    cfg = DOMAIN_CONFIG[domain]
    return Path(os.environ["DATADIR"]) / cfg["data_subdir"] / cfg["metadata_file"]


def cmd_create(domain: str) -> None:
    solr = Solr(domain)
    conf_dir = _conf_dir(domain)
    solr.create_cluster_with_metadata(
        new_managed_schema_xml=conf_dir / "managed-schema.xml",
        new_solrconfig_xml=conf_dir / "solrconfig.xml",
        metadata_path=_metadata_path(domain),
    )


def cmd_delete(domain: str) -> None:
    Solr(domain).delete_cluster()


def cmd_post(domain: str) -> None:
    Solr(domain).post_metadata(_metadata_path(domain))


def cmd_start(domain: str) -> None:
    Solr(domain).start()


def cmd_stop(domain: str) -> None:
    Solr(domain).stop()


# --- CLI ----------------------------------------------------------------------

COMMANDS = {
    "create": cmd_create,
    "delete": cmd_delete,
    "post": cmd_post,
    "start": cmd_start,
    "stop": cmd_stop,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solr_cli",
        description="Manage Solr clusters for supported domains.",
    )
    parser.add_argument(
        "command",
        choices=COMMANDS,
        help=(
            "create  – start Solr, create collection, apply custom config, index metadata\n"
            "delete  – delete the collection\n"
            "post    – (re-)index metadata into an existing collection\n"
            "start   – start the Solr server\n"
            "stop    – stop the Solr server"
        ),
    )
    parser.add_argument(
        "domain",
        choices=DOMAINS,
        help="Target domain / collection name.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    COMMANDS[args.command](args.domain)


if __name__ == "__main__":
    main()

